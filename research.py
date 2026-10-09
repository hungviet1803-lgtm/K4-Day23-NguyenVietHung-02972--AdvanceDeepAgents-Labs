"""research.py - The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import re
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

from agents import (FINALIZER_PATH, NOTES_DIR, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent,
                    read_sources, source_url_problems)
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator
RECURSION_LIMIT = 1000  # graph steps of the LEAD only (about 2 per model -> tool turn); subagents use SUB_LIMITS
REPAIR_ROUNDS = 2       # follow-ups sent to the lead when sources.json still has urls no tool returned


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(topic).lower())[:60].strip("-")
    return slug or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f"Topic: {topic}\n\n"
        f"Today is {date.today().isoformat()}, so 'the last two years' means since {date.today().year - 2}.\n"
        f"Research this topic and write the survey report to {REPORT_PATH}, following your process from planning to "
        f"validation. The run is finished only when `python3 {VALIDATOR_PATH}` prints OK."
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    Lead messages only: the subagents' tokens are not included, so this undercounts the real cost.
    """
    calls = Counter()
    tokens = {"input": 0, "output": 0}
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            calls[call["name"]] += 1
        usage = getattr(message, "usage_metadata", None) or {}
        tokens["input"] += usage.get("input_tokens", 0)
        tokens["output"] += usage.get("output_tokens", 0)
    return {
        "model": model_name,
        "elapsed_s": round(elapsed, 1),
        "subagent_calls": calls["task"],
        "tool_calls": dict(sorted(calls.items())),
        "tokens": tokens,
    }


def _parse_sources(sources_bytes):
    """The sources list from the downloaded bytes, or raise RuntimeError when it is missing or invalid."""
    if sources_bytes is None:
        raise RuntimeError(f"the agent produced no {SOURCES_PATH}")
    try:
        sources = json.loads(sources_bytes.decode("utf-8"))
    except ValueError as exc:  # UnicodeDecodeError is a ValueError too
        raise RuntimeError(f"{SOURCES_PATH} is not valid JSON: {exc}") from exc
    if not isinstance(sources, list) or not sources or not all(isinstance(s, dict) for s in sources):
        raise RuntimeError(f"{SOURCES_PATH} must be a non-empty JSON list of objects")
    return sources


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS, url_problems=source_url_problems):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    Raises RuntimeError and writes NOTHING when the report is missing/empty, sources.json is missing/invalid, or a
    source url was never returned by a search tool: a failed run must never leave a report that looks valid behind.
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes, sources_bytes = files.get(REPORT_PATH), files.get(SOURCES_PATH)
    if not report_bytes or not report_bytes.strip():
        raise RuntimeError(f"the agent produced no report ({REPORT_PATH} is missing or empty)")
    sources = _parse_sources(sources_bytes)
    problems = url_problems(sources)
    if problems:
        raise RuntimeError("urls not returned by any search tool:\n" + "\n".join(problems))
    meta = {
        "topic": topic,
        **summarize(messages, elapsed, model_name),
        "n_sources": len(sources),
        "source_families": sorted({str(s.get("source")) for s in sources if s.get("source")}),
    }
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)
    report_path = reports_dir / f"{slug}.md"
    # byte-for-byte what the sandbox holds (RUBRIC: the submitted report and sources are the downloaded ones)
    (reports_dir / f"{slug}.sources.json").write_bytes(sources_bytes)
    (reports_dir / f"{slug}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_bytes(report_bytes)
    return report_path


def _model_name(model):
    return getattr(model, "model_name", None) or getattr(model, "model", None) or type(model).__name__


def _log_progress(namespace, update, start):
    """Print progress to stderr: every tool call (lead or subagent), the output of the lead's shell commands and
    the lead's final answer, so a run can be followed and diagnosed."""
    who = "lead" if not namespace else "  sub"
    stamp = f"[{time.monotonic() - start:6.0f}s]"
    for node_update in (update or {}).values():
        messages = node_update.get("messages") if isinstance(node_update, dict) else None
        for message in messages if isinstance(messages, list) else []:
            calls = getattr(message, "tool_calls", None) or []
            for call in calls:
                args = call.get("args") or {}
                detail = args.get("subagent_type") or args.get("query") or args.get("url") or args.get("command") \
                    or args.get("file_path") or ""
                print(f"{stamp} {who} {call['name']} {str(detail)[:90]}", file=sys.stderr)
            if namespace:
                continue
            text = message.content if isinstance(message.content, str) else str(message.content)
            if getattr(message, "type", "") == "tool" and getattr(message, "name", "") in ("execute", "check_source_urls"):
                print(f"{stamp}      -> {text.strip()[:400]}", file=sys.stderr)
            elif getattr(message, "type", "") == "ai" and not calls and text.strip():
                print(f"{stamp} lead says: {text.strip()[:600]}", file=sys.stderr)


def run_agent(agent, messages, start):
    """Run the lead agent on `messages`, streaming progress, and return its final state (same as agent.invoke)."""
    state = None
    for namespace, mode, chunk in agent.stream({"messages": messages}, config={"recursion_limit": RECURSION_LIMIT},
                                               stream_mode=["updates", "values"], subgraphs=True):
        if mode == "values" and not namespace:
            state = chunk
        elif mode == "updates":
            _log_progress(namespace, chunk, start)
    if state is None:
        raise RuntimeError("the agent returned no state")
    return state


def repair_message(problems):
    """Follow-up for a lead that finished with urls no search tool returned (it may skip check_source_urls)."""
    return (f"{SOURCES_PATH} still has urls that no search tool returned in this run:\n" + "\n".join(problems) +
            "\nFor each one, use the exact url written in the notes or the tool output, or delete only that entry "
            "(do NOT renumber the others) and rewrite or remove the sentences that cite it. Then call "
            "check_source_urls until it answers OK, run the finalizer, and run the validator until it prints OK.")


def run_research(agent, backend, topic, start):
    """Run the lead; if sources.json still has unknown urls, send it back to fix them (deterministic feedback)."""
    state = run_agent(agent, [{"role": "user", "content": build_prompt(topic)}], start)
    for _ in range(REPAIR_ROUNDS):
        sources = read_sources(backend)
        problems = source_url_problems(sources) if sources else []
        if not problems:
            break
        print(f"[research] {len(problems)} unknown url(s): asking the lead to fix them", file=sys.stderr)
        state = run_agent(agent, [*state["messages"], {"role": "user", "content": repair_message(problems)}], start)
    return state


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic = (topic or "").strip()
    if not topic:
        print('usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    model = make_model()
    start = time.monotonic()
    with open_sandbox() as backend:  # the sandbox is always stopped and removed, even on errors
        setup = backend.execute(f"mkdir -p {NOTES_DIR} {WORKDIR}/report")
        if setup.exit_code != 0:
            print(f"FAILED: cannot prepare the sandbox workspace: {setup.output}", file=sys.stderr)
            return 1
        upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
        agent = build_lead_agent(backend, model)
        print(f"[research] topic: {topic!r} - running the lead agent (this takes a few minutes)", file=sys.stderr)
        try:
            result = run_research(agent, backend, topic, start)
            path = save_outputs(backend, topic, result["messages"], time.monotonic() - start, _model_name(model))
        except Exception as exc:  # noqa: BLE001 - a failed run (incl. GraphRecursionError) must exit non-zero
            print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    print(f"Report saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
