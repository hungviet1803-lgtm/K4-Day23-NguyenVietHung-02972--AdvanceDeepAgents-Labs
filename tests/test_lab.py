"""Offline unit tests (no network, no LLM, no sandbox):   python -m unittest discover tests"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agents  # noqa: E402
import research  # noqa: E402
import tools  # noqa: E402
from check_citations import check  # noqa: E402
from finalize_citations import finalize  # noqa: E402


class WithRetryTest(unittest.TestCase):
    def run_retry(self, outcomes, **kwargs):
        """outcomes: list of exceptions to raise or values to return, in call order."""
        calls, sleeps = [], []

        def fn():
            outcome = outcomes[len(calls)]
            calls.append(outcome)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        try:
            result = tools.with_retry(fn, sleep=sleeps.append, **kwargs)
        except Exception as exc:  # noqa: BLE001
            result = exc
        return result, len(calls), sleeps

    def test_returns_after_transient_failures(self):
        result, calls, sleeps = self.run_retry([tools.RetryableError("429"), tools.RetryableError("503"), "ok"],
                                               base=1.0, cap=30.0)
        self.assertEqual((result, calls, len(sleeps)), ("ok", 3, 2))

    def test_exponential_backoff_with_jitter_and_cap(self):
        _, _, sleeps = self.run_retry([tools.RetryableError("x")] * 5, attempts=5, base=1.0, cap=5.0)
        self.assertEqual(len(sleeps), 4)  # no sleep after the last attempt
        for attempt, delay in enumerate(sleeps):
            backoff = 2 ** attempt
            self.assertGreaterEqual(delay, min(backoff, 5.0))
            self.assertLessEqual(delay, min(2 * backoff, 5.0))

    def test_retry_after_is_honoured_and_capped(self):
        _, _, sleeps = self.run_retry([tools.RetryableError("x", retry_after=7), tools.RetryableError("x", 99), "ok"],
                                      cap=30.0)
        self.assertEqual(sleeps, [7, 30.0])

    def test_gives_up_and_reraises(self):
        result, calls, sleeps = self.run_retry([tools.RetryableError("x")] * 3, attempts=3)
        self.assertIsInstance(result, tools.RetryableError)
        self.assertEqual((calls, len(sleeps)), (3, 2))

    def test_other_errors_are_not_retried(self):
        result, calls, sleeps = self.run_retry([KeyError("bug"), "never"])
        self.assertIsInstance(result, KeyError)
        self.assertEqual((calls, sleeps), (1, []))


class ToolParsingTest(unittest.TestCase):
    def test_exa_rate_limit_flag_is_retryable(self):
        answer = {"result": {"_meta": {"ai.exa/rateLimited": True}, "content": [{"type": "text", "text": "limit"}]}}
        with self.assertRaises(tools.RetryableError):
            tools._exa_text(answer)

    def test_exa_jsonrpc_error_raises(self):
        with self.assertRaises(RuntimeError):
            tools._exa_text({"error": {"code": -32602, "message": "bad arguments"}})

    def test_exa_text_joins_text_parts(self):
        answer = {"result": {"content": [{"type": "text", "text": "a"}, {"type": "image"}, {"type": "text", "text": "b"}]}}
        self.assertEqual(tools._exa_text(answer), "a\nb")

    def test_key_is_redacted_from_errors(self):
        self.assertNotIn("SECRET123", tools._error(ValueError("GET https://x/mcp?exaApiKey=SECRET123"), "SECRET123"))

    def test_arxiv_entry_normalised(self):
        xml = """<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2501.00001v3</id>
        <published>2025-01-02T10:00:00Z</published><title>A   title
        on two lines</title><summary>  Some
        summary. </summary></entry></feed>"""
        self.assertEqual(tools._arxiv_records(xml), [{"id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001",
                                                     "published": "2025-01-02", "title": "A title on two lines",
                                                     "summary": "Some summary."}])

    def test_arxiv_empty_query_does_not_call_network(self):
        self.assertEqual(tools.arxiv_search.invoke({"query": ' :"" AND OR '}), "NO RESULTS")

    def test_hf_record_skips_items_without_id_and_prefers_ai_summary(self):
        self.assertIsNone(tools._hf_record({"paper": {"title": "no id"}}))
        record = tools._hf_record({"paper": {"id": "2402.1", "title": "T", "summary": "long", "ai_summary": "short",
                                             "upvotes": 4, "githubRepo": "https://github.com/a/b", "githubStars": 9,
                                             "publishedAt": "2024-02-13T07:47:36.000Z"}})
        self.assertEqual(record, {"id": "2402.1", "url": "https://huggingface.co/papers/2402.1",
                                  "published": "2024-02-13", "title": "T", "summary": "short", "upvotes": 4,
                                  "github": "https://github.com/a/b", "stars": 9})


class ProvenanceTest(unittest.TestCase):
    def test_only_urls_returned_by_tools_are_known(self):
        real = "https://proceedings.neurips.cc/paper_files/paper/2024/file/x.pdf"
        tools._remember(tools._urls_in_text(f"Title: T\nURL: {real}\nPublished: 2024\n"))
        mangled = real.replace("paper_files/paper/", "paper_files/")  # a copy that dropped a few characters
        sources = [{"n": 1, "url": real}, {"n": 2, "url": mangled}]
        self.assertEqual(agents.source_url_problems(sources),
                         [f"source [2]: {mangled} was never returned by a search tool (mistyped or invented)"])


class MemoryBackend:
    """In-memory stand-in for the sandbox: just enough for download_files / upload_files."""

    def __init__(self, files):
        self.files = dict(files)

    def download_files(self, paths):
        return [SimpleNamespace(path=p, content=self.files.get(p)) for p in paths]

    def upload_files(self, files):
        self.files.update(files)


class SourceNumberGuardTest(unittest.TestCase):
    ORIGINAL = [{"n": 1, "url": "https://a"}, {"n": 2, "url": "https://b"}, {"n": 3, "url": "https://c"}]

    def call(self, new_sources, command="python3 - <<'PY'"):
        path = agents.SOURCES_PATH
        backend = MemoryBackend({path: json.dumps(self.ORIGINAL).encode()})
        guard = agents.SourceNumberGuard(backend)

        def handler(request):  # the tool call rewrites sources.json, like a python script run by the lead
            backend.files[path] = json.dumps(new_sources).encode()
            return agents.ToolMessage(content="ok", tool_call_id="1")

        request = SimpleNamespace(tool_call={"name": "execute", "args": {"command": command}})
        result = guard.wrap_tool_call(request, handler)
        return result.content, json.loads(backend.files[path])

    def test_renumbering_is_reverted(self):
        renumbered = [{"n": 1, "url": "https://a"}, {"n": 2, "url": "https://c"}]  # dropped b and shifted c
        content, sources = self.call(renumbered)
        self.assertIn("REVERTED", content)
        self.assertEqual(sources, self.ORIGINAL)

    def test_deleting_adding_and_relabelling_are_allowed(self):
        edited = [{"n": 1, "url": "https://a", "source": "web"}, {"n": 3, "url": "https://c"}, {"n": 4, "url": "https://d"}]
        content, sources = self.call(edited)
        self.assertEqual((content, sources), ("ok", edited))

    def test_the_finalizer_may_renumber(self):
        renumbered = [{"n": 1, "url": "https://a"}, {"n": 2, "url": "https://c"}]
        content, sources = self.call(renumbered, command=f"python3 {agents.FINALIZER_PATH}")
        self.assertEqual((content, sources), ("ok", renumbered))


SOURCES = [
    {"n": 1, "id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001", "title": "A", "date": "2025-01-02",
     "source": "arxiv"},
    {"n": 2, "id": "2502.1", "url": "https://huggingface.co/papers/2502.1", "title": "B", "date": "2025-02-01",
     "source": "hf-search"},
    {"n": 3, "id": "w", "url": "https://en.wikipedia.org/wiki/World_model_(AI)", "title": "C", "date": "",
     "source": "web"},
]
BODY = "# T\n\n## TL;DR\n- x [1, 2] and y [3]. Code `[9]` and link [4](http://x) are ignored.\n"


class CheckCitationsTest(unittest.TestCase):
    def setUp(self):
        self.report, self.sources, problems = finalize(BODY, SOURCES)
        self.assertEqual(problems, [])

    def test_finalized_report_is_ok(self):
        self.assertEqual(check(self.report, self.sources), [])

    def test_grouped_citations_are_expanded(self):
        report = self.report.replace("[1][2] and y [3]", "[1-3] and y")
        self.assertEqual(check(report, self.sources), [])

    def assertProblem(self, report, sources, fragment):
        problems = check(report, sources)
        self.assertTrue(any(fragment in p for p in problems), problems)

    def test_rules(self):
        self.assertProblem(self.report, [], "no sources")
        self.assertProblem(BODY, self.sources, "no '## References'")
        self.assertProblem(self.report.replace("[3]", "[7]", 1), self.sources, "[7] cited but missing")
        self.assertProblem(self.report.replace("[3]", "", 1), self.sources, "source [3] never cited")
        self.assertProblem(self.report.replace("[3] C. web.", "[3] C; D https://x.org/d. web."), self.sources,
                           "exactly one URL")
        self.assertProblem(self.report.rsplit("\n[3]", 1)[0], self.sources, "no line for source [3]")
        self.assertProblem(self.report + "[2] B. https://huggingface.co/papers/2502.1\n", self.sources,
                           "more than one line")
        self.assertProblem(self.report + "[5] Z. https://z.org\n", self.sources, "[5] is not a source")
        self.assertProblem(self.report.replace("papers/2502.1 ", "papers/9 "), self.sources, "differs")
        bad = [{"n": "1", "url": "https://a"}, {"n": 2, "url": "ftp://b"}, {"n": 3, "url": "https://a"}]
        self.assertProblem(self.report, bad, "not an integer")
        self.assertProblem(self.report, bad, "does not start with http")
        self.assertProblem(self.report, bad, "duplicates source")

    def test_source_family_must_match_url(self):
        def with_entry(**changes):
            return [dict(self.sources[0], **changes)] + self.sources[1:]
        self.assertProblem(self.report, with_entry(url="https://arxiv.org/html/2501.00001"), "does not match")
        self.assertProblem(self.report, with_entry(url="https://arxiv.org/abs/2501.00001v2"), "does not match")
        self.assertProblem(self.report, with_entry(source="hf-daily"), "does not match")
        self.assertProblem(self.report, with_entry(source="blog"), "is not one of")
        self.assertEqual(check(self.report.replace("2501.00001", "cs/0101001"),
                               with_entry(url="https://arxiv.org/abs/cs/0101001")), [])

    def test_at_least_three_source_families(self):
        two_families = [dict(s, source="web") if s["source"] == "hf-search" else s for s in self.sources]
        two_families[1]["url"] = "https://example.org/b"
        report = self.report.replace("https://huggingface.co/papers/2502.1", "https://example.org/b")
        self.assertProblem(report, two_families, "only 2 source families")

    def test_reference_numbers_are_not_citations(self):
        body_without_3 = self.report.replace(" and y [3]", "")
        self.assertProblem(body_without_3, self.sources, "source [3] never cited")


class FakeBackend:
    def __init__(self, files):
        self.files = files

    def download_files(self, paths):
        return [SimpleNamespace(path=p, content=self.files.get(p)) for p in paths]


class ResearchTest(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(research.slugify("Survey about World Model"), "survey-about-world-model")
        self.assertEqual(research.slugify("../../x"), "x")
        self.assertEqual(research.slugify(""), "topic")
        self.assertEqual(research.slugify("?!/"), "topic")
        self.assertLessEqual(len(research.slugify("a" * 200)), 60)
        self.assertNotIn("/", research.slugify("a/b\\c"))

    def test_summarize_counts_lead_calls_and_tokens(self):
        messages = [
            SimpleNamespace(tool_calls=[{"name": "write_todos"}], usage_metadata={"input_tokens": 10, "output_tokens": 2}),
            SimpleNamespace(tool_calls=[{"name": "task"}, {"name": "task"}, {"name": "task"}], usage_metadata=None),
            SimpleNamespace(content="tool result"),
        ]
        self.assertEqual(research.summarize(messages, 12.345, "m"), {
            "model": "m", "elapsed_s": 12.3, "subagent_calls": 3,
            "tool_calls": {"task": 3, "write_todos": 1}, "tokens": {"input": 10, "output": 2}})

    def save(self, files, url_problems=lambda sources: []):
        reports = Path(tempfile.mkdtemp())
        try:
            path = research.save_outputs(FakeBackend(files), "My topic", [], 1.0, "m", reports_dir=reports,
                                         url_problems=url_problems)
        except RuntimeError:
            path = None
        return path, sorted(p.name for p in reports.iterdir())

    def test_unknown_urls_fail_the_run(self):
        files = {research.REPORT_PATH: b"# r", research.SOURCES_PATH: json.dumps(SOURCES).encode()}
        self.assertEqual(self.save(files, url_problems=lambda sources: ["source [1]: bad url"]), (None, []))

    def test_failed_runs_write_nothing(self):
        good_sources = json.dumps(SOURCES).encode()
        for files in ({}, {research.REPORT_PATH: b"  \n", research.SOURCES_PATH: good_sources},
                      {research.REPORT_PATH: b"# r", research.SOURCES_PATH: None},
                      {research.REPORT_PATH: b"# r", research.SOURCES_PATH: b"{not json"},
                      {research.REPORT_PATH: b"# r", research.SOURCES_PATH: b"[]"}):
            self.assertEqual(self.save(files), (None, []))

    def test_successful_run_writes_the_downloaded_bytes_and_meta(self):
        sources_bytes = json.dumps(SOURCES, indent=1).encode()
        path, names = self.save({research.REPORT_PATH: b"# report\n", research.SOURCES_PATH: sources_bytes})
        self.assertEqual(names, ["my-topic.md", "my-topic.meta.json", "my-topic.sources.json"])
        self.assertEqual(path.read_bytes(), b"# report\n")
        self.assertEqual((path.parent / "my-topic.sources.json").read_bytes(), sources_bytes)
        meta = json.loads((path.parent / "my-topic.meta.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["topic"], meta["n_sources"], meta["source_families"]),
                         ("My topic", 3, ["arxiv", "hf-search", "web"]))

    def test_lead_is_sent_back_while_urls_are_unknown(self):
        sources_seen = [[{"n": 1, "url": "https://bad.example/x"}], [{"n": 1, "url": "https://bad.example/x"}], []]
        calls = []
        original = (research.run_agent, research.read_sources, research.source_url_problems)
        research.run_agent = lambda agent, messages, start: calls.append(messages) or {"messages": messages}
        research.read_sources = lambda backend: sources_seen[len(calls) - 1]
        research.source_url_problems = lambda sources: [f"bad {s['url']}" for s in sources]
        try:
            research.run_research(None, None, "t", 0.0)
        finally:
            research.run_agent, research.read_sources, research.source_url_problems = original
        self.assertEqual(len(calls), 1 + research.REPAIR_ROUNDS)  # first run + one follow-up per round still broken
        self.assertIn("https://bad.example/x", calls[1][-1]["content"])

    def test_empty_topic_exits_2(self):
        self.assertEqual(research.main("   "), 2)


if __name__ == "__main__":
    unittest.main()
