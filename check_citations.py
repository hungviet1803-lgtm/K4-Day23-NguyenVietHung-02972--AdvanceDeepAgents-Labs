"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

_CITATION = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")  # [3]  [1, 2]  [1-3]; not a Markdown link [3](url)
_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)         # citations inside code do not count
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_NEXT_HEADING = re.compile(r"(?m)^#{1,2}[ \t]")                # a heading after References ends the list
_REF_LINE = re.compile(r"(?m)^[ \t]*\[(\d+)\](.*)$")
_URL = re.compile(r"https?://[^\s<>]+")
# `source` is the tool family that returned the source; the grader checks that the url belongs to that family
_FAMILY_URL = {
    "arxiv": re.compile(r"https://arxiv\.org/abs/(\d{4}\.\d{4,5}|[a-z][a-z.\-]*/\d{7})"),  # no version suffix
    "hf-daily": re.compile(r"https://huggingface\.co/papers/[^/?#\s]+"),
    "hf-search": re.compile(r"https://huggingface\.co/papers/[^/?#\s]+"),
    "web": re.compile(r"https?://\S+"),
}
MIN_FAMILIES = 3  # RUBRIC 2.2: a report draws on at least 3 of the 4 families


def _numbers(group):
    """'1, 3-5' -> [1, 3, 4, 5]."""
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        else:
            numbers.append(int(part))
    return numbers


def _cited_numbers(body):
    return {n for match in _CITATION.finditer(_CODE.sub(" ", body)) for n in _numbers(match.group(1))}


def _urls(text):
    """URLs of a reference line, without trailing punctuation or an unbalanced closing bracket."""
    urls = []
    for url in _URL.findall(text):
        while url:
            if url[-1] in ".,;:":
                url = url[:-1]
            elif url[-1] == ")" and url.count(")") > url.count("("):
                url = url[:-1]
            else:
                break
        urls.append(url)
    return urls


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json"]
    problems, by_n, seen_urls = [], {}, {}
    for i, entry in enumerate(sources):
        if not isinstance(entry, dict):
            problems.append(f"sources.json entry #{i + 1} is not an object")
            continue
        n, url = entry.get("n"), entry.get("url")
        if not isinstance(n, int) or isinstance(n, bool):
            problems.append(f"sources.json entry #{i + 1}: n={n!r} is not an integer")
        elif n in by_n:
            problems.append(f"source number [{n}] appears twice in sources.json")
        else:
            by_n[n] = entry
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            problems.append(f"source [{n}]: url {url!r} does not start with http:// or https://")
        elif url in seen_urls:
            problems.append(f"source [{n}]: url {url} duplicates source [{seen_urls[url]}]")
        else:
            seen_urls[url] = n
            family = entry.get("source")
            if family not in _FAMILY_URL:
                problems.append(f"source [{n}]: source {family!r} is not one of {sorted(_FAMILY_URL)}")
            elif not _FAMILY_URL[family].fullmatch(url):
                problems.append(f"source [{n}]: url {url} does not match its source family {family!r} "
                                "(arxiv -> https://arxiv.org/abs/<id>, hf-* -> https://huggingface.co/papers/<id>, "
                                "else label it 'web')")

    families = {entry.get("source") for entry in by_n.values()} & _FAMILY_URL.keys()
    if len(families) < MIN_FAMILIES:
        problems.append(f"only {len(families)} source families {sorted(families)}: a report needs at least "
                        f"{MIN_FAMILIES} of {sorted(_FAMILY_URL)} (cite sources of the missing families)")

    headings = list(_REF_HEADING.finditer(report_text))
    if not headings:
        return problems + ["the report has no '## References' heading"]
    body = report_text[:headings[-1].start()]
    references = report_text[headings[-1].end():]
    after = _NEXT_HEADING.search(references)
    if after:
        references = references[:after.start()]

    cited = _cited_numbers(body)
    if not cited:
        problems.append("the report body cites no source")
    problems += [f"[{n}] cited but missing from sources.json" for n in sorted(cited - by_n.keys())]
    problems += [f"source [{n}] never cited in the report body" for n in sorted(by_n.keys() - cited)]

    lines = {}
    for match in _REF_LINE.finditer(references):
        n, rest = int(match.group(1)), match.group(2)
        if n in lines:
            problems.append(f"References: [{n}] has more than one line")
            continue
        lines[n] = rest
        if n not in by_n:
            problems.append(f"References: [{n}] is not a source in sources.json")
            continue
        urls = _urls(rest)
        if len(urls) != 1:
            problems.append(f"References: [{n}] must hold exactly one URL, found {len(urls)}")
        elif urls[0] != by_n[n].get("url"):
            problems.append(f"References: [{n}] url {urls[0]} differs from sources.json ({by_n[n].get('url')})")
    problems += [f"References: no line for source [{n}]" for n in sorted(by_n.keys() - lines.keys())]
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
