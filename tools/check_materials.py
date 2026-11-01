#!/usr/bin/env python3
"""Check chapter inventory, source coverage, anchors and local Markdown links."""
from pathlib import Path
from collections import Counter
from urllib.parse import unquote, urlsplit
import argparse
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["study.md", "lecture.md", "lecture-script.md", "visuals.html",
            "lab.py", "lab-guide.md", "lab-output.txt", "quiz.md",
            "source-map.json", "coverage.json", "status.json"]


def without_fences(text):
    result, fence = [], None
    for line in text.splitlines():
        match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence is None and match:
            fence = match.group(1)
        elif fence is not None and match and match.group(1)[0] == fence[0] and len(match.group(1)) >= len(fence) and not match.group(2).strip():
            fence = None
        elif fence is None:
            result.append(line)
    return "\n".join(result), fence


def github_slug(text):
    text = re.sub(r"<[^>]+>", "", text).strip().lower()
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
    return text.replace(" ", "-")


def anchors(path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".md":
        text, _ = without_fences(text)
    ids = set(re.findall(r'\bid=["\']([^"\']+)["\']', text))
    used = Counter()
    for heading in re.findall(r"^#{1,6}\s+(.+)$", text, re.M):
        slug = github_slug(heading)
        count = used[slug]
        used[slug] += 1
        ids.add(f"{slug}-{count}" if count else slug)
    return ids


def check_target(origin, target, errors):
    target = target.strip().strip("<>")
    if target.startswith(("http:", "https:", "mailto:", "data:", "codex:")):
        return
    if target.startswith("/"):
        errors.append(f"{origin.relative_to(ROOT)}: nonportable absolute link: {target}")
        return
    parsed = urlsplit(target)
    dest = (origin.parent / unquote(parsed.path)).resolve() if parsed.path else origin
    if not dest.exists():
        errors.append(f"{origin.relative_to(ROOT)}: missing target: {target}")
    elif parsed.fragment and dest.suffix.lower() in (".md", ".html", ".svg"):
        if unquote(parsed.fragment) not in anchors(dest):
            errors.append(f"{origin.relative_to(ROOT)}: missing anchor: {target}")


def audit():
    errors, chapters = [], []
    figure_path = ROOT / "source" / "figures.json"
    expected_figures = json.loads(figure_path.read_text()) if figure_path.exists() else []
    for n in range(1, 13):
        folder = ROOT / "chapters" / f"ch{n:02d}"
        missing = [name for name in REQUIRED if not (folder / name).is_file()]
        errors.extend(f"ch{n:02d}: missing {name}" for name in missing)
        row = {"chapter": n, "missing": missing, "source_sections": 0, "covered_sections": 0}
        try:
            source = json.loads((folder / "source-map.json").read_text())
            coverage = json.loads((folder / "coverage.json").read_text())
            expected = source["sections"]
            actual = coverage["sections"]
            counts = Counter((item.get("source_title"), item.get("pdf_page")) for item in actual)
            row["source_sections"] = len(expected)
            for item in expected:
                title = item["title"]
                key = (title, item["pdf_page"])
                if counts[key] != 1:
                    errors.append(f"ch{n:02d}: expected one coverage row for {key!r}, got {counts[key]}")
                    continue
                entry = next(x for x in actual if (x.get("source_title"), x.get("pdf_page")) == key)
                if entry.get("pdf_page") != item["pdf_page"]:
                    errors.append(f"ch{n:02d}: wrong source page for {title!r}")
                if not entry.get("covered_topics"):
                    errors.append(f"ch{n:02d}: no described topics for {title!r}")
                target = entry.get("target", "")
                if not target:
                    errors.append(f"ch{n:02d}: no target for {title!r}")
                else:
                    check_target(folder / "coverage.json", target, errors)
                row["covered_sections"] += 1
            extras = set(counts) - {(x["title"], x["pdf_page"]) for x in expected}
            if extras:
                errors.append(f"ch{n:02d}: unexpected source titles {sorted(extras)}")
            for item in coverage.get("figures_and_tables", []):
                if item.get("explained_in"):
                    check_target(folder / "coverage.json", item["explained_in"], errors)
            figure_index = {item.get("label"): item for item in coverage.get("figures_and_tables", [])}
            for figure in (x for x in expected_figures if x["chapter"] == n):
                entry = figure_index.get(figure["label"])
                if not entry:
                    errors.append(f"ch{n:02d}: missing figure/table coverage: {figure['label']}")
                elif entry.get("pdf_page") != figure["pdf_page"]:
                    errors.append(f"ch{n:02d}: incorrect figure/table page: {figure['label']}")
            row["figures_and_tables"] = len(coverage.get("figures_and_tables", []))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"ch{n:02d}: metadata not ready: {exc}")
        for name, key in [("study.md", "study_characters"), ("lecture-script.md", "script_characters")]:
            path = folder / name
            row[key] = len(path.read_text(encoding="utf-8")) if path.exists() else 0
        chapters.append(row)
    for path in sorted(ROOT.rglob("*.md")):
        if any(p in path.parts for p in (".git", "tmp", ".venv")):
            continue
        text = path.read_text(encoding="utf-8")
        # Only prose links: code examples are not actionable Markdown links.
        text, fence = without_fences(text)
        if fence:
            errors.append(f"{path.relative_to(ROOT)}: unclosed code fence")
        if len(re.findall(r"<details(?:\s[^>]*)?>", text)) != text.count("</details>"):
            errors.append(f"{path.relative_to(ROOT)}: unbalanced details tags")
        for target in re.findall(r"!?\[[^\]]*\]\(([^\n)]+)\)", text):
            target = re.sub(r'\s+["\'][^"\']*["\']$', "", target)
            check_target(path, target, errors)
    for path in sorted((ROOT / "chapters").glob("ch*/visuals.html")):
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r'\b(?:href|src)=["\']([^"\']+)["\']', text):
            check_target(path, target, errors)
    return {"passed": not errors, "chapters": chapters, "errors": errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable results")
    args = parser.parse_args()
    result = audit()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for row in result["chapters"]:
            print(f"ch{row['chapter']:02d}: {row['covered_sections']}/{row['source_sections']} sections, "
                  f"study {row['study_characters']:,} characters, {len(row['missing'])} missing files")
        for error in result["errors"]:
            print("ERROR:", error)
        print("PASS" if result["passed"] else "FAIL")
    sys.exit(0 if result["passed"] else 1)
