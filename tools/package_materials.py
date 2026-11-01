#!/usr/bin/env python3
"""Create a clean study bundle without the supplied book or private extraction."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = ["README.md", "LEARNING_PATH.md", "PREREQUISITES.md", "CONCEPTS.md",
               "ORIENTATION.md", "SOURCE_NOTES.md", "COVERAGE.md", "VALIDATION.md",
               ".gitignore", ".gitattributes"]
CHAPTER_FILES = ["README.md", "study.md", "lecture.md", "lecture-script.md", "visuals.html",
                 "lab.py", "lab-guide.md", "lab-output.txt", "quiz.md", "source-map.json",
                 "coverage.json", "status.json"]


def main():
    files = [ROOT / name for name in PUBLIC_ROOT]
    files += [ROOT / "source" / name for name in ("outline.json", "chapters.json", "figures.json")]
    files += [p for p in (ROOT / "assets").rglob("*") if p.is_file()]
    files += [p for p in (ROOT / "tools").iterdir() if p.is_file() and p.suffix in (".py", ".cjs", ".md")]
    for n in range(1, 13):
        folder = ROOT / "chapters" / f"ch{n:02d}"
        files += [folder / name for name in CHAPTER_FILES]
        if (folder / "visuals.md").exists():
            files.append(folder / "visuals.md")
        files += list(folder.glob("lab-*-output.txt"))
        if (folder / "assets").exists():
            files += [p for p in (folder / "assets").rglob("*") if p.is_file() and p.suffix in (".svg", ".css")]
    missing = [str(p.relative_to(ROOT)) for p in files if not p.is_file()]
    if missing:
        raise SystemExit("Bundle incomplete: " + ", ".join(missing))
    output = ROOT / "output" / "hands-on-llm-study.zip"
    output.parent.mkdir(exist_ok=True)
    files = sorted(set(files))
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p, Path("hands-on-llm-study") / p.relative_to(ROOT))
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise SystemExit(f"Corrupt zip entry: {bad}")
    print(f"Created {output.name}: {len(files)} files, {output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
