#!/usr/bin/env python3
"""Build the human-readable source-to-notes coverage map."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    lines = ["# 원문 절별 학습 자료 대응표", "", "[전체 목차](README.md) · [원문과 범위](SOURCE_NOTES.md)", "",
             "제공된 PDF 북마크의 절·하위절 182개를 학습 노트와 연결합니다. PDF 쪽은 파일상 페이지 번호입니다. "
             "연결 여부는 설명을 찾아갈 수 있다는 뜻이며, 원문의 모든 문장을 복제했다는 뜻은 아닙니다.", ""]
    total, covered = 0, 0
    for n in range(1, 13):
        folder = ROOT / "chapters" / f"ch{n:02d}"
        source = json.loads((folder / "source-map.json").read_text())
        cp = folder / "coverage.json"
        coverage = json.loads(cp.read_text()) if cp.exists() else {"sections": []}
        by_title = {(row["source_title"], row["pdf_page"]): row for row in coverage["sections"]}
        lines += [f"## {n:02d}장 {source['title_ko']}", "",
                  f"[학습 노트](chapters/ch{n:02d}/study.md) · 원문 PDF {source['pdf_start']}–{source['pdf_end']}쪽", "",
                  "| 원문 절 | 시작 PDF 쪽 | 학습 설명 |", "|---|---:|---|"]
        for section in source["sections"]:
            total += 1
            row = by_title.get((section["title"], section["pdf_page"]))
            if row and row.get("target"):
                covered += 1
                target = f"chapters/ch{n:02d}/{row['target']}"
                topics = " · ".join(row.get("covered_topics", [])[:3]).replace("|", "\\|")
                link = f"[설명]({target}) {topics}"
            else:
                link = "작성 중"
            lines.append(f"| {section['title'].replace('|', chr(92)+'|')} | {section['pdf_page']} | {link} |")
        figures = coverage.get("figures_and_tables", [])
        if figures:
            lines += ["", f"원문 그림·표 {len(figures)}개에 대한 위치와 설명 연결은 "
                      f"[장별 대응 데이터](chapters/ch{n:02d}/coverage.json)에 기록했습니다."]
        lines.append("")
    lines[6:6] = [f"절 연결: **{covered}/{total}**. 자세한 파일·링크 검사는 [검증 결과](VALIDATION.md)를 확인하세요.", ""]
    (ROOT / "COVERAGE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote COVERAGE.md: {covered}/{total} source sections")


if __name__ == "__main__":
    main()
