# 5장 텍스트 군집화와 토픽 모델링

[전체 목차](../../README.md) · [학습 노트](study.md) · [강의안](lecture.md) · [강의 대본](lecture-script.md) · [시각화](visuals.html) · [실습](lab-guide.md) · [문제와 해설](quiz.md)

문서를 의미상 비슷한 묶음으로 나누고, 그 묶음의 키워드와 이름을 만드는 과정을 학습한다. 기본 문서는 모두 Markdown이며, GitHub에서 표·수식·Mermaid 도식과 SVG를 읽을 수 있다. 조작 가능한 시각화는 HTML 파일을 내려받아 브라우저에서 연다. GitHub의 일반 파일 미리보기에서는 HTML 상호작용이 실행되지 않는다.

| 자료 | 내용 |
|---|---|
| [학습 노트](study.md) | 원문 11개 절에 대응하는 상세 설명, 계산, 오해, 그림·표 색인 |
| [강의안](lecture.md) | 활동을 포함한 85분 수업 구성과 판서·질문 |
| [강의 대본](lecture-script.md) | 실제 설명 문장, 실습 진행, 예상 질문 |
| [시각화](visuals.html) | 밀도·c-TF-IDF·MMR의 세 가지 조작 실험 |
| [실습 안내](lab-guide.md) | 실행법, 코드 해설, 변경 실험, 실제 모델 확장 절차 |
| [Python 실습](lab.py) · [실행 기록](lab-output.txt) | 표준 라이브러리로 직접 계산한 결과 |
| [복습 문제](quiz.md) | 개념·손계산·디버깅·응용 설계 18문제와 해설 |
| [범위 대응표](coverage.json) | 원문 절·24개 그림·5개 결과 표의 설명 위치 |

저장소 루트에서 시작한다.

```bash
python3 chapters/ch05/lab.py --check
```

이 실습은 가상 문서와 좌표로 DBSCAN, 기본 c-TF-IDF, MMR을 계산한다. 실제 임베딩 모델, UMAP, HDBSCAN, BERTopic, 생성 모델을 실행한 결과와 구분한다. 원문은 Jay Alammar·Maarten Grootendorst, *Hands-On Large Language Models* (2024), Chapter 5, PDF 물리 페이지 192–235다.

처음 공부한다면 학습 노트 → 시각화 → 실습 → 문제 순서로 진행한다. 이 챗은 5장 전용으로 이어서 질문하고 풀이를 점검하는 데 사용할 수 있다.

[전체 목차](../../README.md) · [학습 노트](study.md) · [강의안](lecture.md) · [강의 대본](lecture-script.md) · [시각화](visuals.html) · [실습](lab-guide.md) · [문제와 해설](quiz.md)
