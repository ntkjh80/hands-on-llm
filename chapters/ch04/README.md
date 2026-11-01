# 4장 텍스트 분류

[전체 목차](../../README.md) · [학습 노트](study.md) · [강의안](lecture.md) · [강의 대본](lecture-script.md) · [시각화](visuals.html) · [실습 안내](lab-guide.md) · [복습 문제](quiz.md)

같은 리뷰를 분류하는 다섯 가지 경로를 배우는 장이다. 작업 특화 모델, 고정 임베딩과 작은 분류기, 라벨 설명 기반 제로샷, Flan-T5, 원문 API 모델을 비교한다. **무엇을 학습하고, 어떤 출력을 받고, 어떻게 평가하는가**를 중심으로 공부한다.

원문은 *Hands-On Large Language Models* (2024) 4장, 제공 PDF 물리 페이지 160–191이다. 원문 성능과 코드 예제는 집필 당시 사례이며 현재 서비스 상태나 직접 재측정한 성능을 뜻하지 않는다.

## 추천 학습 순서

1. [학습 노트](study.md)의 1–4절에서 데이터·모델 경로·혼동 행렬을 익힌다.
2. [시각화](visuals.html)에서 임계값을 움직이며 오류의 종류가 바뀌는 것을 확인한다.
3. 5–7절을 읽고 [실습](lab-guide.md)의 고정 특징·분류기·코사인을 계산한다.
4. 8–10절에서 생성 출력 검증과 API 방식의 평가 조건을 익힌다.
5. [복습 16문항](quiz.md)을 먼저 풀고 접힌 해설을 확인한다.

강의로 공부한다면 [85분 강의안](lecture.md)과 [설명 대본](lecture-script.md)을 함께 사용한다. 독학할 때도 대본을 읽으며 질문에 잠깐 멈춰 답해 보면 좋다.

## 실행과 파일

```bash
python3 chapters/ch04/lab.py
```

위 명령은 저장소 루트 기준이다. 표준 Python만 필요하고 실제 LLM·API·GPU는 사용하지 않는다. 작은 분류기를 학습하고 지표와 출력 검증을 실제 계산한다. [저장된 실행 결과](lab-output.txt)와 비교할 수 있다. 관찰용 데이터는 원문 벤치마크가 아니며 일반화 성능 추정용 분할도 아니다.

| 자료 | 역할 |
|---|---|
| [study.md](study.md) | 원문 전 절의 상세 설명, 수식, 예제, Mermaid와 SVG 도식 |
| [lecture.md](lecture.md) | 85분 수업 진행안, 판서, 활동과 기대 답 |
| [lecture-script.md](lecture-script.md) | 교사가 읽고 설명할 수 있는 강의 대본 |
| [visuals.html](visuals.html) | 임계값·코사인·출력 검증의 세 가지 조작 실험 |
| [lab.py](lab.py), [lab-guide.md](lab-guide.md) | 표준 Python 실습과 단계별 해설 |
| [quiz.md](quiz.md) | 16문항, 접힌 해설, 100점 채점 기준 |
| [coverage.json](coverage.json) | 원문 11절·그림 23개·결과표 5개의 설명 위치 |

Markdown과 SVG는 GitHub에서 읽을 수 있다. HTML 상호작용은 파일을 내려받아 브라우저에서 연다. 원본 PDF와 원문 페이지 렌더링은 게시용 자료에 포함하지 않는다.

[전체 목차](../../README.md) · [학습 노트](study.md) · [강의안](lecture.md) · [강의 대본](lecture-script.md) · [시각화](visuals.html) · [실습 안내](lab-guide.md) · [복습 문제](quiz.md)
