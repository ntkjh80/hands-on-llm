# Hands-On Large Language Models 한국어 학습 자료

Jay Alammar와 Maarten Grootendorst의 *Hands-On Large Language Models* (O’Reilly, 2024)를 공부하기 위한 챕터별 학습 노트, 강의 진행안, 강의 대본, 시각화, Python 실습과 복습 문제 모음입니다. 본문을 그대로 옮기지 않고, 각 개념의 목적과 작동 원리, 작은 예제, 혼동하기 쉬운 지점을 한국어로 설명합니다.

기본 문서는 **Markdown**입니다. GitHub에서 노트와 Mermaid 도식을 읽고, 파일을 내려받아 시각화와 실습을 실행할 수 있습니다. 이 자료의 페이지 표시는 제공된 598쪽 PDF의 **파일상 페이지 번호**입니다. 인쇄본 쪽수와 다를 수 있습니다.

## 시작하기

1. [서문과 후기의 학습 방향](ORIENTATION.md), [학습 방법과 12주 계획](LEARNING_PATH.md)을 읽습니다. Python·벡터·확률이 낯설면 [선수지식](PREREQUISITES.md)을 먼저 봅니다.
2. 해당 장의 **학습 노트**를 읽고 도식의 흐름을 자기 말로 설명합니다.
3. **시각화**에서 값을 바꾸기 전에 결과를 예측하고, 실제 변화와 비교합니다.
4. **실습 안내**에 따라 실행하고, 입력을 바꿔 결과가 왜 달라지는지 설명합니다.
5. **문제**를 먼저 풀고 해설을 확인합니다. 설명이 막히는 부분은 해당 챕터 전용 챗에서 질문합니다.

## 챕터별 목차

### 1부 언어 모델의 이해

| 장 | 학습 노트 | 강의 자료 | 실습 및 확인 | 원문 PDF 범위 |
|---|---|---|---|---|
| 01 언어 모델 입문 | [노트](chapters/ch01/study.md) | [강의안](chapters/ch01/lecture.md) · [대본](chapters/ch01/lecture-script.md) | [시각화](chapters/ch01/visuals.html) · [실습](chapters/ch01/lab-guide.md) · [문제](chapters/ch01/quiz.md) | 23–66 |
| 02 토큰과 임베딩 | [노트](chapters/ch02/study.md) | [강의안](chapters/ch02/lecture.md) · [대본](chapters/ch02/lecture-script.md) | [시각화](chapters/ch02/visuals.html) · [실습](chapters/ch02/lab-guide.md) · [문제](chapters/ch02/quiz.md) | 67–115 |
| 03 트랜스포머 내부 구조 | [노트](chapters/ch03/study.md) | [강의안](chapters/ch03/lecture.md) · [대본](chapters/ch03/lecture-script.md) | [시각화](chapters/ch03/visuals.html) · [실습](chapters/ch03/lab-guide.md) · [문제](chapters/ch03/quiz.md) | 116–159 |

### 2부 사전 학습 언어 모델의 활용

| 장 | 학습 노트 | 강의 자료 | 실습 및 확인 | 원문 PDF 범위 |
|---|---|---|---|---|
| 04 텍스트 분류 | [노트](chapters/ch04/study.md) | [강의안](chapters/ch04/lecture.md) · [대본](chapters/ch04/lecture-script.md) | [시각화](chapters/ch04/visuals.html) · [실습](chapters/ch04/lab-guide.md) · [문제](chapters/ch04/quiz.md) | 160–191 |
| 05 텍스트 군집화와 토픽 모델링 | [노트](chapters/ch05/study.md) | [강의안](chapters/ch05/lecture.md) · [대본](chapters/ch05/lecture-script.md) | [시각화](chapters/ch05/visuals.html) · [실습](chapters/ch05/lab-guide.md) · [문제](chapters/ch05/quiz.md) | 192–235 |
| 06 프롬프트 엔지니어링 | [노트](chapters/ch06/study.md) | [강의안](chapters/ch06/lecture.md) · [대본](chapters/ch06/lecture-script.md) | [시각화](chapters/ch06/visuals.html) · [실습](chapters/ch06/lab-guide.md) · [문제](chapters/ch06/quiz.md) | 236–279 |
| 07 고급 텍스트 생성과 도구 | [노트](chapters/ch07/study.md) | [강의안](chapters/ch07/lecture.md) · [대본](chapters/ch07/lecture-script.md) | [시각화](chapters/ch07/visuals.html) · [실습](chapters/ch07/lab-guide.md) · [문제](chapters/ch07/quiz.md) | 280–315 |
| 08 시맨틱 검색과 RAG | [노트](chapters/ch08/study.md) | [강의안](chapters/ch08/lecture.md) · [대본](chapters/ch08/lecture-script.md) | [시각화](chapters/ch08/visuals.html) · [실습](chapters/ch08/lab-guide.md) · [문제](chapters/ch08/quiz.md) | 316–359 |
| 09 멀티모달 언어 모델 | [노트](chapters/ch09/study.md) | [강의안](chapters/ch09/lecture.md) · [대본](chapters/ch09/lecture-script.md) | [시각화](chapters/ch09/visuals.html) · [실습](chapters/ch09/lab-guide.md) · [문제](chapters/ch09/quiz.md) | 360–398 |

### 3부 언어 모델의 학습과 미세 조정

| 장 | 학습 노트 | 강의 자료 | 실습 및 확인 | 원문 PDF 범위 |
|---|---|---|---|---|
| 10 텍스트 임베딩 모델 만들기 | [노트](chapters/ch10/study.md) | [강의안](chapters/ch10/lecture.md) · [대본](chapters/ch10/lecture-script.md) | [시각화](chapters/ch10/visuals.html) · [실습](chapters/ch10/lab-guide.md) · [문제](chapters/ch10/quiz.md) | 399–445 |
| 11 표현 모델 미세 조정 | [노트](chapters/ch11/study.md) | [강의안](chapters/ch11/lecture.md) · [대본](chapters/ch11/lecture-script.md) | [시각화](chapters/ch11/visuals.html) · [실습](chapters/ch11/lab-guide.md) · [문제](chapters/ch11/quiz.md) | 446–486 |
| 12 생성 모델 미세 조정 | [노트](chapters/ch12/study.md) | [강의안](chapters/ch12/lecture.md) · [대본](chapters/ch12/lecture-script.md) | [시각화](chapters/ch12/visuals.html) · [실습](chapters/ch12/lab-guide.md) · [문제](chapters/ch12/quiz.md) | 487–535 |

범위 끝에 있는 주석·참고문헌·부 구분 페이지도 읽기 범위에 포함합니다. 각 절의 실제 시작 페이지는 장별 `source-map.json`과 `coverage.json`으로 확인할 수 있습니다.

## 자료 구성

```text
chapters/ch01/  ...  chapters/ch12/
├── study.md           # 원문 절별 상세 설명과 도식
├── lecture.md         # 수업 진행안, 판서, 질문과 활동
├── lecture-script.md  # 교사의 설명 대본
├── visuals.html       # 값을 조작하는 오프라인 시각 실험
├── lab.py             # Python 표준 라이브러리 기반 개념 실습
├── lab-guide.md       # 실행법, 코드 해설, 확장 실습
├── lab-output.txt     # 실제 실행 결과
├── quiz.md            # 복습 문제와 접어서 보는 해설
├── source-map.json    # 원문 목차와 PDF 페이지
├── coverage.json      # 원문 절·그림과 노트의 대응
└── status.json        # 검증 결과와 한계
```

## 실습 실행

기본 실습은 API 키와 GPU 없이 실행하도록 구성했습니다. 실제 사전 학습 LLM을 내려받아 돌리는 책의 실습과, 원리를 작은 데이터로 계산하는 이 저장소의 개념 실습을 구별하세요.

```bash
python3 chapters/ch01/lab.py
python3 tools/run_labs.py
python3 tools/check_materials.py
```

시각화는 내려받은 `chapters/chNN/visuals.html`을 브라우저로 열면 됩니다. GitHub의 파일 보기 화면에서는 HTML의 JavaScript가 실행되지 않습니다. 노트 속 Mermaid와 정적 도식은 시각화 파일을 실행하지 않아도 읽을 수 있게 구성했습니다.

## 원문과 검증

- [원문·출처·실습 범위](SOURCE_NOTES.md)
- [전 장 범위 점검표](COVERAGE.md)
- [자료 검증 결과](VALIDATION.md)
- [챕터를 연결하는 개념 지도와 용어집](CONCEPTS.md)

원본 PDF와 전체 본문 추출 파일은 개인 학습용 입력입니다. `.gitignore`로 게시 대상에서 제외합니다. 저장소에는 직접 작성한 설명·도식·학습 코드와 원문 절의 위치 정보를 보관합니다. 책의 2024년 예제에 등장하는 모델·서비스·패키지는 해당 판본의 설명이며, 현재 서비스의 가격·무료 사용 조건이나 최신 API 호환성을 보장하는 목록이 아닙니다.
