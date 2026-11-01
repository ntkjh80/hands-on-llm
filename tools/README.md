# 자료를 검증하고 다시 만드는 도구

[전체 목차](../README.md)

## 기본 검증

Python 표준 라이브러리만 있으면 다음 두 명령을 실행할 수 있습니다.

```bash
python3 tools/check_materials.py
python3 tools/run_labs.py
```

첫 번째는 12개 장의 필수 파일, 원문 182개 절의 대응, 로컬 Markdown 링크와 앵커를 검사합니다. 이는 파일과 연결의 검사이며 설명의 정확성을 자동 증명하지는 않습니다. 두 번째는 각 장의 개념 실습을 실제로 실행합니다. `--chapter 3`을 붙이면 한 장만 실행하고, `--save`를 붙이면 해당 실행 출력을 `lab-output.txt`에 기록합니다.

## 시각화 브라우저 검사

`check_visuals.cjs`는 선택적인 개발 도구입니다. Node.js와 Playwright 및 실행 가능한 Chromium 계열 브라우저가 필요합니다. 학습자가 노트·시각화·기본 실습을 사용하는 데 이 도구를 설치할 필요는 없습니다.

```bash
node tools/check_visuals.cjs
```

Playwright가 별도 위치에 있으면 `PLAYWRIGHT_MODULE`, Chrome을 직접 지정하려면 `CHROME_EXECUTABLE` 환경 변수를 지정합니다. 이 도구는 데스크톱·휴대폰 너비의 화면 이미지, 브라우저 오류, 입력 변경 전후의 화면 변화를 `output/qa/`에 저장합니다. 화면 변화 검사는 수학적 정확성의 증명이 아니므로 각 장의 설명과 수치도 함께 검토해야 합니다.

## 원문 위치 정보 추출

`prepare_source.py`는 원문을 개인적으로 보유한 사용자가 같은 PDF에서 위치 정보를 다시 추출할 때만 쓰는 보조 도구입니다. 별도 Python 환경의 `pypdf`가 필요합니다.

```bash
python3 tools/prepare_source.py /path/to/your/book.pdf
```

이 스크립트는 제공된 **598쪽 판본**에 맞춰져 있습니다. 다른 판본은 페이지 수뿐 아니라 챕터 시작과 끝도 재확인해야 합니다. `source-map.json` 등을 다시 쓰므로 주석을 직접 추가했다면 먼저 보관하세요. 추출한 전체 본문과 원문 페이지 이미지는 `.gitignore`에서 제외합니다.

## GitHub에서 읽을 때

노트의 수식은 `$...$`, `$$...$$` 또는 `math` 코드 블록, 흐름도는 `mermaid` 코드 블록을 사용합니다. 지원 표기는 [GitHub 수식 문서](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)와 [GitHub 도식 문서](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-diagrams)를 기준으로 확인했습니다. 일반 텍스트 편집기는 같은 내용을 소스 형태로 표시할 수 있습니다.

## 깨끗한 학습 묶음 만들기

```bash
python3 tools/package_materials.py
```

완성된 학습 문서·실습·시각화·검증 기록을 `output/hands-on-llm-study.zip`으로 묶습니다. 원문 PDF, 본문 추출, 원문 확인용 이미지, 개인 챗 기록과 작업용 파일은 포함하지 않습니다. 파일이 빠져 있으면 불완전한 압축 파일을 만드는 대신 누락 목록을 표시합니다.

원문 절 링크에는 명시적인 앵커를 둡니다. `name` 속성은 [GitHub 사용자 정의 앵커 문서](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax#custom-anchors)에 맞추고, `id`도 함께 두어 일반 HTML 기반 Markdown 뷰어에서 같은 위치를 가리키도록 했습니다.
