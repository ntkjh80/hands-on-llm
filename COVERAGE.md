# 원문 절별 학습 자료 대응표

[전체 목차](README.md) · [원문과 범위](SOURCE_NOTES.md)

제공된 PDF 북마크의 절·하위절 182개를 학습 노트와 연결합니다. PDF 쪽은 파일상 페이지 번호입니다. 연결 여부는 설명을 찾아갈 수 있다는 뜻이며, 원문의 모든 문장을 복제했다는 뜻은 아닙니다.

절 연결: **182/182**. 자세한 파일·링크 검사는 [검증 결과](VALIDATION.md)를 확인하세요.

## 01장 언어 모델 입문

[학습 노트](chapters/ch01/study.md) · 원문 PDF 23–66쪽

| 원문 절 | 시작 PDF 쪽 | 학습 설명 |
|---|---:|---|
| What Is Language AI? | 24 | [설명](chapters/ch01/study.md#language-ai) 언어 AI와 NLP · 생성과 검색의 구분 · 입력·출력·평가 |
| A Recent History of Language AI | 25 | [설명](chapters/ch01/study.md#history) 기술 발전의 문제 중심 흐름 · 기존 기술과 새로운 기술의 결합 · 원문 연표 |
| Representing Language as a Bag-of-Words | 26 | [설명](chapters/ch01/study.md#bow) 공백 토큰화 · 공유 어휘 · 단어 빈도 벡터 |
| Better Representations with Dense Vector Embeddings | 28 | [설명](chapters/ch01/study.md#dense) word2vec · 주변 단어 예측 · 신경망 파라미터 |
| Types of Embeddings | 31 | [설명](chapters/ch01/study.md#embedding-types) 단어·토큰·문장·문서 표현 · 희소/밀집 · 정적/문맥별 |
| Encoding and Decoding Context with Attention | 32 | [설명](chapters/ch01/study.md#rnn-attention) 정적 임베딩의 문맥 한계 · RNN · 인코더/디코더 번역 |
| Attention Is All You Need | 37 | [설명](chapters/ch01/study.md#transformer) 셀프 어텐션 · 피드포워드 · 교차 어텐션 |
| Representation Models: Encoder-Only Models | 42 | [설명](chapters/ch01/study.md#encoder) BERT-base · 마스크 언어 모델링 · CLS |
| Generative Models: Decoder-Only Models | 45 | [설명](chapters/ch01/study.md#decoder) GPT · 디코더 전용 구조 · 다음 토큰 예측 |
| The Year of Generative AI | 49 | [설명](chapters/ch01/study.md#generative-year) 제품과 모델 · 2023년 공개/비공개 모델 사례 · 기반 모델 |
| The Moving Definition of a “Large Language Model” | 50 | [설명](chapters/ch01/study.md#large-definition) 대규모의 상대성 · 비생성·소형 모델 포함 · 기능과 평가 중심 선택 |
| The Training Paradigm of Large Language Models | 51 | [설명](chapters/ch01/study.md#training) 전통적 과제 학습 · 사전 학습 · 미세 조정 |
| Large Language Model Applications: What Makes Them So Useful? | 53 | [설명](chapters/ch01/study.md#applications) 감성 분류 · 군집화·주제 발견 · 검색 |
| Responsible LLM Development and Usage | 55 | [설명](chapters/ch01/study.md#responsibility) 편향과 공정성 · 투명성과 책임 · 그럴듯한 오류와 유해 출력 |
| Limited Resources Are All You Need | 56 | [설명](chapters/ch01/study.md#resources) GPU와 VRAM · 가중치 저장량 · 양자화 |
| Interfacing with Large Language Models | 57 | [설명](chapters/ch01/study.md#interfaces) 인터페이스·모델·실행 엔진 · API/로컬 비교 · 데이터 흐름과 운영 책임 |
| Proprietary, Private Models | 57 | [설명](chapters/ch01/study.md#private-models) 비공개 가중치 · API 호출 · 운영 장점 |
| Open Models | 59 | [설명](chapters/ch01/study.md#open-models) 공개 가중치 범위 · 라이선스 · OSI 용어 교정 |
| Open Source Frameworks | 60 | [설명](chapters/ch01/study.md#frameworks) Transformers · llama.cpp · LangChain 역할 |
| Generating Your First Text | 61 | [설명](chapters/ch01/study.md#first-generation) Hugging Face Hub · Phi-3-mini · 모델과 토크나이저 |
| Summary | 64 | [설명](chapters/ch01/study.md#summary) 표현과 생성 전체 연결 · 학습과 실행 선택 · 2·3장 연결 |

원문 그림·표 32개에 대한 위치와 설명 연결은 [장별 대응 데이터](chapters/ch01/coverage.json)에 기록했습니다.

## 02장 토큰과 임베딩

[학습 노트](chapters/ch02/study.md) · 원문 PDF 67–115쪽

| 원문 절 | 시작 PDF 쪽 | 학습 설명 |
|---|---:|---|
| LLM Tokenization | 68 | [설명](chapters/ch02/study.md#sec-01) 토큰의 정의 · 입력과 순차 생성의 단위 · 토큰화와 임베딩 구별 |
| How Tokenizers Prepare the Inputs to the Language Model | 68 | [설명](chapters/ch02/study.md#sec-02) 문자열 분할 · ID 매핑 · ID와 의미 좌표 구별 |
| Downloading and Running an LLM | 69 | [설명](chapters/ch02/study.md#sec-03) Phi-3 원문 실행 흐름 · input_ids와 장치 · max_new_tokens |
| How Does the Tokenizer Break Down Text? | 74 | [설명](chapters/ch02/study.md#sec-04) 알고리즘·설정·학습 데이터 · 토크나이저 학습과 모델 학습 · BPE 병합 보충 |
| Word Versus Subword Versus Character Versus Byte Tokens | 75 | [설명](chapters/ch02/study.md#sec-05) 단어·서브워드·문자·바이트 · 어휘 밖 표현 · 컨텍스트 길이 |
| Comparing Trained LLM Tokenizers | 78 | [설명](chapters/ch02/study.md#sec-06) BERT uncased/cased · GPT-2 · Flan-T5와 SentencePiece |
| Tokenizer Properties | 91 | [설명](chapters/ch02/study.md#sec-07) 알고리즘 · 어휘 크기 · 특수 토큰 |
| Token Embeddings | 94 | [설명](chapters/ch02/study.md#sec-08) 숫자 ID에서 학습 벡터로 · 분산 표현 · 언어 패턴과 사실성 구별 |
| A Language Model Holds Embeddings for the Vocabulary of Its Tokenizer | 95 | [설명](chapters/ch02/study.md#sec-09) 어휘와 임베딩 행렬 · lookup · 무작위 초기화와 학습 |
| Creating Contextualized Word Embeddings with Language Models | 96 | [설명](chapters/ch02/study.md#sec-10) 정적/문맥 벡터 · DeBERTa 예제 · [B,T,d] 모양 |
| Text Embeddings (for Sentences and Whole Documents) | 100 | [설명](chapters/ch02/study.md#sec-11) 텍스트 전체의 벡터 · 평균 풀링 · 마스크 |
| Word Embeddings Beyond LLMs | 102 | [설명](chapters/ch02/study.md#sec-12) LLM 이전의 임베딩 · Word2vec/GloVe/fastText · 표현 대상 단위 |
| Using pretrained Word Embeddings | 103 | [설명](chapters/ch02/study.md#sec-13) Gensim 사전 학습 GloVe · 50차원 · 최근접 단어와 관계 |
| The Word2vec Algorithm and Contrastive Training | 103 | [설명](chapters/ch02/study.md#sec-14) 슬라이딩 윈도우 · 양성 쌍 · 음성 샘플링 |
| Embeddings for Recommendation Systems | 108 | [설명](chapters/ch02/study.md#sec-15) 추천의 관계 표현 · 곡 ID와 오디오 구별 · 공동 소비 데이터의 의미 |
| Recommending Songs by Embeddings | 108 | [설명](chapters/ch02/study.md#sec-16) 곡=단어/재생목록=문장 · 라디오 재생목록 데이터 · 팝과 랩 원문 추천 사례 |
| Training a Song Embedding Model | 110 | [설명](chapters/ch02/study.md#sec-17) 재생목록·메타데이터 전처리 · Word2Vec 매개변수 · sg 기본값 점검 |
| Summary | 113 | [설명](chapters/ch02/study.md#sec-18) 핵심 개념 통합 · 원문 그림 17개와 비교표 연결 · 실습과 자기 점검 |

원문 그림·표 18개에 대한 위치와 설명 연결은 [장별 대응 데이터](chapters/ch02/coverage.json)에 기록했습니다.

## 03장 트랜스포머 내부 구조

[학습 노트](chapters/ch03/study.md) · 원문 PDF 116–159쪽

| 원문 절 | 시작 PDF 쪽 | 학습 설명 |
|---|---:|---|
| An Overview of Transformer Models | 117 | [설명](chapters/ch03/study.md#sec-01) decoder-only 중심 범위 · 순전파와 생성의 구별 · 2024년 Phi-3 예제 설정 |
| The Inputs and Outputs of a Trained Transformer LLM | 117 | [설명](chapters/ch03/study.md#sec-02) 자동 회귀 생성 · 토큰 추가 루프 · 생성 종료 조건 |
| The Components of the Forward Pass | 121 | [설명](chapters/ch03/study.md#sec-03) Tokenizer와 임베딩의 역할 · 어휘와 모델 차원 · Phi-3 계층 출력 |
| Choosing a Single Token from the Probability Distribution (Sampling/Decoding) | 124 | [설명](chapters/ch03/study.md#sec-04) greedy와 sampling · 온도의 개념 · 안정적인 softmax |
| Parallel Token Processing and Context Size | 127 | [설명](chapters/ch03/study.md#sec-05) 입력 토큰의 병렬 처리 · 출력의 순차 의존성 · 문맥 길이 |
| Speeding Up Generation by Caching Keys and Values | 129 | [설명](chapters/ch03/study.md#sec-06) causal 불변성과 KV 재사용 · prefill과 decode · use_cache 비교 |
| Inside the Transformer Block | 132 | [설명](chapters/ch03/study.md#sec-07) 블록 스택 · FFN과 학습된 패턴 · 기본 모델과 instruct 모델 |
| Recent Improvements to the Transformer Architecture | 143 | [설명](chapters/ch03/study.md#sec-08) 2024년 기준 개선 범위 · 데이터·학습·구조의 구별 · 비용 관점의 비교 |
| More Efficient Attention | 144 | [설명](chapters/ch03/study.md#sec-09) local와 sparse attention · full/sparse 혼합 · 마스크 그림 해석 |
| The Transformer Block | 150 | [설명](chapters/ch03/study.md#sec-10) residual · post-norm과 pre-norm · LayerNorm과 RMSNorm |
| Positional Embeddings (RoPE) | 153 | [설명](chapters/ch03/study.md#sec-11) 절대 위치와 학습형 위치 · packing · 문서 경계 마스크 |
| Other Architectural Experiments and Improvements | 156 | [설명](chapters/ch03/study.md#sec-12) 시각·로보틱스·시계열의 Transformer · 입력·출력·위치 설계의 과제별 변화 · 원문 추가 읽기 범위 |
| Summary | 157 | [설명](chapters/ch03/study.md#sec-13) 한 토큰의 전체 경로 · 파라미터·캐시·대화 기록의 구별 · 4장 분류 연결 |

원문 그림·표 33개에 대한 위치와 설명 연결은 [장별 대응 데이터](chapters/ch03/coverage.json)에 기록했습니다.

## 04장 텍스트 분류

[학습 노트](chapters/ch04/study.md) · 원문 PDF 160–191쪽

| 원문 절 | 시작 PDF 쪽 | 학습 설명 |
|---|---:|---|
| The Sentiment of Movie Reviews | 161 | [설명](chapters/ch04/study.md#sec-01) 영화 리뷰 이진 라벨 · 전체 10662개와 분할 8530/1066/1066 · 학습·검증·시험 구분 |
| Text Classification with Representation Models | 163 | [설명](chapters/ch04/study.md#sec-02) 작업 특화 모델과 임베딩 모델 · 미세 조정과 동결 · 출력과 후속 분류 규칙 |
| Model Selection | 164 | [설명](chapters/ch04/study.md#sec-03) 언어·도메인·크기·추론 속도 · BERT류 기반 모델 선택지 · 트윗 RoBERTa |
| Using a Task-Specific Model | 166 | [설명](chapters/ch04/study.md#sec-04) pipeline·토크나이저 · 하위 토큰과 미등록 단어 · KeyDataset·argmax |
| Classification Tasks That Leverage Embeddings | 171 | [설명](chapters/ch04/study.md#sec-05) 특징 추출과 분류의 분리 · 임베딩 재사용·캐시 · 같은 좌표계 조건 |
| Supervised Classification | 172 | [설명](chapters/ch04/study.md#sec-06) SentenceTransformer encode · 8530×768 배열 · 로지스틱 회귀 fit/predict |
| What If We Do Not Have Labeled Data? | 176 | [설명](chapters/ch04/study.md#sec-07) 제로샷의 의미 · 라벨 설명 임베딩 · 코사인 내적·길이·차원 |
| Text Classification with Generative Models | 180 | [설명](chapters/ch04/study.md#sec-08) 클래스 점수와 생성 문자열 · 프롬프트 · seq2seq 작업과 구조 구분 |
| Using the Text-to-Text Transfer Transformer | 181 | [설명](chapters/ch04/study.md#sec-09) encoder-decoder T5 · span corruption · text-to-text 다중 과제 |
| ChatGPT for Classification | 186 | [설명](chapters/ch04/study.md#sec-10) 지시 데이터와 선호 데이터 · 원격 API와 클라이언트 · 프롬프트·메시지·temperature |
| Summary | 190 | [설명](chapters/ch04/study.md#sec-11) 다섯 경로 비교 · 평가 조건 차이 · 설계 선택·오류 분석 |

원문 그림·표 28개에 대한 위치와 설명 연결은 [장별 대응 데이터](chapters/ch04/coverage.json)에 기록했습니다.

## 05장 텍스트 군집화와 토픽 모델링

[학습 노트](chapters/ch05/study.md) · 원문 PDF 192–235쪽

| 원문 절 | 시작 PDF 쪽 | 학습 설명 |
|---|---:|---|
| ArXiv’s Articles: Computation and Language | 193 | [설명](chapters/ch05/study.md#sec-01) arXiv cs.CL 44,949개 초록 · 제목/초록의 행 정렬 · 데이터 범위와 연도 |
| A Common Pipeline for Text Clustering | 194 | [설명](chapters/ch05/study.md#sec-02) 임베딩-차원축소-군집화 · 각 단계의 행렬 모양 · 모듈의 역할 |
| Embedding Documents | 195 | [설명](chapters/ch05/study.md#sec-03) gte-small과 384차원 · 군집화 평가와 비용의 균형 · 코사인 유사도 |
| Reducing the Dimensionality of Embeddings | 196 | [설명](chapters/ch05/study.md#sec-04) PCA와 UMAP · 5차원 군집화 · min_dist/metric/random_state |
| Cluster the Reduced Embeddings | 199 | [설명](chapters/ch05/study.md#sec-05) k-means와 HDBSCAN 비교 · min_cluster_size · euclidean/eom |
| Inspecting the Clusters | 201 | [설명](chapters/ch05/study.md#sec-06) 대표/표본 문서 읽기 · 수어 군집 · 2차원 투영 |
| From Text Clustering to Topic Modeling | 204 | [설명](chapters/ch05/study.md#sec-07) 토픽 키워드와 자연어 이름 · LDA와 어휘 분포 · 임베딩 기반 군집화와의 차이 |
| BERTopic: A Modular Topic Modeling Framework | 206 | [설명](chapters/ch05/study.md#sec-08) 군집과 표현의 분리 · BoW와 군집별 집계 · 정규화 cTF |
| Adding a Special Lego Block | 220 | [설명](chapters/ch05/study.md#sec-09) 표현 블록 재정렬 · deepcopy와 전후 비교 · KeyBERTInspired 대표 문서 |
| The Text Generation Lego Block | 228 | [설명](chapters/ch05/study.md#sec-10) 대표 문서와 키워드 프롬프트 · 토픽당 생성 호출 · Flan-T5와 whitespace 길이 제한 |
| Summary | 233 | [설명](chapters/ch05/study.md#sec-11) 전체 파이프라인 통합 · 검토 질문 · 다음 프롬프트 장과의 연결 |

원문 그림·표 29개에 대한 위치와 설명 연결은 [장별 대응 데이터](chapters/ch05/coverage.json)에 기록했습니다.

## 06장 프롬프트 엔지니어링

[학습 노트](chapters/ch06/study.md) · 원문 PDF 236–279쪽

| 원문 절 | 시작 PDF 쪽 | 학습 설명 |
|---|---:|---|
| Using Text Generation Models | 236 | [설명](chapters/ch06/study.md#sec-01) 모델·입력·디코딩의 역할 · 생성 모델과 표현 모델 |
| Choosing a Text Generation Model | 236 | [설명](chapters/ch06/study.md#sec-02) 비공개 모델과 공개 모델 · 기반 모델 계열·크기 · Phi-3-mini와 메모리 해석 |
| Loading a Text Generation Model | 237 | [설명](chapters/ch06/study.md#sec-03) 모델·토크나이저·pipeline · 역할과 chat template · 생성 길이와 반환 설정 |
| Controlling Model Output | 240 | [설명](chapters/ch06/study.md#sec-04) greedy와 sampling · temperature softmax · top-p와 top-k |
| Intro to Prompt Engineering | 244 | [설명](chapters/ch06/study.md#sec-05) 반복적인 프롬프트 최적화 · 성공 기준과 평가 |
| The Basic Ingredients of a Prompt | 244 | [설명](chapters/ch06/study.md#sec-06) 문장 완성 · instruction·data·output indicator · 자료 경계 |
| Instruction-Based Prompting | 247 | [설명](chapters/ch06/study.md#sec-07) 분류·검색·요약·코드·NER · 구체성 · 환각 대처 |
| Advanced Prompt Engineering | 250 | [설명](chapters/ch06/study.md#sec-08) 입력 내부 설계와 호출 간 흐름 · 필요에 맞는 복잡성 |
| The Potential Complexity of a Prompt | 251 | [설명](chapters/ch06/study.md#sec-09) 일곱 모듈 · 문자열 조립 · 반복 실험 |
| In-Context Learning: Providing Examples | 255 | [설명](chapters/ch06/study.md#sec-10) zero·one·few-shot · user와 assistant 예시 · 가상 단어 예제 |
| Chain Prompting: Breaking up the Problem | 257 | [설명](chapters/ch06/study.md#sec-11) 제품 이름·슬로건·판매 문구 · 그림의 세 단계와 코드의 두 호출 · 중간 결과·오류 전파 |
| Reasoning with Generative Models | 260 | [설명](chapters/ch06/study.md#sec-12) system 1·2 비유 · 검증 가능한 문제 해결 · 추가 계산의 위치 |
| Chain-of-Thought: Think Before Answering | 261 | [설명](chapters/ch06/study.md#sec-13) 풀이 예시 기반 CoT · zero-shot CoT · 중간 계산과 검산 |
| Self-Consistency: Sampling Outputs | 265 | [설명](chapters/ch06/study.md#sec-14) 다양한 응답 샘플링 · 최종 답 추출·정규화·투표 · 독립성과 상관 오류 |
| Tree-of-Thought: Exploring Intermediate Steps | 266 | [설명](chapters/ch06/study.md#sec-15) 중간 상태 생성·평가·가지치기 · 탐색량 · 전문가 대화형 모방과 실제 탐색 구분 |
| Output Verification | 270 | [설명](chapters/ch06/study.md#sec-16) 구조·허용값·윤리·정확도 · 예시·문법·미세 조정 · 파싱·스키마·내용 검증 |
| Providing Examples | 272 | [설명](chapters/ch06/study.md#sec-17) RPG JSON의 잘린 출력 · 형식 예시의 효과와 한계 · 길이 예산과 필드 계약 |
| Grammar: Constrained Sampling | 274 | [설명](chapters/ch06/study.md#sec-18) 사후 모델 검사 · 알려진 뼈대와 값 생성 · 후보 마스크 |
| Summary | 278 | [설명](chapters/ch06/study.md#sec-19) 장 전체 연결 · 7장의 메모리·도구 활용 · 복습과 실험 설계 |

원문 그림·표 22개에 대한 위치와 설명 연결은 [장별 대응 데이터](chapters/ch06/coverage.json)에 기록했습니다.

## 07장 고급 텍스트 생성과 도구

[학습 노트](chapters/ch07/study.md) · 원문 PDF 280–315쪽

| 원문 절 | 시작 PDF 쪽 | 학습 설명 |
|---|---:|---|
| Model I/O: Loading Quantized Models with LangChain | 281 | [설명](chapters/ch07/study.md#sec-01) GGUF and precision · FP16 versus 8-bit source inconsistency · quantization size and error |
| Chains: Extending the Capabilities of LLMs | 284 | [설명](chapters/ch07/study.md#sec-02) chain definition · module boundaries · I/O contracts |
| A Single Link in the Chain: Prompt Template | 285 | [설명](chapters/ch07/study.md#sec-03) chat format versus task template · Phi-3 delimiters · PromptTemplate fields |
| A Chain with Multiple Prompts | 289 | [설명](chapters/ch07/study.md#sec-04) title, character, story · state preservation · output_key |
| Memory: Helping LLMs to Remember Conversations | 293 | [설명](chapters/ch07/study.md#sec-05) stateless calls · external state · training and KV cache distinction |
| Conversation Buffer | 295 | [설명](chapters/ch07/study.md#sec-06) chat_history and memory_key · load before, save after · cumulative input |
| Windowed Conversation Buffer | 297 | [설명](chapters/ch07/study.md#sec-07) last k interactions · name repetition and age loss · turn count versus token budget |
| Conversation Summary | 300 | [설명](chapters/ch07/study.md#sec-08) summary and new_lines · separate summary call · loss and error propagation |
| Agents: Creating a System of LLMs | 306 | [설명](chapters/ch07/study.md#sec-09) tools and selection policy · model-executor-tool boundaries · chain versus agent |
| The Driving Power Behind Agents: Step-by-step Reasoning | 307 | [설명](chapters/ch07/study.md#sec-10) Thought, Action, Observation · lookup then conversion · scratchpad |
| ReAct in LangChain | 310 | [설명](chapters/ch07/study.md#sec-11) historical model choice · four template variables · search and llm-math |
| Summary | 314 | [설명](chapters/ch07/study.md#sec-12) combined components · selection criteria · connection to chapter 8 |

원문 그림·표 18개에 대한 위치와 설명 연결은 [장별 대응 데이터](chapters/ch07/coverage.json)에 기록했습니다.

## 08장 시맨틱 검색과 RAG

[학습 노트](chapters/ch08/study.md) · 원문 PDF 316–359쪽

| 원문 절 | 시작 PDF 쪽 | 학습 설명 |
|---|---:|---|
| Overview of Semantic Search and RAG | 317 | [설명](chapters/ch08/study.md#sec-01) dense retrieval와 reranking과 RAG 입출력 · 생성형 검색과 근거 기반 답변 |
| Semantic Search with Language Models | 319 | [설명](chapters/ch08/study.md#sec-02) 오프라인 색인과 온라인 검색 · 문서 ID와 출처 연결 |
| Dense Retrieval | 319 | [설명](chapters/ch08/study.md#sec-03) 코사인과 제곱 L2 · 영화 API 예제와 행렬 형태 · BM25 비교 |
| Reranking | 336 | [설명](chapters/ch08/study.md#sec-04) 2단계 검색과 후보 상한 · bi-encoder와 cross-encoder · Cohere Rerank 예제 |
| Retrieval Evaluation Metrics | 341 | [설명](chapters/ch08/study.md#sec-05) 평가 코퍼스와 질의와 관련성 판정 · Precision과 Recall · AP와 전체 정답 분모 |
| Retrieval-Augmented Generation (RAG) | 349 | [설명](chapters/ch08/study.md#sec-06) RAG 구조와 환각 감소의 한계 · 외부 자료 활용과 미세 조정의 차이 |
| From Search to RAG | 349 | [설명](chapters/ch08/study.md#sec-07) 질문과 검색 문맥을 생성기에 전달 · 문맥 예산 · 자료 ID와 근거 지원 |
| Example: Grounded Generation with an LLM API | 351 | [설명](chapters/ch08/study.md#sec-08) 검색 결과를 documents로 전달 · 답변 span과 문서 ID 인용 · 관리형 API 예제의 범위 |
| Example: RAG with Local Models | 352 | [설명](chapters/ch08/study.md#sec-09) Phi-3 GGUF 로딩 · fp16과 파일 형식 구분 · BGE 설명과 GTE 코드 불일치 |
| Advanced RAG Techniques | 355 | [설명](chapters/ch08/study.md#sec-10) 질의 재작성 · 다중 질의 · 다단계 검색 |
| RAG Evaluation | 357 | [설명](chapters/ch08/study.md#sec-11) 유창성 · 인지된 유용성 · 인용 재현율 |
| Summary | 358 | [설명](chapters/ch08/study.md#sec-12) 검색·재순위화·RAG 통합 · 오류 위치 진단 · 9장과 10장 연결 |

원문 그림·표 26개에 대한 위치와 설명 연결은 [장별 대응 데이터](chapters/ch08/coverage.json)에 기록했습니다.

## 09장 멀티모달 언어 모델

[학습 노트](chapters/ch09/study.md) · 원문 PDF 360–398쪽

| 원문 절 | 시작 PDF 쪽 | 학습 설명 |
|---|---:|---|
| Transformers for Vision | 361 | [설명](chapters/ch09/study.md#sec-01) 이미지 패치 · 패치 개수와 원시 벡터 길이 · 선형 투영 |
| Multimodal Embedding Models | 364 | [설명](chapters/ch09/study.md#sec-02) 공동 임베딩 공간 · 차원 일치와 의미 정렬의 차이 · 모달리티 간 검색 |
| CLIP: Connecting Text and Images | 366 | [설명](chapters/ch09/study.md#sec-03) 제로샷 분류 · 군집화 · 텍스트·이미지 검색 |
| How Can CLIP Generate Multimodal Embeddings? | 367 | [설명](chapters/ch09/study.md#sec-04) 이미지·캡션 쌍 · 두 인코더 · 코사인 유사도 |
| OpenCLIP | 370 | [설명](chapters/ch09/study.md#sec-05) OpenCLIP 절과 실제 Transformers 체크포인트 구분 · 토큰화 · 텍스트 특징 형상 |
| Making Text Generation Models Multimodal | 377 | [설명](chapters/ch09/study.md#sec-06) 시각 이해와 자유로운 텍스트 생성 · 모달리티 간 간격 · 사전학습 모델 연결 |
| BLIP-2: Bridging the Modality Gap | 378 | [설명](chapters/ch09/study.md#sec-07) 시각 인코더와 LLM 동결 · Q-Former의 학습 가능한 query · 첫 단계 세 목적 |
| Preprocessing Multimodal Inputs | 383 | [설명](chapters/ch09/study.md#sec-08) processor와 모델 · RGB·배치·공간 차원 · resize·crop·padding |
| Use Case 1: Image Captioning | 387 | [설명](chapters/ch09/study.md#sec-09) 캡셔닝 흐름 · generate의 텍스트 토큰 ID · batch_decode |
| Use Case 2: Multimodal Chat-Based Prompting | 391 | [설명](chapters/ch09/study.md#sec-10) 이미지와 질문 동시 입력 · 프롬프트 형식 · 이력 재전달 |
| Summary | 395 | [설명](chapters/ch09/study.md#sec-11) ViT·CLIP·BLIP-2 역할 비교 · 입출력과 자료형 정리 · 10장 대조 학습 연결 |

원문 그림·표 21개에 대한 위치와 설명 연결은 [장별 대응 데이터](chapters/ch09/coverage.json)에 기록했습니다.

## 10장 텍스트 임베딩 모델 만들기

[학습 노트](chapters/ch10/study.md) · 원문 PDF 399–445쪽

| 원문 절 | 시작 PDF 쪽 | 학습 설명 |
|---|---:|---|
| Embedding Models | 399 | [설명](chapters/ch10/study.md#sec-01) 작업별 유사성 · 고정 차원 임베딩 · 코사인 |
| What Is Contrastive Learning? | 402 | [설명](chapters/ch10/study.md#sec-02) 대조 관계 · positive와 negative · word2vec 연결 |
| SBERT | 404 | [설명](chapters/ch10/study.md#sec-03) cross-encoder · bi-encoder · 계산량 |
| Creating an Embedding Model | 408 | [설명](chapters/ch10/study.md#sec-04) 데이터·모델·손실·평가 파이프라인 · from scratch 의미 |
| Generating Contrastive Examples | 408 | [설명](chapters/ch10/study.md#sec-05) MNLI와 NLI 라벨 · 관계 매핑 · 함의 비대칭성 |
| Train Model | 410 | [설명](chapters/ch10/study.md#sec-06) BERT 시작점 · SoftmaxLoss · STS-B |
| In-Depth Evaluation | 414 | [설명](chapters/ch10/study.md#sec-07) MTEB · Banking77 · 지연 시간 |
| Loss Functions | 416 | [설명](chapters/ch10/study.md#sec-08) cosine 회귀 · MNR 수식 · temperature |
| Fine-Tuning an Embedding Model | 426 | [설명](chapters/ch10/study.md#sec-09) 기존 임베딩 모델 재사용 · 기본 모델 비교 |
| Supervised | 426 | [설명](chapters/ch10/study.md#sec-10) MiniLM · MNR 데이터 전제 · 기학습 데이터 |
| Augmented SBERT | 429 | [설명](chapters/ch10/study.md#sec-11) gold와 silver · cross-encoder 교사 · 후보 쌍 생성 |
| Unsupervised Learning | 435 | [설명](chapters/ch10/study.md#sec-12) 비지도 학습 신호 · SimCSE·CT·TSDAE·GPL 언급 |
| Transformer-Based Sequential Denoising Auto-Encoder | 435 | [설명](chapters/ch10/study.md#sec-13) 단어 삭제 · 복원 병목 · CLS pooling |
| Using TSDAE for Domain Adaptation | 441 | [설명](chapters/ch10/study.md#sec-14) 도메인 차이 · adaptive pretraining · 목표 문장 적응 |
| Summary | 443 | [설명](chapters/ch10/study.md#sec-15) 학습 방법 선택 · 다음 장 연결 · 각주 연구 흐름 |

원문 그림·표 15개에 대한 위치와 설명 연결은 [장별 대응 데이터](chapters/ch10/coverage.json)에 기록했습니다.

## 11장 표현 모델 미세 조정

[학습 노트](chapters/ch11/study.md) · 원문 PDF 446–486쪽

| 원문 절 | 시작 PDF 쪽 | 학습 설명 |
|---|---:|---|
| Supervised Classification | 446 | [설명](chapters/ch11/study.md#sec-01) 동결 특징 추출과 전체 미세 조정 · 분류 헤드와 역전파 · 로짓·softmax·교차 엔트로피 |
| Fine-Tuning a Pretrained BERT Model | 449 | [설명](chapters/ch11/study.md#sec-02) Rotten Tomatoes 데이터 · bert-base-cased · 토큰화·절단·동적 패딩 |
| Freezing Layers | 452 | [설명](chapters/ch11/study.md#sec-03) requires_grad와 eval 모드의 차이 · 헤드만 학습 · 앞 10개 블록 동결 |
| Few-Shot Classification | 459 | [설명](chapters/ch11/study.md#sec-04) few-shot training과 prompting 구별 · 클래스당 예제 수 · 작은 데이터의 라벨 품질 |
| SetFit: Efficient Fine-Tuning with Few Training Examples | 460 | [설명](chapters/ch11/study.md#sec-05) 같은·다른 클래스 쌍 생성 · 고유 양성·음성 쌍 수 · 대조학습으로 임베딩 조정 |
| Fine-Tuning for Few-Shot Classification | 464 | [설명](chapters/ch11/study.md#sec-06) 클래스당 16개 · all-mpnet-base-v2 · 3 epoch·num_iterations 20 |
| Continued Pretraining with Masked Language Modeling | 468 | [설명](chapters/ch11/study.md#sec-07) 일반→도메인→과제의 세 단계 · MLM 자기지도 목표 · token masking과 whole-word masking |
| Named-Entity Recognition | 475 | [설명](chapters/ch11/study.md#sec-08) 문장 분류와 토큰 분류 · B×T×C 출력 · 문맥 기반 개체 인식 |
| Preparing Data for Named-Entity Recognition | 476 | [설명](chapters/ch11/study.md#sec-09) CoNLL-2003 구조 · BIO 라벨 매핑 · 대체 데이터셋 팁 |
| Fine-Tuning for Named-Entity Recognition | 483 | [설명](chapters/ch11/study.md#sec-10) AutoModelForTokenClassification · DataCollatorForTokenClassification · 학습·저장·추론 |
| Summary | 485 | [설명](chapters/ch11/study.md#sec-11) 방법 선택표 · 데이터·목표·파라미터·평가의 연결 · 통합 설계 |

원문 그림·표 21개에 대한 위치와 설명 연결은 [장별 대응 데이터](chapters/ch11/coverage.json)에 기록했습니다.

## 12장 생성 모델 미세 조정

[학습 노트](chapters/ch12/study.md) · 원문 PDF 487–535쪽

| 원문 절 | 시작 PDF 쪽 | 학습 설명 |
|---|---:|---|
| The Three LLM Training Steps: Pretraining, Supervised Fine-Tuning, and Preference Tuning | 487 | [설명](chapters/ch12/study.md#sec-01) 자기지도 다음 토큰 예측 · 기반 모델 · SFT와 선호 학습의 데이터 차이 |
| Supervised Fine-Tuning (SFT) | 489 | [설명](chapters/ch12/study.md#sec-02) 지시 준수 · 다음 토큰 NLL · 응답 마스킹 |
| Full Fine-Tuning | 490 | [설명](chapters/ch12/study.md#sec-03) 전체 가중치 업데이트 · 훈련 메모리 구성 · 작업별 저장 비용 |
| Parameter-Efficient Fine-Tuning (PEFT) | 492 | [설명](chapters/ch12/study.md#sec-04) 어댑터 구조와 교환 · LoRA 차원과 변화량 · rank와 alpha |
| Instruction Tuning with QLoRA | 503 | [설명](chapters/ch12/study.md#sec-05) TinyLlama QLoRA 실습 전체 흐름 |
| Templating Instruction Data | 503 | [설명](chapters/ch12/study.md#sec-06) 채팅 템플릿 · UltraChat 3000개 · 역할 및 종료 토큰 |
| Model Quantization | 505 | [설명](chapters/ch12/study.md#sec-07) BitsAndBytesConfig · 4비트 저장과 계산 dtype · 캐시 |
| LoRA Configuration | 506 | [설명](chapters/ch12/study.md#sec-08) rank 64와 alpha 32 · dropout · 대상 투영층 |
| Training Configuration | 508 | [설명](chapters/ch12/study.md#sec-09) 배치 누적 · 학습률 · cosine과 warmup 구분 |
| Training | 509 | [설명](chapters/ch12/study.md#sec-10) SFTTrainer 역할 · 토큰 길이 제한 · 어댑터 저장 |
| Merge Weights | 510 | [설명](chapters/ch12/study.md#sec-11) 병합 수식 · 분리 추론 · 정밀도 |
| Evaluating Generative Models | 511 | [설명](chapters/ch12/study.md#sec-12) 사용 사례별 평가 · 학습 손실과 품질 구분 |
| Word-Level Metrics | 512 | [설명](chapters/ch12/study.md#sec-13) perplexity · ROUGE · BLEU |
| Benchmarks | 512 | [설명](chapters/ch12/study.md#sec-14) MMLU · GLUE · TruthfulQA |
| Leaderboards | 515 | [설명](chapters/ch12/study.md#sec-15) 리더보드의 목적 · 평균 점수의 한계 · 목표 과제 차이 |
| Automated Evaluation | 516 | [설명](chapters/ch12/study.md#sec-16) LLM-as-a-judge · 쌍별 비교 · 순서와 문체 편향 |
| Human Evaluation | 516 | [설명](chapters/ch12/study.md#sec-17) 사람 평가 · 익명 쌍별 비교 · Elo 직관 |
| Preference-Tuning / Alignment / RLHF | 518 | [설명](chapters/ch12/study.md#sec-18) 선호도 · 정렬과 RLHF 구분 · 평가와 정책 변화 |
| Automating Preference Evaluation Using Reward Models | 519 | [설명](chapters/ch12/study.md#sec-19) 보상 모델 학습의 필요 · 점수 헤드 |
| The Inputs and Outputs of a Reward Model | 520 | [설명](chapters/ch12/study.md#sec-20) 질문과 응답 입력 · 스칼라 점수 · 절대 점수 해석 |
| Training a Reward Model | 521 | [설명](chapters/ch12/study.md#sec-21) chosen와 rejected · 수집 · 쌍별 점수 차이 손실 |
| Training No Reward Model | 526 | [설명](chapters/ch12/study.md#sec-22) 기준 모델 · 응답 로그확률 합 · DPO 손실 |
| Preference Tuning with DPO | 528 | [설명](chapters/ch12/study.md#sec-23) SFT에서 DPO로 이어지는 모델 계보 |
| Templating Alignment Data | 528 | [설명](chapters/ch12/study.md#sec-24) prompt chosen rejected · distilabel 데이터 · 필터 |
| Model Quantization | 529 | [설명](chapters/ch12/study.md#sec-25) SFT 모델의 양자화 · DPO 어댑터 출발점 · 원문 모델 변수 점검 |
| Training Configuration | 530 | [설명](chapters/ch12/study.md#sec-26) 200 step · warmup 20 step · 학습률 |
| Training | 531 | [설명](chapters/ch12/study.md#sec-27) DPOTrainer · 어댑터 저장 · 순차 병합 |
| Summary | 533 | [설명](chapters/ch12/study.md#sec-28) 전체 단계 연결 · 응용 설계 · 독립 평가 |

원문 그림·표 34개에 대한 위치와 설명 연결은 [장별 대응 데이터](chapters/ch12/coverage.json)에 기록했습니다.

