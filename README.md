# PDF RAG Chatbot

Streamlit 기반의 **PDF 문서 질의응답 RAG 챗봇**입니다.

사용자가 PDF 파일을 업로드하면 문서 내용을 추출하고, 이를 작은 chunk 단위로 나눈 뒤 embedding 벡터로 변환하여 ChromaDB에 저장합니다. 이후 사용자가 질문을 입력하면 관련 문서 조각을 검색하고, Gemini API를 통해 문서 기반 답변을 생성합니다.

---

## 주요 기능

- PDF 파일 업로드
- PDF 텍스트 추출
- 문서 chunk 분할
- embedding 생성
- ChromaDB 벡터 저장
- 질문 기반 유사 문서 검색
- Gemini API 기반 답변 생성
- 답변에 사용된 문서 조각 확인

---

## 프로젝트 구조

```text
pdf_rag_chatbot/
│
├── app.py
├── requirements.txt
├── data/
│   ├── uploads/
│   └── chroma_db/
├── utils/
│   ├── pdf_loader.py
│   ├── text_splitter.py
│   ├── embedding.py
│   ├── vector_store.py
│   └── rag_chain.py
└── README.md
```

---

## 기술 스택

- **Streamlit**: 웹 UI 구성
- **pypdf**: PDF 텍스트 추출
- **SentenceTransformers**: 문서 embedding 생성
- **ChromaDB**: 벡터 저장 및 유사 문서 검색
- **Google Gemini API**: 문서 기반 답변 생성
- **uv**: 가상환경 및 패키지 관리

---

## RAG 처리 흐름

```text
PDF 업로드
→ 텍스트 추출
→ chunk 분할
→ embedding 생성
→ ChromaDB 저장
→ 사용자 질문 입력
→ 유사 chunk 검색
→ Gemini API 답변 생성
→ 답변 및 참고 문서 조각 출력
```

---

## 설치 및 실행

### 1. 프로젝트 폴더 이동

```bash
cd pdf_rag_chatbot
```

### 2. Python 3.11 환경 설정

```bash
uv python pin 3.11
uv venv --python 3.11
```

### 3. 패키지 설치

```bash
uv pip install -r requirements.txt
```

### 4. 환경변수 설정

프로젝트 루트에 `.env` 파일을 생성합니다.

```env
GOOGLE_API_KEY=your_google_ai_studio_api_key
GEMINI_MODEL=gemini-2.5-flash
```

### 5. 실행

```bash
uv run streamlit run app.py
```

만약 실행 환경이 꼬일 경우 아래 명령어로 실행할 수 있습니다.

```bash
.venv/Scripts/python.exe -m streamlit run app.py
```

---

## 사용 방법

1. 왼쪽 사이드바에서 PDF 파일을 업로드합니다.
2. `PDF 인덱싱하기` 버튼을 클릭합니다.
3. 문서가 벡터 DB에 저장될 때까지 기다립니다.
4. 채팅창에 PDF 내용과 관련된 질문을 입력합니다.
5. 답변과 함께 참고한 문서 조각을 확인합니다.

---

## TOP-K 설정

`TOP-K`는 질문과 가장 관련 있는 문서 조각을 몇 개 가져올지 정하는 값입니다.

```text
top_k = 2~3  → 구체적인 정보 검색
top_k = 4    → 일반적인 질문
top_k = 5~6  → 요약 또는 종합 분석
```

기본값은 `4`입니다.

---

## 예시 질문

```text
이 문서의 핵심 내용을 요약해줘.
```

```text
이 문서에서 중요한 키워드는 뭐야?
```

```text
이 문서에서 문제점과 해결방안을 정리해줘.
```

```text
이 문서를 기반으로 예상 질문 5개를 만들어줘.
```

---

## 파일별 역할

| 파일 | 역할 |
|---|---|
| `app.py` | Streamlit 메인 화면 및 사용자 입력 처리 |
| `pdf_loader.py` | PDF 저장 및 텍스트 추출 |
| `text_splitter.py` | 문서 chunk 분할 |
| `embedding.py` | embedding 모델 로드 및 벡터 생성 |
| `vector_store.py` | ChromaDB 저장 및 검색 |
| `rag_chain.py` | 검색 결과를 바탕으로 Gemini 답변 생성 |

---

## 프로젝트 특징

이 프로젝트는 일반 챗봇과 달리 사용자가 업로드한 PDF 문서를 기준으로 답변합니다.  
따라서 보고서, 논문, 자기소개서, 매뉴얼 등 긴 문서에서 필요한 내용을 빠르게 찾는 데 활용할 수 있습니다.

답변에 사용된 문서 조각도 함께 확인할 수 있어, 답변의 근거를 검토할 수 있습니다.
