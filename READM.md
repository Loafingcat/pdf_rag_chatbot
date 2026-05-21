# PDF RAG Chatbot

Streamlit 기반 PDF RAG 챗봇입니다.

PDF를 업로드하면 문서를 chunk 단위로 분할하고, embedding을 생성한 뒤 ChromaDB에 저장합니다.  
사용자가 질문하면 질문과 유사한 문서 조각을 검색하고, 해당 문맥을 기반으로 답변을 생성합니다.

## 주요 기능

- PDF 업로드
- PDF 텍스트 추출
- 문서 chunk 분할
- SentenceTransformer embedding
- ChromaDB 벡터 저장
- 질문 기반 유사 문서 검색
- OpenAI API 기반 답변 생성
- 답변에 사용된 source chunk 확인

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