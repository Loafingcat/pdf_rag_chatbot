from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    pages: List[Dict[str, Any]],
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> List[Dict[str, Any]]:
    """
    페이지 단위 텍스트를 chunk 단위로 나누는 함수.

    RAG에서는 PDF 전체를 그대로 LLM에 넣는 것이 아니라,
    문서를 작은 조각으로 나누고 그중 질문과 관련 있는 조각만 검색해서 사용한다.

    Parameters
    ----------
    pages:
        pdf_loader.load_pdf_pages() 결과
    chunk_size:
        chunk 하나의 최대 길이
    chunk_overlap:
        chunk끼리 겹치게 할 문자 수

    Returns
    -------
    List[Dict[str, Any]]
        [
            {
                "text": "나뉜 문서 조각",
                "metadata": {
                    "source": "파일명.pdf",
                    "page": 1,
                    "chunk_index": 0
                }
            }
        ]
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = []

    for page in pages:
        page_text = page["text"]
        page_metadata = page["metadata"]

        split_texts = splitter.split_text(page_text)

        for chunk_index, chunk_text in enumerate(split_texts):
            chunk_text = chunk_text.strip()

            if not chunk_text:
                continue

            metadata = dict(page_metadata)
            metadata["chunk_index"] = chunk_index

            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": metadata,
                }
            )

    return chunks