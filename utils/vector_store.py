from pathlib import Path
from typing import List, Dict, Any

import chromadb

from utils.embedding import embed_texts, embed_query


DEFAULT_CHROMA_DIR = "data/chroma_db"
DEFAULT_COLLECTION_NAME = "pdf_documents"


def get_chroma_client(chroma_dir: str = DEFAULT_CHROMA_DIR):
    """
    ChromaDB 클라이언트를 생성하는 함수.

    PersistentClient를 사용하면 벡터 DB가 메모리에만 있는 게 아니라
    data/chroma_db 폴더에 저장된다.
    """

    Path(chroma_dir).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=chroma_dir)
    return client


def get_or_create_collection(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chroma_dir: str = DEFAULT_CHROMA_DIR,
):
    """
    ChromaDB collection을 가져오거나 없으면 생성한다.

    Collection은 쉽게 말하면 벡터 데이터를 저장하는 테이블 같은 개념이다.
    """

    client = get_chroma_client(chroma_dir)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "PDF RAG chatbot document collection"},
    )

    return collection


def make_chunk_id(metadata: Dict[str, Any]) -> str:
    """
    chunk마다 고유 ID를 만드는 함수.

    source + page + chunk_index 조합으로 ID를 만든다.
    """

    source = metadata.get("source", "unknown")
    page = metadata.get("page", 0)
    chunk_index = metadata.get("chunk_index", 0)

    return f"{source}__page_{page}__chunk_{chunk_index}"


def add_chunks_to_vector_store(
    chunks: List[Dict[str, Any]],
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chroma_dir: str = DEFAULT_CHROMA_DIR,
) -> int:
    """
    chunk 목록을 embedding한 뒤 ChromaDB에 저장하는 함수.

    Parameters
    ----------
    chunks:
        text_splitter.split_documents() 결과

    Returns
    -------
    int
        저장한 chunk 개수
    """

    if not chunks:
        return 0

    collection = get_or_create_collection(
        collection_name=collection_name,
        chroma_dir=chroma_dir,
    )

    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    ids = [make_chunk_id(chunk["metadata"]) for chunk in chunks]

    embeddings = embed_texts(texts)

    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(chunks)


def search_similar_chunks(
    query: str,
    top_k: int = 4,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chroma_dir: str = DEFAULT_CHROMA_DIR,
) -> List[Dict[str, Any]]:
    """
    사용자 질문과 가장 유사한 chunk를 ChromaDB에서 검색하는 함수.

    Returns
    -------
    List[Dict[str, Any]]
        [
            {
                "text": "검색된 문서 조각",
                "metadata": {...},
                "distance": 0.123
            }
        ]
    """

    collection = get_or_create_collection(
        collection_name=collection_name,
        chroma_dir=chroma_dir,
    )

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    search_results = []

    for text, metadata, distance in zip(documents, metadatas, distances):
        search_results.append(
            {
                "text": text,
                "metadata": metadata,
                "distance": distance,
            }
        )

    return search_results


def reset_vector_store(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chroma_dir: str = DEFAULT_CHROMA_DIR,
) -> None:
    """
    현재 collection을 삭제하고 새로 만드는 함수.
    """

    client = get_chroma_client(chroma_dir)

    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "PDF RAG chatbot document collection"},
    )