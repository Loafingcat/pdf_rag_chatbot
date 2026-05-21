from functools import lru_cache

from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """
    Embedding 모델을 로드하는 함수.

    @lru_cache를 사용한 이유:
    Streamlit은 버튼 클릭이나 입력 변경 시 스크립트가 다시 실행된다.
    모델을 매번 새로 로드하면 속도가 느려지므로, 한 번 로드한 모델을 재사용한다.
    """

    model = SentenceTransformer(model_name)
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    여러 문장을 embedding 벡터로 변환하는 함수.
    """

    model = load_embedding_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    사용자 질문 하나를 embedding 벡터로 변환하는 함수.
    """

    model = load_embedding_model()

    embedding = model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embedding.tolist()