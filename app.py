import os
from pathlib import Path

import streamlit as st

from utils.pdf_loader import save_uploaded_file, load_multiple_pdfs
from utils.text_splitter import split_documents
from utils.vector_store import add_chunks_to_vector_store, reset_vector_store
from utils.rag_chain import answer_question


APP_TITLE = "PDF RAG Chatbot"
UPLOAD_DIR = "data/uploads"
CHROMA_DIR = "data/chroma_db"


def init_directories():
    """
    필요한 폴더를 자동 생성한다.
    """

    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)


def init_session_state():
    """
    Streamlit session_state 초기화.

    Streamlit은 사용자가 버튼을 누르거나 입력할 때마다
    app.py가 위에서 아래로 다시 실행된다.

    그래서 대화 기록 같은 값은 st.session_state에 저장해야 유지된다.
    """

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    if "indexed" not in st.session_state:
        st.session_state.indexed = False


def render_sidebar():
    """
    왼쪽 사이드바 UI.
    PDF 업로드, 인덱싱, DB 초기화 기능을 담당한다.
    """

    st.sidebar.header("문서 업로드")

    uploaded_files = st.sidebar.file_uploader(
        "PDF 파일을 업로드하세요",
        type=["pdf"],
        accept_multiple_files=True,
    )

    top_k = st.sidebar.slider(
        "검색할 문서 조각 수 TOP-K",
        min_value=1,
        max_value=8,
        value=4,
    )

    use_gemini = st.sidebar.checkbox(
        "Gemini로 답변 생성",
        value=True,
        help="GOOGLE_API_KEY가 없으면 자동으로 fallback 답변을 보여줍니다.",
    )

    st.sidebar.divider()

    index_button = st.sidebar.button("PDF 인덱싱하기", type="primary")
    reset_button = st.sidebar.button("벡터DB 초기화")

    if reset_button:
        reset_vector_store()
        st.session_state.messages = []
        st.session_state.uploaded_files = []
        st.session_state.indexed = False
        st.sidebar.success("벡터DB를 초기화했습니다.")

    if index_button:
        if not uploaded_files:
            st.sidebar.warning("먼저 PDF 파일을 업로드하세요.")
        else:
            with st.spinner("PDF를 읽고 벡터DB에 저장하는 중입니다..."):
                saved_paths = []

                for uploaded_file in uploaded_files:
                    saved_path = save_uploaded_file(
                        uploaded_file=uploaded_file,
                        upload_dir=UPLOAD_DIR,
                    )
                    saved_paths.append(saved_path)

                pages = load_multiple_pdfs(saved_paths)

                if not pages:
                    st.sidebar.error("PDF에서 텍스트를 추출하지 못했습니다.")

                    # 중요:
                    # 여기서도 반드시 top_k, use_gemini를 반환해야 함
                    return top_k, use_gemini

                chunks = split_documents(
                    pages=pages,
                    chunk_size=700,
                    chunk_overlap=120,
                )

                saved_count = add_chunks_to_vector_store(chunks)

                st.session_state.uploaded_files = [
                    Path(path).name for path in saved_paths
                ]
                st.session_state.indexed = True

                st.sidebar.success(f"{saved_count}개의 문서 조각을 저장했습니다.")

    st.sidebar.divider()

    st.sidebar.subheader("현재 문서")

    if st.session_state.uploaded_files:
        for file_name in st.session_state.uploaded_files:
            st.sidebar.write(f"- {file_name}")
    else:
        st.sidebar.write("아직 인덱싱된 문서가 없습니다.")

    st.sidebar.divider()

    api_key_exists = bool(os.getenv("GOOGLE_API_KEY"))

    if api_key_exists:
        st.sidebar.success("GOOGLE_API_KEY 감지됨")
    else:
        st.sidebar.warning("GOOGLE_API_KEY 없음")

    # 중요:
    # 함수 마지막에서 반드시 2개 값을 반환해야 함
    return top_k, use_gemini


def render_chat_history():
    """
    기존 대화 이력을 화면에 출력한다.
    """

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                render_sources(message["sources"])


def render_sources(sources):
    """
    답변에 사용된 source chunk를 펼쳐보기 형태로 보여준다.
    """

    with st.expander("참고한 문서 조각 보기"):
        for index, source in enumerate(sources, start=1):
            metadata = source["metadata"]
            text = source["text"]
            distance = source.get("distance", 0)

            file_name = metadata.get("source", "unknown")
            page = metadata.get("page", "unknown")
            chunk_index = metadata.get("chunk_index", "unknown")

            st.markdown(f"### Source {index}")
            st.write(f"파일명: `{file_name}`")
            st.write(f"페이지: `{page}`")
            st.write(f"Chunk Index: `{chunk_index}`")
            st.write(f"Distance: `{distance:.4f}`")
            st.text_area(
                label=f"문서 내용 {index}",
                value=text,
                height=180,
                key=f"source_{index}_{file_name}_{page}_{chunk_index}_{distance}",
            )


def main():
    init_directories()
    init_session_state()

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📄",
        layout="wide",
    )

    st.title("PDF RAG Chatbot")
    st.caption("PDF를 업로드하고, 문서 내용을 기반으로 질문해보세요.")

    top_k, use_gemini = render_sidebar()

    if not st.session_state.indexed:
        st.info("왼쪽 사이드바에서 PDF를 업로드한 뒤, `PDF 인덱싱하기` 버튼을 눌러주세요.")

    render_chat_history()

    user_question = st.chat_input("PDF 내용에 대해 질문하세요.")

    if user_question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question,
            }
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            if not st.session_state.indexed:
                answer = "먼저 PDF를 업로드하고 인덱싱을 진행해주세요."
                sources = []
            else:
                with st.spinner("문서에서 관련 내용을 검색하고 답변을 생성하는 중입니다..."):
                    result = answer_question(
                        question=user_question,
                        top_k=top_k,
                        use_gemini=use_gemini,
                    )

                    answer = result["answer"]
                    sources = result["sources"]

            st.markdown(answer)

            if sources:
                render_sources(sources)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )


if __name__ == "__main__":
    main()