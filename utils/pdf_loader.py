from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader


def save_uploaded_file(uploaded_file, upload_dir: str = "data/uploads") -> str:
    """
    Streamlit에서 업로드된 PDF 파일을 로컬 폴더에 저장하는 함수.

    Parameters
    ----------
    uploaded_file:
        st.file_uploader()로 받은 파일 객체
    upload_dir:
        저장할 폴더 경로

    Returns
    -------
    str
        저장된 파일 경로
    """

    upload_path = Path(upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)

    file_path = upload_path / uploaded_file.name

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(file_path)


def load_pdf_pages(file_path: str) -> List[Dict[str, Any]]:
    """
    PDF 파일에서 페이지별 텍스트를 추출하는 함수.

    Returns
    -------
    List[Dict[str, Any]]
        [
            {
                "text": "페이지 텍스트",
                "metadata": {
                    "source": "파일명.pdf",
                    "page": 1
                }
            }
        ]
    """

    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {file_path}")

    reader = PdfReader(str(pdf_path))
    pages = []

    for page_index, page in enumerate(reader.pages):
        text = page.extract_text()

        if text is None:
            text = ""

        text = text.strip()

        if not text:
            continue

        pages.append(
            {
                "text": text,
                "metadata": {
                    "source": pdf_path.name,
                    "page": page_index + 1,
                    "file_path": str(pdf_path),
                },
            }
        )

    return pages


def load_multiple_pdfs(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    여러 PDF 파일을 한 번에 읽어오는 함수.
    """

    all_pages = []

    for file_path in file_paths:
        pages = load_pdf_pages(file_path)
        all_pages.extend(pages)

    return all_pages