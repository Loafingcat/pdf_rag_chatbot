import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from google import genai

from utils.vector_store import search_similar_chunks


load_dotenv()


def build_context(search_results: List[Dict[str, Any]]) -> str:
    """
    검색된 chunk들을 LLM에 넣을 context 문자열로 변환한다.
    """

    context_parts = []

    for index, result in enumerate(search_results, start=1):
        metadata = result["metadata"]
        text = result["text"]

        source = metadata.get("source", "unknown")
        page = metadata.get("page", "unknown")

        context_parts.append(
            f"[문서 {index}]\n"
            f"출처: {source}, 페이지: {page}\n"
            f"내용:\n{text}"
        )

    return "\n\n".join(context_parts)


def build_prompt(question: str, context: str) -> str:
    """
    Gemini에게 전달할 프롬프트를 만드는 함수.
    """

    prompt = f"""
당신은 PDF 문서를 기반으로 답변하는 RAG 챗봇입니다.

아래 [문서 내용]만 참고해서 사용자의 질문에 답변하세요.

규칙:
1. 문서 내용에 근거가 있는 내용만 답변하세요.
2. 문서에 없는 내용은 추측하지 말고 "업로드된 문서에서 관련 내용을 찾지 못했습니다."라고 답변하세요.
3. 답변은 한국어로 작성하세요.
4. 가능하면 핵심을 먼저 말하고, 그다음 근거를 설명하세요.
5. 답변 마지막에는 참고한 문서명과 페이지를 간단히 적어주세요.

[문서 내용]
{context}

[사용자 질문]
{question}

[답변]
""".strip()

    return prompt


def call_gemini_llm(prompt: str) -> str:
    """
    Google AI Studio의 Gemini API를 사용해서 답변을 생성하는 함수.

    필요한 환경변수:
    - GOOGLE_API_KEY
    - GEMINI_MODEL
    """

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY가 설정되어 있지 않습니다.")

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    if not response.text:
        return "Gemini 응답을 생성하지 못했습니다."

    return response.text


def fallback_answer(question: str, search_results: List[Dict[str, Any]]) -> str:
    """
    Gemini API Key가 없거나 호출 실패 시 사용하는 fallback 답변.
    """

    if not search_results:
        return "업로드된 문서에서 관련 내용을 찾지 못했습니다."

    lines = []
    lines.append("Gemini API Key가 없거나 호출에 실패하여, 질문과 가장 관련 있는 문서 조각을 보여드립니다.")
    lines.append("")
    lines.append(f"질문: {question}")
    lines.append("")

    for index, result in enumerate(search_results, start=1):
        metadata = result["metadata"]
        text = result["text"]

        source = metadata.get("source", "unknown")
        page = metadata.get("page", "unknown")
        distance = result.get("distance", 0)

        preview = text[:500].replace("\n", " ")

        lines.append(f"관련 문서 {index}")
        lines.append(f"- 출처: {source}")
        lines.append(f"- 페이지: {page}")
        lines.append(f"- 거리값: {distance:.4f}")
        lines.append(f"- 내용: {preview}")
        lines.append("")

    return "\n".join(lines)


def answer_question(
    question: str,
    top_k: int = 4,
    use_gemini: bool = True,
) -> Dict[str, Any]:
    """
    RAG 전체 흐름을 실행하는 함수.

    1. 사용자 질문 입력
    2. 질문과 유사한 chunk 검색
    3. 검색 결과를 context로 구성
    4. Gemini 프롬프트 생성
    5. Gemini 답변 생성
    6. 답변과 source chunk 반환
    """

    search_results = search_similar_chunks(
        query=question,
        top_k=top_k,
    )

    if not search_results:
        return {
            "answer": "업로드된 문서에서 관련 내용을 찾지 못했습니다.",
            "sources": [],
        }

    context = build_context(search_results)
    prompt = build_prompt(question=question, context=context)

    if use_gemini:
        try:
            answer = call_gemini_llm(prompt)
        except Exception as e:
            answer = fallback_answer(question, search_results)
            answer += f"\n\nGemini 호출 실패 사유: {str(e)}"
    else:
        answer = fallback_answer(question, search_results)

    return {
        "answer": answer,
        "sources": search_results,
    }