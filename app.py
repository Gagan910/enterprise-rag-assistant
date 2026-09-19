import os

import requests
import streamlit as st


DEFAULT_API_URL = "https://enterprise-rag-assistant-ik30.onrender.com"


st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="📄",
    layout="wide",
)


# -----------------------------
# Configuration
# -----------------------------

api_url = os.getenv("RAG_API_URL")
api_key = os.getenv("RAG_API_KEY")

if not api_url or not api_key:
    try:
        api_url = api_url or st.secrets.get(
            "RAG_API_URL",
            DEFAULT_API_URL,
        )
        api_key = api_key or st.secrets.get(
            "RAG_API_KEY",
            "",
        )
    except StreamlitSecretNotFoundError:
        api_url = api_url or DEFAULT_API_URL
        api_key = api_key or ""

api_url = api_url.rstrip("/")


# -----------------------------
# Header
# -----------------------------

st.title("📄 Enterprise Document Intelligence & RAG Assistant")
st.caption("AI-powered document search, retrieval, and grounded answers.")


# -----------------------------
# Authentication status
# -----------------------------

with st.sidebar:
    st.header("⚙️ Connection")

    if api_key:
        st.success("API authentication configured.")
    else:
        st.error("API authentication is not configured.")

    st.divider()

    if st.button(
        "🔌 Check Backend",
        use_container_width=True,
    ):
        try:
            response = requests.get(
                f"{api_url}/health",
                timeout=15,
            )

            if response.ok:
                st.success("Backend is healthy.")
            else:
                st.error(
                    f"Health check failed: {response.status_code}"
                )

        except requests.RequestException as exc:
            st.error(f"Connection failed: {exc}")


def api_headers() -> dict[str, str]:
    return {
        "X-API-Key": api_key,
    }


def show_api_error(response: requests.Response) -> None:
    try:
        detail = response.json().get(
            "detail",
            "Request failed.",
        )
    except ValueError:
        detail = "Request failed."

    st.error(
        f"{response.status_code}: {detail}"
    )


def authentication_available() -> bool:
    if api_key:
        return True

    st.error(
        "Backend API authentication is not configured. "
        "Configure RAG_API_KEY in the frontend environment."
    )

    return False


# -----------------------------
# Main dashboard
# -----------------------------

left_col, main_col = st.columns(
    [1, 2],
    gap="large",
)


# ============================================================
# LEFT: Documents + Upload
# ============================================================

with left_col:
    st.subheader("📚 Documents")

    if st.button(
        "🔄 Refresh Documents",
        use_container_width=True,
    ):
        if authentication_available():
            try:
                response = requests.get(
                    f"{api_url}/documents",
                    headers=api_headers(),
                    timeout=30,
                )

                if response.ok:
                    st.session_state["documents"] = (
                        response.json()
                    )
                else:
                    show_api_error(response)

            except requests.RequestException as exc:
                st.error(
                    f"Connection failed: {exc}"
                )

    documents = st.session_state.get(
        "documents",
        [],
    )

    st.metric(
        "Indexed Documents",
        len(documents),
    )

    if documents:
        for document in documents:
            with st.container(border=True):
                st.write(
                    f"**{document.get('source', 'Unknown')}**"
                )

                st.caption(
                    f"Chunks: "
                    f"{document.get('chunk_count', 0)}"
                )

                st.caption(
                    f"ID: "
                    f"{document.get('document_id', 'Unknown')}"
                )
    else:
        st.info(
            "Click Refresh Documents to load indexed documents."
        )

    st.divider()

    st.subheader("⬆️ Upload")

    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        help="Maximum file size: 10 MB.",
    )

    if st.button(
        "📥 Upload & Index",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    ):
        if authentication_available():
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }

                with st.spinner(
                    "Uploading and indexing..."
                ):
                    response = requests.post(
                        f"{api_url}/documents/upload",
                        headers=api_headers(),
                        files=files,
                        timeout=180,
                    )

                if response.ok:
                    data = response.json()

                    st.success(
                        "Document indexed successfully."
                    )

                    st.write(
                        f"**Source:** "
                        f"{data.get('source', 'Unknown')}"
                    )

                    st.caption(
                        f"Chunks: "
                        f"{data.get('chunk_count', 0)}"
                    )

                    st.session_state.pop(
                        "documents",
                        None,
                    )

                else:
                    show_api_error(response)

            except requests.RequestException as exc:
                st.error(
                    f"Connection failed: {exc}"
                )


# ============================================================
# RIGHT: RAG Query
# ============================================================

with main_col:
    st.subheader("💬 Ask Your Documents")

    question = st.text_area(
        "Your question",
        placeholder=(
            "Example: How many days of paid annual leave "
            "do employees receive?"
        ),
        height=140,
    )

    query_col1, query_col2 = st.columns(
        [1, 1]
    )

    with query_col1:
        top_k = st.slider(
            "Retrieval results",
            min_value=1,
            max_value=20,
            value=5,
        )

    with query_col2:
        st.write("")
        st.write("")

        ask_clicked = st.button(
            "🚀 Ask",
            type="primary",
            use_container_width=True,
            disabled=not question.strip(),
        )

    if ask_clicked:
        if authentication_available():
            payload = {
                "question": question.strip(),
                "top_k": top_k,
            }

            try:
                with st.spinner(
                    "Searching documents and generating answer..."
                ):
                    response = requests.post(
                        f"{api_url}/query",
                        headers=api_headers(),
                        json=payload,
                        timeout=180,
                    )

                if response.ok:
                    st.session_state["last_answer"] = (
                        response.json()
                    )
                else:
                    show_api_error(response)

            except requests.RequestException as exc:
                st.error(
                    f"Connection failed: {exc}"
                )

    # -------------------------
    # Answer
    # -------------------------

    answer_data = st.session_state.get(
        "last_answer"
    )

    if answer_data:
        st.divider()

        st.markdown("### 🧠 Answer")

        st.markdown(
            answer_data.get(
                "answer",
                "No answer returned.",
            )
        )

        provider = answer_data.get("provider")
        fallback_used = answer_data.get(
            "fallback_used",
            False,
        )
        provider_attempts = answer_data.get(
            "provider_attempts",
            [],
        )

        st.markdown("### ⚡ Generation")

        if (
            fallback_used
            and len(provider_attempts) >= 2
        ):
            st.info(
                f"{provider_attempts[0].title()} unavailable → "
                f"{provider_attempts[1].title()} fallback → "
                f"Answer generated"
            )

        elif provider:
            st.success(
                f"Answer generated by "
                f"{provider.title()}"
            )

        if provider_attempts:
            st.caption(
                "Provider attempts: "
                + " → ".join(
                    attempt.title()
                    for attempt in provider_attempts
                )
            )

        # -------------------------
        # Sources
        # -------------------------

        sources = answer_data.get(
            "sources",
            [],
        )

        st.markdown("### 📚 Sources")

        if sources:
            for index, source in enumerate(
                sources,
                start=1,
            ):
                st.markdown(
                    f"**Source {index}** — "
                    f"{source.get('source', 'Unknown')} "
                    f"(chunk "
                    f"{source.get('chunk_id', 'Unknown')})"
                )
        else:
            st.info(
                "No sources were returned."
            )