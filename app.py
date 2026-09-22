import os

import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from supabase import Client, create_client


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

if not api_url:
    try:
        api_url = st.secrets.get(
            "RAG_API_URL",
            DEFAULT_API_URL,
        )
    except StreamlitSecretNotFoundError:
        api_url = DEFAULT_API_URL

api_url = api_url.rstrip("/")


# -----------------------------
# Supabase configuration
# -----------------------------

def get_secret(name: str, default: str = "") -> str:
    """Read a Streamlit secret, then fall back to an environment variable."""
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except StreamlitSecretNotFoundError:
        pass

    return os.getenv(name, default)


SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_ANON_KEY = get_secret("SUPABASE_ANON_KEY")


@st.cache_resource
def get_supabase_client() -> Client | None:
    """Create the Supabase client when configuration is available."""
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return None

    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


supabase = get_supabase_client()


# -----------------------------
# Authentication
# -----------------------------

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

if "supabase_session" not in st.session_state:
    st.session_state.supabase_session = None

if "supabase_user" not in st.session_state:
    st.session_state.supabase_user = None


def current_access_token() -> str:
    """Return the current Supabase access token, if available."""
    session = st.session_state.get("supabase_session")

    if not session:
        return ""

    if isinstance(session, dict):
        return session.get("access_token", "")

    return getattr(session, "access_token", "") or ""


def handle_auth_response(session) -> None:
    """Store a Supabase authentication session in Streamlit state."""
    if session is None:
        return

    st.session_state.supabase_session = session

    user = getattr(session, "user", None)

    if user is None and isinstance(session, dict):
        user = session.get("user")

    st.session_state.supabase_user = user


def sign_out() -> None:
    """Sign out and clear the local authentication state."""
    if supabase is not None:
        try:
            supabase.auth.sign_out()
        except Exception:
            pass

    st.session_state.supabase_session = None
    st.session_state.supabase_user = None
    st.session_state.api_key = ""
    st.session_state.pop("documents", None)
    st.session_state.pop("last_answer", None)


with st.sidebar:
    st.header("🔐 Account")

    if supabase is None:
        st.warning(
            "Supabase authentication is not configured yet. "
            "Set SUPABASE_URL and SUPABASE_ANON_KEY in Streamlit secrets."
        )
    elif st.session_state.supabase_user is None:
        login_tab, register_tab = st.tabs(["Login", "Register"])

        with login_tab:
            login_email = st.text_input("Email", key="login_email")
            login_password = st.text_input(
                "Password",
                type="password",
                key="login_password",
            )

            if st.button("🔑 Login", use_container_width=True, type="primary"):
                if not login_email.strip() or not login_password:
                    st.error("Enter your email and password.")
                else:
                    try:
                        response = supabase.auth.sign_in_with_password(
                            {
                                "email": login_email.strip(),
                                "password": login_password,
                            }
                        )
                        handle_auth_response(response.session)

                        if response.session:
                            st.success("Login successful.")
                            st.rerun()
                        else:
                            st.error(
                                "Login did not return a session. "
                                "If email confirmation is enabled, confirm "
                                "your email first."
                            )
                    except Exception as exc:
                        st.error(f"Login failed: {exc}")

        with register_tab:
            register_email = st.text_input("Email", key="register_email")
            register_password = st.text_input(
                "Password",
                type="password",
                key="register_password",
            )
            register_password_confirm = st.text_input(
                "Confirm password",
                type="password",
                key="register_password_confirm",
            )

            if st.button("📝 Create Account", use_container_width=True):
                if not register_email.strip():
                    st.error("Enter your email.")
                elif len(register_password) < 6:
                    st.error("Password must contain at least 6 characters.")
                elif register_password != register_password_confirm:
                    st.error("Passwords do not match.")
                else:
                    try:
                        response = supabase.auth.sign_up(
                            {
                                "email": register_email.strip(),
                                "password": register_password,
                            }
                        )
                        if response.session:
                            handle_auth_response(response.session)
                            st.success("Account created.")
                            st.rerun()
                        else:
                            st.success(
                                "Account created. Check your email and confirm "
                                "your account before logging in."
                            )
                    except Exception as exc:
                        st.error(f"Registration failed: {exc}")
    else:
        user = st.session_state.supabase_user
        user_email = getattr(user, "email", None)

        if user_email is None and isinstance(user, dict):
            user_email = user.get("email")

        st.success(f"Logged in as {user_email or 'authenticated user'}")

        if st.button("🚪 Logout", use_container_width=True):
            sign_out()
            st.rerun()

    st.divider()

    st.header("⚙️ Connection")

    # Temporary compatibility with the current API-key backend.
    api_key_input = st.text_input(
        "API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="Enter your API key",
        help=(
            "Temporary compatibility with the current backend. "
            "Supabase authentication is being added in stages."
        ),
    )

    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input

    if st.session_state.api_key:
        st.success("API key configured.")
    elif current_access_token():
        st.info("Supabase session active.")
    else:
        st.warning("Authentication is required to access the current backend.")

    st.divider()

# -----------------------------
# Header
# -----------------------------

st.title("📄 Enterprise Document Intelligence & RAG Assistant")
st.caption("AI-powered document search, retrieval, and grounded answers.")


# -----------------------------
# Backend status
# -----------------------------

with st.sidebar:
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
    """Build backend authentication headers."""
    headers: dict[str, str] = {}

    token = current_access_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    if st.session_state.api_key:
        headers["X-API-Key"] = st.session_state.api_key

    return headers


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
    """Check that at least one supported authentication mechanism exists."""
    if current_access_token():
        return True

    if st.session_state.api_key:
        return True

    st.error(
        "Authentication required. "
        "Log in with Supabase or enter an API key in the sidebar."
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

                document_id = document.get("document_id")

                if document_id:
                    if st.button(
                        "🗑️ Delete",
                        key=f"delete-{document_id}",
                        use_container_width=True,
                    ):
                        try:
                            with st.spinner(
                                f"Deleting {document.get('source', 'document')}..."
                            ):
                                response = requests.delete(
                                    f"{api_url}/documents/{document_id}",
                                    headers=api_headers(),
                                    timeout=60,
                                )

                            if response.ok:
                                st.success(
                                    f"{document.get('source', 'Document')} deleted successfully."
                                )

                                # Remove the deleted document from the
                                # current workspace's cached list.
                                st.session_state["documents"] = [
                                    item
                                    for item in st.session_state.get(
                                        "documents", []
                                    )
                                    if item.get("document_id") != document_id
                                ]

                                st.rerun()
                            else:
                                show_api_error(response)

                        except requests.RequestException as exc:
                            st.error(
                                f"Connection failed: {exc}"
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