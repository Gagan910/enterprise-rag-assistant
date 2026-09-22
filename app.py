import os

import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from supabase import Client, create_client
from supabase.lib.client_options import ClientOptions


DEFAULT_API_URL = "https://enterprise-rag-assistant-ik30.onrender.com"


st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="R",
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
FRONTEND_URL = get_secret(
    "RAG_FRONTEND_URL",
    "https://enterprise-rag-frontend-g1bm.onrender.com",
).rstrip("/")


@st.cache_resource
def get_supabase_client() -> Client | None:
    """Create the Supabase client when configuration is available."""
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return None

    return create_client(
        SUPABASE_URL,
        SUPABASE_ANON_KEY,
        options=ClientOptions(flow_type="pkce"),
    )


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

if "password_reset_mode" not in st.session_state:
    st.session_state.password_reset_mode = False


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


# -----------------------------
# Login / Registration page
# -----------------------------

def extract_auth_session(response):
    """Extract the session object from a Supabase auth response."""
    if response is None:
        return None

    session = getattr(response, "session", None)

    if session is not None:
        return session

    if isinstance(response, dict):
        return response.get("session")

    return None


def process_auth_code() -> bool:
    """Exchange a Supabase PKCE auth code returned by an email redirect."""
    code = st.query_params.get("code")

    if not code or supabase is None:
        return False

    try:
        response = supabase.auth.exchange_code_for_session(
            {"auth_code": code}
        )
        session = extract_auth_session(response)

        if session is None:
            st.error("Authentication callback did not return a session.")
            return False

        handle_auth_response(session)
        st.query_params.clear()
        return True

    except Exception as exc:
        st.error(f"Authentication callback failed: {exc}")
        return False


def get_user_email() -> str:
    """Return the authenticated user's email."""
    user = st.session_state.get("supabase_user")

    if not user:
        return ""

    if isinstance(user, dict):
        return user.get("email", "") or ""

    return getattr(user, "email", "") or ""


def render_auth_page() -> None:
    """Render the authentication-only page."""
    st.markdown(
        """
        <style>
        .auth-shell {
            max-width: 620px;
            margin: 5vh auto 0 auto;
            text-align: center;
        }

        .auth-subtitle {
            color: #6b7280;
            margin-bottom: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="auth-shell">', unsafe_allow_html=True)
    st.title("Enterprise RAG Assistant")
    st.markdown(
        '<div class="auth-subtitle">'
        "Sign in to securely access your documents and AI assistant."
        "</div>",
        unsafe_allow_html=True,
    )

    if supabase is None:
        st.error(
            "Authentication is not configured. "
            "Please configure SUPABASE_URL and SUPABASE_ANON_KEY."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    auth_container = st.container(border=True)

    with auth_container:
        login_tab, register_tab = st.tabs(["Login", "Register"])

        with login_tab:
            st.subheader("Welcome back")

            login_email = st.text_input(
                "Email",
                key="login_email",
                placeholder="you@example.com",
            )

            login_password = st.text_input(
                "Password",
                type="password",
                key="login_password",
                placeholder="Enter your password",
            )

            if st.button(
                "Login",
                type="primary",
                use_container_width=True,
                key="login_button",
            ):
                if not login_email.strip() or not login_password:
                    st.warning("Enter both email and password.")
                else:
                    try:
                        with st.spinner("Signing in..."):
                            response = supabase.auth.sign_in_with_password(
                                {
                                    "email": login_email.strip(),
                                    "password": login_password,
                                }
                            )

                        session = extract_auth_session(response)

                        if session is None:
                            st.error(
                                "Login did not return an authenticated session."
                            )
                        else:
                            handle_auth_response(session)
                            st.rerun()

                    except Exception as exc:
                        st.error(f"Login failed: {exc}")

            st.divider()
            st.caption("Forgot your password?")

            forgot_email = st.text_input(
                "Account email",
                key="forgot_email",
                placeholder="you@example.com",
            )

            if st.button(
                "Send reset email",
                use_container_width=True,
                key="forgot_password_button",
            ):
                email = forgot_email.strip()

                if not email:
                    st.warning("Enter the email address for your account.")
                else:
                    try:
                        with st.spinner("Sending password reset email..."):
                            supabase.auth.reset_password_for_email(
                                email,
                                {
                                    "redirect_to": (
                                        f"{FRONTEND_URL}/?mode=recovery"
                                    )
                                },
                            )

                        st.success(
                            "If an account exists for that email, a password "
                            "reset link has been sent. Check your inbox."
                        )
                    except Exception as exc:
                        st.error(f"Could not send reset email: {exc}")

        with register_tab:
            st.subheader("Create your account")

            register_email = st.text_input(
                "Email",
                key="register_email",
                placeholder="you@example.com",
            )

            register_password = st.text_input(
                "Password",
                type="password",
                key="register_password",
                placeholder="Minimum 6 characters",
            )

            register_confirm = st.text_input(
                "Confirm password",
                type="password",
                key="register_confirm",
                placeholder="Re-enter your password",
            )

            if st.button(
                "Create account",
                type="primary",
                use_container_width=True,
                key="register_button",
            ):
                email = register_email.strip()

                if not email or not register_password:
                    st.warning("Enter an email and password.")
                elif len(register_password) < 6:
                    st.warning("Password must be at least 6 characters.")
                elif register_password != register_confirm:
                    st.warning("Passwords do not match.")
                else:
                    try:
                        with st.spinner("Creating your account..."):
                            response = supabase.auth.sign_up(
                                {
                                    "email": email,
                                    "password": register_password,
                                }
                            )

                        session = extract_auth_session(response)

                        if session is not None:
                            handle_auth_response(session)
                            st.success("Account created successfully.")
                            st.rerun()
                        else:
                            st.success(
                                "Account created. Please check your email "
                                "and confirm your account before logging in."
                            )

                    except Exception as exc:
                        st.error(f"Registration failed: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)


# Process email confirmation / password-recovery callbacks before
# rendering the authentication gate. Supabase PKCE returns an auth code
# in the query string, which the Python client exchanges for a session.
callback_mode = st.query_params.get("mode")
if callback_mode == "recovery" and st.query_params.get("code"):
    if process_auth_code():
        st.session_state.password_reset_mode = True
        st.rerun()

elif st.query_params.get("code") and not current_access_token():
    if process_auth_code():
        st.rerun()


def render_password_reset_page() -> None:
    """Render the authenticated password-change form after recovery."""
    st.markdown(
        """
        <style>
        .auth-shell {
            max-width: 620px;
            margin: 5vh auto 0 auto;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="auth-shell">', unsafe_allow_html=True)
    st.title("Reset your password")
    st.caption("Choose a new password for your Enterprise RAG Assistant account.")

    with st.container(border=True):
        new_password = st.text_input(
            "New password",
            type="password",
            key="new_password",
            placeholder="Minimum 6 characters",
        )
        confirm_password = st.text_input(
            "Confirm new password",
            type="password",
            key="confirm_new_password",
            placeholder="Re-enter your new password",
        )

        if st.button(
            "Update password",
            type="primary",
            use_container_width=True,
            key="update_password_button",
        ):
            if len(new_password) < 6:
                st.warning("Password must be at least 6 characters.")
            elif new_password != confirm_password:
                st.warning("Passwords do not match.")
            else:
                try:
                    with st.spinner("Updating password..."):
                        response = supabase.auth.update_user(
                            {"password": new_password}
                        )

                    session = extract_auth_session(response)
                    if session is not None:
                        handle_auth_response(session)

                    st.session_state.password_reset_mode = False
                    st.session_state.pop("new_password", None)
                    st.session_state.pop("confirm_new_password", None)
                    st.success("Password updated successfully. You are now signed in.")
                    st.rerun()

                except Exception as exc:
                    st.error(f"Could not update password: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)


# Authentication is the first screen. Nothing from the RAG dashboard
# is rendered until a valid Supabase session exists.
if st.session_state.password_reset_mode:
    if current_access_token():
        render_password_reset_page()
        st.stop()
    st.session_state.password_reset_mode = False

if not current_access_token():
    render_auth_page()
    st.stop()


# -----------------------------
# Authenticated account sidebar
# -----------------------------

with st.sidebar:
    st.header("Account")

    email = get_user_email()
    if email:
        st.caption("Signed in as")
        st.write(f"**{email}**")
    else:
        st.success("Authenticated with Supabase.")

    if st.button("Logout", use_container_width=True):
        sign_out()

    st.divider()

# -----------------------------
# Header
# -----------------------------

st.title("Enterprise Document Intelligence & RAG Assistant")
st.caption("AI-powered document search, retrieval, and grounded answers.")


# -----------------------------
# Backend status
# -----------------------------

with st.sidebar:
    if st.button(
        "Check Backend",
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
    """Build backend authentication headers using the Supabase JWT."""
    token = current_access_token()

    if not token:
        return {}

    return {
        "Authorization": f"Bearer {token}",
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
    """Check that the current user has an authenticated Supabase session."""
    if current_access_token():
        return True

    st.error("Authentication required. Please log in again.")
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
    st.subheader("Documents")

    if st.button(
        "Refresh Documents",
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

    st.subheader("Upload")

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
    st.subheader("Ask Your Documents")

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

        st.markdown("### Answer")

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

        st.markdown("### Generation")

        if (
            fallback_used
            and len(provider_attempts) >= 2
        ):
            st.info(
                f"{provider_attempts[0].title()} unavailable â†’ "
                f"{provider_attempts[1].title()} fallback â†’ "
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

        st.markdown("### Sources")

        if sources:
            for index, source in enumerate(
                sources,
                start=1,
            ):
                st.markdown(
                    f"**Source {index}** â€” "
                    f"{source.get('source', 'Unknown')} "
                    f"(chunk "
                    f"{source.get('chunk_id', 'Unknown')})"
                )
        else:
            st.info(
                "No sources were returned."
            )
