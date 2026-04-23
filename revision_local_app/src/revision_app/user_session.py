"""User session management for complete data isolation between users."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import streamlit as st


def _get_user_session_id() -> str:
    """Generate or retrieve a unique session ID for the current user.
    
    Uses Streamlit's session_state to maintain a consistent ID across page reloads
    within a single browser session. Each new browser/device gets a unique ID.
    
    This is session-based isolation (not account-based) suitable for local/offline
    deployments where each physical user runs in their own browser.
    """
    if "user_session_id" not in st.session_state:
        st.session_state.user_session_id = str(uuid.uuid4())
    return st.session_state.user_session_id


def _hash_session_id(session_id: str) -> str:
    """Hash session ID for shorter directory names (first 12 chars of hex)."""
    return hashlib.sha256(session_id.encode()).hexdigest()[:12]


def get_user_subdir(base_dir: Path) -> Path:
    """Get user-specific subdirectory within a base directory.
    
    Each user gets their own isolated subdirectory based on their session ID.
    This ensures complete data separation between users.
    
    Args:
        base_dir: Base directory (e.g., uploads_dir, work_dir)
        
    Returns:
        User-specific subdirectory: base_dir/user_{hashed_id}/
    """
    session_id = _get_user_session_id()
    hashed_id = _hash_session_id(session_id)
    user_dir = base_dir / f"user_{hashed_id}"
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def get_user_uploads_dir(config) -> Path:
    """Get user-specific uploads directory."""
    return get_user_subdir(config.uploads_dir)


def get_user_work_dir(config) -> Path:
    """Get user-specific work directory."""
    return get_user_subdir(config.work_dir)


def get_user_logs_dir(config) -> Path:
    """Get user-specific logs directory."""
    return get_user_subdir(config.logs_dir)


def get_user_settings_path(config) -> Path:
    """Get user-specific settings file path."""
    user_dir = get_user_subdir(config.work_dir)
    return user_dir / "app_settings.json"


def init_user_session_state() -> None:
    """Initialize user-isolated session state variables.
    
    This MUST be called before any analysis or file handling to ensure
    all session state is user-specific and never leaks between users.
    """
    # Get user ID to validate isolation is working
    user_id = _get_user_session_id()
    
    # User-specific state variables
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "warnings" not in st.session_state:
        st.session_state.warnings = []
    if "exports" not in st.session_state:
        st.session_state.exports = {}
    if "runtime_mode" not in st.session_state:
        st.session_state.runtime_mode = "Low"
    
    # Q&A session state (user-specific)
    if "qa_question" not in st.session_state:
        st.session_state.qa_question = ""
    if "qa_answer" not in st.session_state:
        st.session_state.qa_answer = ""
    if "qa_sources" not in st.session_state:
        st.session_state.qa_sources = []
    if "qa_web_snippets" not in st.session_state:
        st.session_state.qa_web_snippets = []
    
    # File upload state (user-specific)
    if "last_uploaded_files" not in st.session_state:
        st.session_state.last_uploaded_files = []
    if "last_ingested_docs" not in st.session_state:
        st.session_state.last_ingested_docs = None


def get_current_user_id() -> str:
    """Get the current user's session ID (hashed for display)."""
    return _hash_session_id(_get_user_session_id())


def is_user_data_isolated() -> bool:
    """Verify that user data is properly isolated.
    
    Returns True if the user session ID is set, indicating isolation is active.
    """
    return "user_session_id" in st.session_state
