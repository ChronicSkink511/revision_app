from __future__ import annotations

import re
import sys
from dataclasses import replace
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bootstrap import ensure_project_tree
from revision_app.analysis import run_analysis
from revision_app.config import load_config
from revision_app.export import export_bundle
from revision_app.image_analysis import EngineeringImageAnalyzer
from revision_app.ingestion import ingest_uploaded_files
from revision_app.llm import LocalLLMClient
from revision_app.logging_utils import setup_logging
from revision_app.parsing import parse_documents
from revision_app.settings_store import load_settings, save_settings
from revision_app.user_session import (
    get_current_user_id,
    get_user_settings_path,
    get_user_uploads_dir,
    get_user_work_dir,
    init_user_session_state,
)
from revision_app.web import gather_trusted_web_context


def _parse_csv_values(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _init_state(default_mode: str = "Low") -> None:
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "warnings" not in st.session_state:
        st.session_state.warnings = []
    if "exports" not in st.session_state:
        st.session_state.exports = {}
    if "runtime_mode" not in st.session_state:
        st.session_state.runtime_mode = default_mode
    elif st.session_state.runtime_mode not in {"Low", "High"}:
        st.session_state.runtime_mode = default_mode
    if "qa_question" not in st.session_state:
        st.session_state.qa_question = ""
    if "qa_answer" not in st.session_state:
        st.session_state.qa_answer = ""
    if "qa_sources" not in st.session_state:
        st.session_state.qa_sources = []
    if "qa_web_snippets" not in st.session_state:
        st.session_state.qa_web_snippets = []


def _build_qa_context(documents, question: str, max_chars: int = 5000) -> tuple[str, list[str]]:
    tokens = {word.lower() for word in re.findall(r"[A-Za-z0-9_+-]{3,}", question)}
    chunks: list[tuple[int, str, str]] = []

    for doc in documents:
        source = doc.source_path.name
        for segment in doc.text.splitlines():
            line = segment.strip()
            if not line:
                continue
            score = sum(1 for token in tokens if token in line.lower())
            if score > 0:
                chunks.append((score, source, line))

    if not chunks:
        fallback_docs = documents[:4]
        fallback_text = "\n\n".join(doc.text[:700] for doc in fallback_docs if doc.text)
        fallback_sources = [doc.source_path.name for doc in fallback_docs]
        return fallback_text[:max_chars], fallback_sources

    chunks.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    sources: list[str] = []
    used_chars = 0

    for _, source, line in chunks:
        addition = f"[{source}] {line}"
        if used_chars + len(addition) + 1 > max_chars:
            break
        selected.append(addition)
        used_chars += len(addition) + 1
        if source not in sources:
            sources.append(source)
        if len(selected) >= 14:
            break

    return "\n".join(selected), sources


@st.cache_resource(show_spinner=False)
def _get_llm_client(config, _logger):
    return LocalLLMClient(config=config, logger=_logger)


@st.cache_resource(show_spinner=False)
def _get_image_analyzer(_llm_client, _logger, tesseract_cmd: str | None):
    return EngineeringImageAnalyzer(
        llm_client=_llm_client,
        logger=_logger,
        tesseract_cmd=tesseract_cmd,
    )


def main() -> None:
    ensure_project_tree()
    config = load_config(PROJECT_ROOT)

    st.set_page_config(page_title="Local Revision Assistant", layout="wide")

    init_user_session_state()
    user_id = get_current_user_id()
    user_uploads_dir = get_user_uploads_dir(config)
    user_work_dir = get_user_work_dir(config)
    user_settings_path = get_user_settings_path(config)

    logger = setup_logging(user_work_dir / "logs")
    persisted_settings = load_settings(config, user_settings_path)

    def setting(name: str, default):
        return persisted_settings.get(name, default)

    st.title("Local Engineering Revision Assistant")
    st.caption("Offline, low-resource study generation using local GGUF models.")
    _init_state(default_mode=setting("default_mode", "Low"))

    effective_config = replace(
        config,
        gguf_model_path=config.models_dir / setting("gguf_model", config.gguf_model_path.name),
        llm_ctx_size=setting("llm_ctx_size", config.llm_ctx_size),
        llm_max_tokens=setting("llm_max_tokens", config.llm_max_tokens),
        llm_threads=setting("llm_threads", config.llm_threads),
        llm_temperature=setting("llm_temperature", config.llm_temperature),
        enable_embeddings=setting("enable_embeddings", config.enable_embeddings),
        embedding_model_name=setting("embedding_model_name", config.embedding_model_name),
        tesseract_cmd=(setting("tesseract_cmd", "") or None),
    )

    llm = _get_llm_client(config=effective_config, _logger=logger)
    image_analyzer = _get_image_analyzer(
        _llm_client=llm,
        _logger=logger,
        tesseract_cmd=effective_config.tesseract_cmd,
    )

    with st.sidebar:
        st.subheader("Runtime")
        st.caption(f"Session: {user_id}")

        with st.expander("Settings", expanded=True):
            with st.form("settings_form"):
                default_mode = st.selectbox(
                    "Default resource mode",
                    options=["Low", "High"],
                    index=0 if setting("default_mode", "Low") == "Low" else 1,
                    help="Applied as default each time the app starts.",
                )
                gguf_model = st.text_input("GGUF model filename", value=setting("gguf_model", config.gguf_model_path.name))
                llm_ctx_size = st.number_input("LLM context size", min_value=512, max_value=4096, value=int(setting("llm_ctx_size", config.llm_ctx_size)), step=128)
                llm_max_tokens = st.number_input("LLM max tokens", min_value=64, max_value=1024, value=int(setting("llm_max_tokens", config.llm_max_tokens)), step=32)
                llm_threads = st.number_input("LLM CPU threads", min_value=1, max_value=32, value=int(setting("llm_threads", config.llm_threads)), step=1)
                llm_temperature = st.slider("LLM temperature", min_value=0.0, max_value=1.2, value=float(setting("llm_temperature", config.llm_temperature)), step=0.05)
                enable_embeddings = st.checkbox("Enable embedding topic merge", value=bool(setting("enable_embeddings", config.enable_embeddings)))
                embedding_model_name = st.text_input("Embedding model", value=setting("embedding_model_name", config.embedding_model_name))
                tesseract_cmd = st.text_input("Tesseract command path (optional)", value=setting("tesseract_cmd", ""))
                allow_web_browsing = st.checkbox(
                    "Allow trusted internet sources in Q&A",
                    value=bool(setting("allow_web_browsing", False)),
                    help="When enabled, document Q&A can augment context from trusted web domains.",
                )
                trusted_domains = st.text_input(
                    "Trusted domains (comma-separated)",
                    value=setting("trusted_domains", "wikipedia.org,nist.gov,nasa.gov,asme.org,ieee.org,mit.edu"),
                    help="Only these domains are used for web augmentation.",
                )
                trusted_urls = st.text_area(
                    "Optional trusted URLs (comma-separated)",
                    value=setting("trusted_urls", ""),
                    height=70,
                    help="Provide specific trusted pages to use during Q&A.",
                )

                low_col, high_col = st.columns(2)
                with low_col:
                    max_topics_low = st.number_input("Low: max topics", min_value=3, max_value=12, value=int(setting("max_topics_low", 6)), step=1)
                    quiz_per_topic_low = st.number_input("Low: quiz questions/topic", min_value=5, max_value=10, value=int(setting("quiz_per_topic_low", 6)), step=1)
                with high_col:
                    max_topics_high = st.number_input("High: max topics", min_value=3, max_value=12, value=int(setting("max_topics_high", 10)), step=1)
                    quiz_per_topic_high = st.number_input("High: quiz questions/topic", min_value=5, max_value=10, value=int(setting("quiz_per_topic_high", 8)), step=1)

                save_clicked = st.form_submit_button("Save Settings", type="primary")
                reset_clicked = st.form_submit_button("Reset to Defaults")

            if save_clicked:
                saved = save_settings(
                    config,
                    {
                        "default_mode": default_mode,
                        "gguf_model": gguf_model,
                        "llm_ctx_size": llm_ctx_size,
                        "llm_max_tokens": llm_max_tokens,
                        "llm_threads": llm_threads,
                        "llm_temperature": llm_temperature,
                        "enable_embeddings": enable_embeddings,
                        "embedding_model_name": embedding_model_name,
                        "tesseract_cmd": tesseract_cmd,
                        "allow_web_browsing": allow_web_browsing,
                        "trusted_domains": trusted_domains,
                        "trusted_urls": trusted_urls,
                        "max_topics_low": max_topics_low,
                        "quiz_per_topic_low": quiz_per_topic_low,
                        "max_topics_high": max_topics_high,
                        "quiz_per_topic_high": quiz_per_topic_high,
                    },
                    user_settings_path,
                )
                st.session_state.runtime_mode = saved["default_mode"]
                _get_llm_client.clear()
                _get_image_analyzer.clear()
                st.success("Settings saved.")
                st.rerun()

            if reset_clicked:
                saved = save_settings(config, {}, user_settings_path)
                st.session_state.runtime_mode = saved["default_mode"]
                _get_llm_client.clear()
                _get_image_analyzer.clear()
                st.success("Settings reset to defaults.")
                st.rerun()

        mode = st.selectbox(
            "Resource mode",
            options=["Low", "High"],
            index=0 if st.session_state.runtime_mode == "Low" else 1,
            help="Low minimizes CPU/RAM usage. High increases output depth at higher compute cost.",
            key="runtime_mode",
        )
        st.write(f"LLM status: {llm.status}")
        st.write(f"Model path: {effective_config.gguf_model_path}")
        st.write(f"Embeddings enabled: {effective_config.enable_embeddings}")

    if mode == "Low":
        runtime_config = replace(
            effective_config,
            max_topics=min(effective_config.max_topics, setting("max_topics_low", 6)),
            quiz_questions_per_topic=min(effective_config.quiz_questions_per_topic, setting("quiz_per_topic_low", 6)),
        )
    else:
        runtime_config = replace(
            effective_config,
            max_topics=max(effective_config.max_topics, setting("max_topics_high", 10)),
            quiz_questions_per_topic=max(effective_config.quiz_questions_per_topic, setting("quiz_per_topic_high", 8)),
        )

    uploads = st.file_uploader(
        "Upload files or ZIP folder archive",
        accept_multiple_files=True,
        type=["pdf", "docx", "pptx", "txt", "md", "png", "jpg", "jpeg", "bmp", "tif", "tiff", "zip"],
    )

    if st.button("Run Analysis", type="primary"):
        if not uploads:
            st.warning("Please upload at least one file.")
        else:
            with st.spinner("Ingesting and analyzing files..."):
                persisted, ingest_warnings = ingest_uploaded_files(uploads, user_uploads_dir, logger)
                documents, parse_warnings = parse_documents(persisted, image_analyzer, logger)

                if not documents:
                    st.error("No readable content found in uploaded material.")
                    st.session_state.analysis_result = None
                    st.session_state.warnings = ingest_warnings + parse_warnings
                    st.session_state.exports = {}
                else:
                    result = run_analysis(documents, llm_client=llm, config=runtime_config, logger=logger)
                    st.session_state.analysis_result = result
                    st.session_state.warnings = ingest_warnings + parse_warnings + result.warnings
                    st.session_state.exports = {label: export_bundle(result, label) for label in ("json", "csv", "docx", "pdf")}

    warnings = st.session_state.warnings
    if warnings:
        with st.expander("Warnings"):
            for warning in warnings:
                st.write(f"- {warning}")

    result = st.session_state.analysis_result
    if result:
        st.subheader("Detected Topics")
        for idx, topic in enumerate(result.topics, start=1):
            with st.expander(f"{idx}. {topic.topic}", expanded=(idx == 1)):
                st.markdown(f"**Evidence:** {topic.evidence}")
                st.markdown("**Revision Notes**")
                st.text(topic.notes)

                st.markdown("**Multiple Choice**")
                for q_idx, q in enumerate(topic.mcq, start=1):
                    st.write(f"{q_idx}. {q.get('question', '')}")
                    for opt in q.get("options", []):
                        st.write(f"- {opt}")
                    with st.expander(f"Reveal MCQ answer {q_idx}"):
                        st.write(f"Answer: {q.get('answer', '')}")
                        st.write(f"Why: {q.get('explanation', '')}")

                st.markdown("**Short Answer**")
                for q_idx, q in enumerate(topic.short_answer, start=1):
                    st.write(f"{q_idx}. {q.get('question', '')}")
                    with st.expander(f"Reveal guide {q_idx}"):
                        st.write(q.get("answer_guide", ""))

        st.subheader("Ask Documents")
        with st.form("qa_form"):
            question = st.text_input(
                "Ask a question about your uploaded documents/slides",
                value=st.session_state.qa_question,
                placeholder="Example: What assumptions are stated in the simulation slides?",
            )
            ask_clicked = st.form_submit_button("Ask")

        if ask_clicked:
            st.session_state.qa_question = question
            if not question.strip():
                st.session_state.qa_answer = "Please enter a question."
                st.session_state.qa_sources = []
                st.session_state.qa_web_snippets = []
            else:
                context, sources = _build_qa_context(result.raw_documents, question)
                web_entries: list[dict] = []
                if setting("allow_web_browsing", False):
                    web_entries = gather_trusted_web_context(
                        question=question,
                        user_urls=_parse_csv_values(setting("trusted_urls", "")),
                        extra_domains=_parse_csv_values(setting("trusted_domains", "")),
                        max_sources=3,
                    )
                    if web_entries:
                        web_context = "\n".join(f"[WEB:{item['source']}] {item['text']}" for item in web_entries)
                        context = (context + "\n\n" + web_context).strip()[:5000]
                        sources.extend(item["source"] for item in web_entries)
                st.session_state.qa_answer = llm.answer_question(question, context)
                deduped_sources: list[str] = []
                seen = set()
                for src in sources:
                    if src not in seen:
                        seen.add(src)
                        deduped_sources.append(src)
                st.session_state.qa_sources = deduped_sources
                st.session_state.qa_web_snippets = web_entries

        if st.session_state.qa_answer:
            using_web = len(st.session_state.qa_web_snippets) > 0
            if using_web:
                st.info("Answer mode: Local documents + trusted web sources")
            else:
                st.info("Answer mode: Local documents only")

            st.markdown("**Answer**")
            st.markdown(st.session_state.qa_answer)

            if using_web:
                st.markdown("**Trusted Web Evidence Used (bold text)**")
                for item in st.session_state.qa_web_snippets:
                    src = item.get("source", "unknown source")
                    txt = (item.get("text", "") or "").strip()
                    st.markdown(f"- **Source:** {src}")
                    if txt:
                        st.markdown(f"**{txt[:420]}**")

            if st.session_state.qa_sources:
                st.caption("Sources: " + ", ".join(st.session_state.qa_sources))

        st.subheader("Export")
        export_col1, export_col2, export_col3, export_col4 = st.columns(4)
        export_cache = st.session_state.exports or {}

        for label, col in [("json", export_col1), ("csv", export_col2), ("docx", export_col3), ("pdf", export_col4)]:
            if label not in export_cache:
                export_cache[label] = export_bundle(result, label)
            data, mime, filename = export_cache[label]
            col.download_button(
                label=f"Download {label.upper()}",
                data=data,
                file_name=filename,
                mime=mime,
                use_container_width=True,
            )


if __name__ == "__main__":
    main()