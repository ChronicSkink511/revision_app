from __future__ import annotations

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


def _init_state(default_mode: str):
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "warnings" not in st.session_state:
        st.session_state.warnings = []
    if "exports" not in st.session_state:
        st.session_state.exports = {}
    if "runtime_mode" not in st.session_state:
        st.session_state.runtime_mode = default_mode if default_mode in {"Low", "High"} else "Low"


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
    logger = setup_logging(config.logs_dir)
    persisted_settings = load_settings(config)

    st.set_page_config(page_title="Local Revision Assistant", layout="wide")
    st.title("Local Engineering Revision Assistant")
    st.caption("Offline, low-resource study generation using local GGUF models.")

    _init_state(default_mode=persisted_settings.get("default_mode", "Low"))

    effective_config = replace(
        config,
        gguf_model_path=config.models_dir / persisted_settings["gguf_model"],
        llm_ctx_size=persisted_settings["llm_ctx_size"],
        llm_max_tokens=persisted_settings["llm_max_tokens"],
        llm_threads=persisted_settings["llm_threads"],
        llm_temperature=persisted_settings["llm_temperature"],
        enable_embeddings=persisted_settings["enable_embeddings"],
        embedding_model_name=persisted_settings["embedding_model_name"],
        tesseract_cmd=(persisted_settings["tesseract_cmd"] or None),
    )

    llm = _get_llm_client(config=effective_config, _logger=logger)
    image_analyzer = _get_image_analyzer(_llm_client=llm, _logger=logger, tesseract_cmd=effective_config.tesseract_cmd)

    with st.sidebar:
        st.subheader("Runtime")

        with st.expander("Settings", expanded=True):
            with st.form("settings_form"):
                default_mode = st.selectbox(
                    "Default resource mode",
                    options=["Low", "High"],
                    index=0 if persisted_settings["default_mode"] == "Low" else 1,
                    help="Applied as default each time the app starts.",
                )
                gguf_model = st.text_input("GGUF model filename", value=persisted_settings["gguf_model"])
                llm_ctx_size = st.number_input("LLM context size", min_value=512, max_value=4096, value=int(persisted_settings["llm_ctx_size"]), step=128)
                llm_max_tokens = st.number_input("LLM max tokens", min_value=64, max_value=1024, value=int(persisted_settings["llm_max_tokens"]), step=32)
                llm_threads = st.number_input("LLM CPU threads", min_value=1, max_value=32, value=int(persisted_settings["llm_threads"]), step=1)
                llm_temperature = st.slider("LLM temperature", min_value=0.0, max_value=1.2, value=float(persisted_settings["llm_temperature"]), step=0.05)
                enable_embeddings = st.checkbox("Enable embedding topic merge", value=bool(persisted_settings["enable_embeddings"]))
                embedding_model_name = st.text_input("Embedding model", value=persisted_settings["embedding_model_name"])
                tesseract_cmd = st.text_input("Tesseract command path (optional)", value=persisted_settings["tesseract_cmd"])

                low_col, high_col = st.columns(2)
                with low_col:
                    max_topics_low = st.number_input("Low: max topics", min_value=3, max_value=12, value=int(persisted_settings["max_topics_low"]), step=1)
                    quiz_per_topic_low = st.number_input("Low: quiz questions/topic", min_value=5, max_value=10, value=int(persisted_settings["quiz_per_topic_low"]), step=1)
                with high_col:
                    max_topics_high = st.number_input("High: max topics", min_value=3, max_value=12, value=int(persisted_settings["max_topics_high"]), step=1)
                    quiz_per_topic_high = st.number_input("High: quiz questions/topic", min_value=5, max_value=10, value=int(persisted_settings["quiz_per_topic_high"]), step=1)

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
                        "max_topics_low": max_topics_low,
                        "quiz_per_topic_low": quiz_per_topic_low,
                        "max_topics_high": max_topics_high,
                        "quiz_per_topic_high": quiz_per_topic_high,
                    },
                )
                st.session_state.runtime_mode = saved["default_mode"]
                _get_llm_client.clear()
                _get_image_analyzer.clear()
                st.success("Settings saved.")
                st.rerun()

            if reset_clicked:
                saved = save_settings(config, {})
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
            max_topics=min(effective_config.max_topics, persisted_settings["max_topics_low"]),
            quiz_questions_per_topic=min(effective_config.quiz_questions_per_topic, persisted_settings["quiz_per_topic_low"]),
        )
    else:
        runtime_config = replace(
            effective_config,
            max_topics=max(effective_config.max_topics, persisted_settings["max_topics_high"]),
            quiz_questions_per_topic=max(effective_config.quiz_questions_per_topic, persisted_settings["quiz_per_topic_high"]),
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
                persisted, ingest_warnings = ingest_uploaded_files(uploads, config.uploads_dir, logger)

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
                    export_cache = {}
                    for label in ("json", "csv", "docx", "pdf"):
                        export_cache[label] = export_bundle(result, label)
                    st.session_state.exports = export_cache

    warnings = st.session_state.warnings
    if warnings:
        with st.expander("Warnings"):
            for w in warnings:
                st.write(f"- {w}")

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
