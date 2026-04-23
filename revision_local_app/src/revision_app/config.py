from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    models_dir: Path
    uploads_dir: Path
    work_dir: Path
    logs_dir: Path
    gguf_model_path: Path
    llm_ctx_size: int
    llm_max_tokens: int
    llm_threads: int
    llm_temperature: float
    enable_embeddings: bool
    embedding_model_name: str
    tesseract_cmd: str | None
    max_topics: int
    quiz_questions_per_topic: int



def _as_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}



def load_config(project_root: Path) -> AppConfig:
    models_dir = project_root / "data" / "models"
    uploads_dir = project_root / "data" / "uploads"
    work_dir = project_root / "data" / "work"
    logs_dir = project_root / "logs"

    model_filename = os.getenv("REVAPP_GGUF_MODEL", "Qwen2.5-3B-Instruct-Q4_K_M.gguf")

    return AppConfig(
        project_root=project_root,
        models_dir=models_dir,
        uploads_dir=uploads_dir,
        work_dir=work_dir,
        logs_dir=logs_dir,
        gguf_model_path=models_dir / model_filename,
        llm_ctx_size=int(os.getenv("REVAPP_CTX", "1024")),
        llm_max_tokens=int(os.getenv("REVAPP_MAX_TOKENS", "256")),
        llm_threads=int(os.getenv("REVAPP_THREADS", "4")),
        llm_temperature=float(os.getenv("REVAPP_TEMP", "0.2")),
        enable_embeddings=_as_bool(os.getenv("REVAPP_ENABLE_EMBEDDINGS", "false"), default=False),
        embedding_model_name=os.getenv("REVAPP_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        tesseract_cmd=os.getenv("TESSERACT_CMD"),
        max_topics=int(os.getenv("REVAPP_MAX_TOPICS", "6")),
        quiz_questions_per_topic=int(os.getenv("REVAPP_QUIZ_PER_TOPIC", "6")),
    )
