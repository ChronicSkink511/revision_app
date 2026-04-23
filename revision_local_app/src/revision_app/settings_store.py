from __future__ import annotations

import json
from pathlib import Path

from revision_app.config import AppConfig


SETTINGS_FILENAME = "app_settings.json"



def default_settings(config: AppConfig) -> dict:
    return {
        "default_mode": "Low",
        "gguf_model": config.gguf_model_path.name,
        "llm_ctx_size": config.llm_ctx_size,
        "llm_max_tokens": config.llm_max_tokens,
        "llm_threads": config.llm_threads,
        "llm_temperature": config.llm_temperature,
        "enable_embeddings": config.enable_embeddings,
        "embedding_model_name": config.embedding_model_name,
        "tesseract_cmd": config.tesseract_cmd or "",
        "max_topics_low": 6,
        "quiz_per_topic_low": 6,
        "max_topics_high": 10,
        "quiz_per_topic_high": 8,
    }



def settings_path(config: AppConfig) -> Path:
    return config.work_dir / SETTINGS_FILENAME



def _clamp_int(value, lower: int, upper: int, fallback: int) -> int:
    try:
        return max(lower, min(upper, int(value)))
    except Exception:
        return fallback



def _clamp_float(value, lower: float, upper: float, fallback: float) -> float:
    try:
        return max(lower, min(upper, float(value)))
    except Exception:
        return fallback



def normalize_settings(raw: dict, defaults: dict) -> dict:
    mode = str(raw.get("default_mode", defaults["default_mode"]))
    if mode not in {"Low", "High"}:
        mode = defaults["default_mode"]

    normalized = {
        "default_mode": mode,
        "gguf_model": str(raw.get("gguf_model", defaults["gguf_model"]))[:220],
        "llm_ctx_size": _clamp_int(raw.get("llm_ctx_size"), 512, 4096, defaults["llm_ctx_size"]),
        "llm_max_tokens": _clamp_int(raw.get("llm_max_tokens"), 64, 1024, defaults["llm_max_tokens"]),
        "llm_threads": _clamp_int(raw.get("llm_threads"), 1, 32, defaults["llm_threads"]),
        "llm_temperature": _clamp_float(raw.get("llm_temperature"), 0.0, 1.2, defaults["llm_temperature"]),
        "enable_embeddings": bool(raw.get("enable_embeddings", defaults["enable_embeddings"])),
        "embedding_model_name": str(raw.get("embedding_model_name", defaults["embedding_model_name"]))[:220],
        "tesseract_cmd": str(raw.get("tesseract_cmd", defaults["tesseract_cmd"]))[:260],
        "max_topics_low": _clamp_int(raw.get("max_topics_low"), 3, 12, defaults["max_topics_low"]),
        "quiz_per_topic_low": _clamp_int(raw.get("quiz_per_topic_low"), 5, 10, defaults["quiz_per_topic_low"]),
        "max_topics_high": _clamp_int(raw.get("max_topics_high"), 3, 12, defaults["max_topics_high"]),
        "quiz_per_topic_high": _clamp_int(raw.get("quiz_per_topic_high"), 5, 10, defaults["quiz_per_topic_high"]),
    }

    return normalized



def load_settings(config: AppConfig) -> dict:
    defaults = default_settings(config)
    path = settings_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        return defaults

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return defaults
        return normalize_settings(payload, defaults)
    except Exception:
        return defaults



def save_settings(config: AppConfig, settings: dict) -> dict:
    defaults = default_settings(config)
    normalized = normalize_settings(settings, defaults)
    path = settings_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return normalized
