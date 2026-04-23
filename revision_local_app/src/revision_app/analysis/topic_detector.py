from __future__ import annotations

import logging
import re
from collections import Counter

import numpy as np

from revision_app.schemas import DocumentContent



def _heading_candidates(text: str) -> list[str]:
    topics: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            topics.append(line.lstrip("# "))
        elif len(line) < 90 and re.match(r"^[A-Za-z0-9\-\s()/:]+$", line):
            words = line.split()
            titled = sum(1 for w in words if w[:1].isupper())
            if words and titled / len(words) > 0.5:
                topics.append(line)
    return topics



def _merge_with_embeddings(candidates: list[str], model_name: str, logger: logging.Logger) -> list[str]:
    if len(candidates) < 2:
        return candidates

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        embeddings = model.encode(candidates, convert_to_numpy=True, normalize_embeddings=True)

        kept: list[str] = []
        kept_vectors: list[np.ndarray] = []
        for idx, topic in enumerate(candidates):
            if not kept:
                kept.append(topic)
                kept_vectors.append(embeddings[idx])
                continue

            vec = embeddings[idx]
            sims = [float(np.dot(vec, kept_vec)) for kept_vec in kept_vectors]
            if (not sims) or max(sims) < 0.86:
                kept.append(topic)
                kept_vectors.append(vec)
        return kept
    except Exception as exc:
        logger.warning("Embedding merge unavailable: %s", exc)
        return candidates



def detect_topics(
    documents: list[DocumentContent],
    llm_client,
    logger: logging.Logger,
    enable_embeddings: bool,
    embedding_model_name: str,
    max_topics: int = 8,
) -> list[dict]:
    heading_pool: list[str] = []
    snippets: list[str] = []

    for doc in documents:
        heading_pool.extend(_heading_candidates(doc.text))
        snippets.append(doc.text[:600])

    if not heading_pool:
        words = re.findall(r"[A-Za-z][A-Za-z0-9+-]{3,}", "\n".join(snippets))
        common = [word for word, _ in Counter(w.lower() for w in words).most_common(20)]
        heading_pool = [w.title() for w in common]

    deduped: list[str] = []
    seen = set()
    for t in heading_pool:
        normalized = t.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(t.strip())

    candidates = deduped[:40]

    if enable_embeddings:
        candidates = _merge_with_embeddings(candidates, embedding_model_name, logger)

    llm_topics = llm_client.generate_topics(snippets, candidates, max_topics=max_topics)
    if llm_topics:
        return llm_topics[:max_topics]

    return [{"topic": t, "evidence": "Detected from document structure"} for t in candidates[:max_topics]]
