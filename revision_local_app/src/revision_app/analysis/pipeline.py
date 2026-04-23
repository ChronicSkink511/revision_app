from __future__ import annotations

import logging

from revision_app.analysis.notes_generator import generate_notes
from revision_app.analysis.quiz_generator import generate_quiz
from revision_app.analysis.topic_detector import detect_topics
from revision_app.schemas import AnalysisResult, TopicBundle



def _build_topic_context(topic: str, docs_text: list[str], max_context_chars: int = 5000) -> str:
    topic_words = {w.lower() for w in topic.split() if len(w) > 2}
    selected: list[str] = []

    # First pass: find sections explicitly mentioning topic words
    for chunk in docs_text:
        low = chunk.lower()
        if any(word in low for word in topic_words):
            selected.append(chunk)
        if len(selected) >= 6:
            break

    # If no topic matches found, use all available content
    if not selected:
        selected = docs_text[:5]

    # Increase context by using more content per chunk
    return "\n\n".join(selected)[:max_context_chars]



def run_analysis(documents, llm_client, config, logger: logging.Logger) -> AnalysisResult:
    warnings: list[str] = []

    topics = detect_topics(
        documents=documents,
        llm_client=llm_client,
        logger=logger,
        enable_embeddings=config.enable_embeddings,
        embedding_model_name=config.embedding_model_name,
        max_topics=max(3, min(12, config.max_topics)),
    )

    # Use more content per document for better context
    docs_text = [doc.text[:6000] for doc in documents if doc.text]
    bundles: list[TopicBundle] = []

    for t in topics:
        topic_name = str(t.get("topic", "General Topic")).strip() or "General Topic"
        evidence = str(t.get("evidence", "Detected from corpus"))
        context = _build_topic_context(topic_name, docs_text, max_context_chars=5000)

        notes = generate_notes(topic=topic_name, context=context, llm_client=llm_client)
        quiz = generate_quiz(
            topic=topic_name,
            context=context,
            llm_client=llm_client,
            total_questions=max(5, min(10, config.quiz_questions_per_topic)),
        )

        mcq = quiz.get("mcq", []) if isinstance(quiz, dict) else []
        short_answer = quiz.get("short_answer", []) if isinstance(quiz, dict) else []

        requested_total = max(5, min(10, config.quiz_questions_per_topic))
        mcq_target = min(6, requested_total - 2)
        short_target = requested_total - mcq_target
        mcq = mcq[:mcq_target]
        short_answer = short_answer[:short_target]

        bundles.append(
            TopicBundle(
                topic=topic_name,
                evidence=evidence,
                notes=notes,
                mcq=mcq,
                short_answer=short_answer,
            )
        )

    if not bundles:
        warnings.append("No topics identified from available content.")

    return AnalysisResult(topics=bundles, raw_documents=documents, warnings=warnings)
