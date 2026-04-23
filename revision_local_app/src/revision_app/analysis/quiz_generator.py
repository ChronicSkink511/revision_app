from __future__ import annotations



def generate_quiz(topic: str, context: str, llm_client, total_questions: int = 6) -> dict:
    return llm_client.generate_quiz(topic=topic, context=context, total_questions=total_questions)
