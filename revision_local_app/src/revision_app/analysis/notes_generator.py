from __future__ import annotations



def generate_notes(topic: str, context: str, llm_client) -> str:
    return llm_client.generate_notes(topic=topic, context=context)
