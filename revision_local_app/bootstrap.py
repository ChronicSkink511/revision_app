from __future__ import annotations

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

REQUIRED_DIRS = [
    "data",
    "data/models",
    "data/uploads",
    "data/work",
    "logs",
    "scripts",
    "src",
    "src/revision_app",
    "src/revision_app/analysis",
    "src/revision_app/export",
    "src/revision_app/image_analysis",
    "src/revision_app/ingestion",
    "src/revision_app/llm",
    "src/revision_app/parsing",
]


def ensure_project_tree() -> None:
    for rel in REQUIRED_DIRS:
        (PROJECT_DIR / rel).mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_project_tree()
    print(f"Project tree ready at: {PROJECT_DIR}")


if __name__ == "__main__":
    main()
