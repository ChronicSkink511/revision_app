from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from bootstrap import ensure_project_tree
from revision_app.analysis import run_analysis
from revision_app.config import load_config
from revision_app.logging_utils import setup_logging
from revision_app.llm import LocalLLMClient
from revision_app.schemas import DocumentContent



def main() -> None:
    ensure_project_tree()
    config = load_config(ROOT)
    logger = setup_logging(config.logs_dir)
    llm = LocalLLMClient(config, logger)

    docs = [
        DocumentContent(
            source_path=ROOT / "data" / "work" / "sample.txt",
            file_type="txt",
            text=(
                "# Thermodynamics\n"
                "Entropy and enthalpy relations are used in cycle analysis. "
                "The Rankine cycle contains pump, boiler, turbine, condenser.\n"
                "# Control Systems\n"
                "PID control uses proportional, integral, derivative terms."
            ),
        )
    ]

    result = run_analysis(docs, llm_client=llm, config=config, logger=logger)

    payload = {
        "topics": [t.topic for t in result.topics],
        "warning_count": len(result.warnings),
        "topic_count": len(result.topics),
    }

    print(json.dumps(payload))


if __name__ == "__main__":
    main()
