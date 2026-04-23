from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from bootstrap import ensure_project_tree
from revision_app.analysis import run_analysis
from revision_app.config import load_config
from revision_app.export import export_bundle
from revision_app.image_analysis import EngineeringImageAnalyzer
from revision_app.ingestion import ingest_uploaded_files
from revision_app.llm import LocalLLMClient
from revision_app.logging_utils import setup_logging
from revision_app.parsing import parse_documents


class MockUpload:
    def __init__(self, name: str, payload: bytes) -> None:
        self.name = name
        self._payload = payload

    def getbuffer(self):
        return memoryview(self._payload)



def build_test_payloads() -> list[MockUpload]:
    txt = MockUpload(
        "systems.txt",
        (
            "# Signals and Systems\n"
            "Linear time-invariant systems are represented by transfer functions.\n"
            "Impulse response h(t) and convolution describe behavior."
        ).encode("utf-8"),
    )

    md = MockUpload(
        "materials.md",
        (
            "# Material Science\n"
            "Stress-strain curves define elastic and plastic regimes.\n"
            "Young's modulus E = stress/strain."
        ).encode("utf-8"),
    )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "nested/thermo_notes.txt",
            "# Thermodynamics\nRankine cycle includes boiler, turbine, condenser, and pump.",
        )
        zf.writestr(
            "nested/controls.md",
            "# Control Systems\nPID control combines proportional, integral, derivative actions.",
        )

    zipped = MockUpload("bundle.zip", zip_buffer.getvalue())
    return [txt, md, zipped]



def main() -> None:
    ensure_project_tree()
    config = load_config(ROOT)
    logger = setup_logging(config.logs_dir)

    llm = LocalLLMClient(config=config, logger=logger)
    image_analyzer = EngineeringImageAnalyzer(llm_client=llm, logger=logger, tesseract_cmd=config.tesseract_cmd)

    uploads = build_test_payloads()
    persisted, ingest_warnings = ingest_uploaded_files(uploads, config.uploads_dir, logger)
    docs, parse_warnings = parse_documents(persisted, image_analyzer, logger)

    if not docs:
        raise RuntimeError("No documents parsed in integration test")

    result = run_analysis(docs, llm_client=llm, config=config, logger=logger)

    export_sizes = {}
    for fmt in ["json", "csv", "docx", "pdf"]:
        data, _, _ = export_bundle(result, fmt)
        if not data:
            raise RuntimeError(f"Empty export output for {fmt}")
        export_sizes[fmt] = len(data)

    summary = {
        "parsed_docs": len(docs),
        "topics": [t.topic for t in result.topics],
        "ingest_warnings": len(ingest_warnings),
        "parse_warnings": len(parse_warnings),
        "analysis_warnings": len(result.warnings),
        "export_sizes": export_sizes,
        "llm_status": llm.status,
    }

    print(json.dumps(summary))


if __name__ == "__main__":
    main()
