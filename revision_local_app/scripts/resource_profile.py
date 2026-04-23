from __future__ import annotations

import io
import json
import os
import threading
import time
import zipfile
from pathlib import Path
import sys

import psutil

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


class ResourceSampler:
    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.proc = psutil.Process(os.getpid())
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.peak_rss_mb = 0.0
        self.peak_cpu_pct = 0.0

    def _loop(self) -> None:
        self.proc.cpu_percent(None)
        while not self.stop_flag.is_set():
            rss_mb = self.proc.memory_info().rss / (1024 * 1024)
            cpu_pct = self.proc.cpu_percent(None)
            if rss_mb > self.peak_rss_mb:
                self.peak_rss_mb = rss_mb
            if cpu_pct > self.peak_cpu_pct:
                self.peak_cpu_pct = cpu_pct
            time.sleep(self.interval)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_flag.set()
        self.thread.join(timeout=2.0)



def build_payloads() -> list[MockUpload]:
    docs = []
    for idx in range(1, 9):
        txt = (
            f"# Topic {idx}\n"
            "Control systems, thermodynamics, materials, and circuit analysis are common engineering areas.\n"
            "Equations like F=ma, V=IR, and stress=strain*E may appear in notes.\n"
            "This line repeats to make a moderate test corpus. " * 20
        ).encode("utf-8")
        docs.append(MockUpload(f"notes_{idx}.txt", txt))

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx in range(1, 7):
            zf.writestr(
                f"nested/chapter_{idx}.md",
                (
                    f"# Chapter {idx}\n"
                    "Signals and systems, process control, and heat transfer fundamentals.\n"
                    "Useful for low-resource pipeline testing. " * 30
                ),
            )
    docs.append(MockUpload("bundle.zip", zip_buffer.getvalue()))
    return docs



def main() -> None:
    ensure_project_tree()
    config = load_config(ROOT)
    logger = setup_logging(config.logs_dir)

    # Force low-resource profile for this measurement.
    config = type(config)(
        **{
            **config.__dict__,
            "max_topics": min(config.max_topics, 6),
            "quiz_questions_per_topic": min(config.quiz_questions_per_topic, 6),
            "llm_ctx_size": min(config.llm_ctx_size, 1024),
            "llm_max_tokens": min(config.llm_max_tokens, 256),
        }
    )

    sampler = ResourceSampler(interval=0.1)
    start = time.perf_counter()
    sampler.start()

    llm = LocalLLMClient(config=config, logger=logger)
    analyzer = EngineeringImageAnalyzer(llm_client=llm, logger=logger, tesseract_cmd=config.tesseract_cmd)

    uploads = build_payloads()
    persisted, ingest_warnings = ingest_uploaded_files(uploads, config.uploads_dir, logger)
    docs, parse_warnings = parse_documents(persisted, analyzer, logger)
    result = run_analysis(docs, llm_client=llm, config=config, logger=logger)

    exports = {}
    for fmt in ["json", "csv", "docx", "pdf"]:
        blob, _, _ = export_bundle(result, fmt)
        exports[fmt] = len(blob)

    elapsed = time.perf_counter() - start
    sampler.stop()

    print(
        json.dumps(
            {
                "elapsed_s": round(elapsed, 3),
                "peak_rss_mb": round(sampler.peak_rss_mb, 2),
                "peak_cpu_percent_process": round(sampler.peak_cpu_pct, 2),
                "docs_parsed": len(docs),
                "topics": len(result.topics),
                "warnings": len(ingest_warnings) + len(parse_warnings) + len(result.warnings),
                "exports": exports,
                "llm_status": llm.status,
            }
        )
    )


if __name__ == "__main__":
    main()
