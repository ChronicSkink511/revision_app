from __future__ import annotations

import io
import logging
import re
import uuid
import zipfile
from pathlib import Path

SAFE_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".txt",
    ".md",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".zip",
}

MAX_ZIP_FILES = 1500
MAX_ZIP_MEMBER_SIZE = 50 * 1024 * 1024



def _clean_name(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name[:180] if name else "uploaded_file"



def _safe_write(target: Path, data: bytes) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target



def _safe_extract_zip(zip_path: Path, output_root: Path, logger: logging.Logger) -> tuple[list[Path], list[str]]:
    extracted: list[Path] = []
    warnings: list[str] = []

    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = zf.infolist()
            if len(members) > MAX_ZIP_FILES:
                warnings.append(f"Skipping {zip_path.name}: too many files in archive.")
                return extracted, warnings

            for info in members:
                if info.is_dir():
                    continue
                if info.file_size > MAX_ZIP_MEMBER_SIZE:
                    warnings.append(f"Skipping large member {info.filename} in {zip_path.name}.")
                    continue

                member_path = Path(info.filename)
                safe_name = _clean_name(member_path.name)
                suffix = Path(safe_name).suffix.lower()
                if suffix not in SAFE_EXTENSIONS or suffix == ".zip":
                    continue

                target = output_root / f"{uuid.uuid4().hex[:10]}_{safe_name}"
                resolved_target = target.resolve()
                if output_root.resolve() not in resolved_target.parents and resolved_target != output_root.resolve():
                    warnings.append(f"Blocked suspicious archive path {info.filename}")
                    continue

                try:
                    data = zf.read(info)
                    _safe_write(target, data)
                    extracted.append(target)
                except Exception as exc:
                    logger.warning("Failed extracting %s from %s: %s", info.filename, zip_path, exc)
                    warnings.append(f"Could not extract member {info.filename} from {zip_path.name}.")
    except zipfile.BadZipFile:
        warnings.append(f"Corrupted zip skipped: {zip_path.name}")
    except Exception as exc:
        logger.exception("Zip extraction error for %s", zip_path)
        warnings.append(f"Error reading zip {zip_path.name}: {exc}")

    return extracted, warnings



def ingest_uploaded_files(uploaded_files: list, upload_dir: Path, logger: logging.Logger) -> tuple[list[Path], list[str]]:
    persisted_paths: list[Path] = []
    warnings: list[str] = []

    upload_dir.mkdir(parents=True, exist_ok=True)

    for item in uploaded_files:
        safe_name = _clean_name(getattr(item, "name", "upload.bin"))
        suffix = Path(safe_name).suffix.lower()

        if suffix not in SAFE_EXTENSIONS:
            warnings.append(f"Unsupported file skipped: {safe_name}")
            continue

        try:
            payload = item.getbuffer() if hasattr(item, "getbuffer") else io.BytesIO(item.read()).getbuffer()
            target = upload_dir / f"{uuid.uuid4().hex[:10]}_{safe_name}"
            _safe_write(target, bytes(payload))
        except Exception as exc:
            logger.exception("Upload write failure for %s", safe_name)
            warnings.append(f"Could not save upload {safe_name}: {exc}")
            continue

        if suffix == ".zip":
            extracted, zip_warnings = _safe_extract_zip(target, upload_dir, logger)
            persisted_paths.extend(extracted)
            warnings.extend(zip_warnings)
        else:
            persisted_paths.append(target)

    return persisted_paths, warnings
