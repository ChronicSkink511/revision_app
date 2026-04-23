from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    app_file = root / "streamlit_app.py"

    if not app_file.exists():
        raise SystemExit(f"Could not find app entrypoint: {app_file}")

    cmd = [sys.executable, "-m", "streamlit", "run", str(app_file)]
    raise SystemExit(subprocess.call(cmd, cwd=str(root)))


if __name__ == "__main__":
    main()
