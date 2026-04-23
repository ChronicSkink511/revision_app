from __future__ import annotations

import subprocess
import sys
from pathlib import Path



def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, "-m", "streamlit", "run", str(root / "streamlit_app.py")]
    raise SystemExit(subprocess.call(cmd, cwd=str(root)))


if __name__ == "__main__":
    main()
