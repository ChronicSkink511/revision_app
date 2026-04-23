from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "revision_local_app" / "streamlit_app.py"


def _is_running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False

if __name__ == "__main__":
    if not _is_running_under_streamlit():
        cmd = [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())]
        raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))

    app_root = TARGET.parent
    os.chdir(app_root)
    sys.path.insert(0, str(app_root))
    sys.path.insert(0, str(app_root / "src"))
    runpy.run_path(str(TARGET), run_name="__main__")
