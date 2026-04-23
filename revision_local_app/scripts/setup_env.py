from __future__ import annotations

import subprocess
import sys
from pathlib import Path



def run(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)



def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run([sys.executable, "bootstrap.py"], cwd=root)
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=root)
    print("Setup complete. Run: python scripts/run_app.py")


if __name__ == "__main__":
    main()
