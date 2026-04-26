from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path


DEFAULT_REPO = "https://github.com/ChronicSkink511/revision_app.git"
DEFAULT_BRANCH = "main"
DEFAULT_APP_FILE = "streamlit_app.py"


def run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout.strip())
    if completed.stderr:
        print(completed.stderr.strip(), file=sys.stderr)
    if check and completed.returncode != 0:
        stderr_text = (completed.stderr or "").strip()
        stdout_text = (completed.stdout or "").strip()
        detail = stderr_text or stdout_text
        message = f"Command failed ({completed.returncode}): {' '.join(cmd)}"
        if detail:
            message = f"{message}\n{detail}"
        raise RuntimeError(message)
    return completed


def ensure_git_repo(root: Path) -> None:
    probe = run(["git", "rev-parse", "--is-inside-work-tree"], root, check=False)
    if probe.returncode != 0:
        run(["git", "init"], root)


def ensure_branch(root: Path, branch: str) -> None:
    run(["git", "branch", "-M", branch], root)


def ensure_remote(root: Path, repo_url: str) -> None:
    current = run(["git", "remote", "get-url", "origin"], root, check=False)
    if current.returncode == 0:
        existing = current.stdout.strip()
        if existing != repo_url:
            run(["git", "remote", "set-url", "origin", repo_url], root)
    else:
        run(["git", "remote", "add", "origin", repo_url], root)


def commit_if_needed(root: Path, message: str) -> None:
    status = run(["git", "status", "--porcelain"], root)
    if not status.stdout.strip():
        print("No local changes to commit.")
        return
    run(["git", "add", "-A"], root)
    run(["git", "commit", "-m", message], root)


def push(root: Path, branch: str) -> None:
    run(["git", "push", "-u", "origin", branch], root)


def _is_non_fast_forward(error: RuntimeError) -> bool:
    text = str(error).lower()
    return "non-fast-forward" in text or "failed to push some refs" in text


def sync_branch(root: Path, branch: str) -> None:
    run(["git", "pull", "--rebase", "origin", branch], root)


def open_streamlit_cloud() -> None:
    url = "https://share.streamlit.io/"
    print(f"Opening Streamlit Community Cloud: {url}")
    webbrowser.open(url)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "One-file deploy helper: commits local changes, pushes to GitHub, "
            "and opens Streamlit Community Cloud."
        )
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repo URL")
    parser.add_argument("--branch", default=DEFAULT_BRANCH, help="Git branch")
    parser.add_argument("--app-file", default=DEFAULT_APP_FILE, help="Streamlit app file path")
    parser.add_argument("--commit-message", default="Deploy update", help="Commit message if changes exist")
    parser.add_argument("--no-open", action="store_true", help="Do not open Streamlit website")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    app_file = root / args.app_file
    if not app_file.exists():
        print(f"ERROR: app file not found: {app_file}", file=sys.stderr)
        return 2

    try:
        ensure_git_repo(root)
        ensure_branch(root, args.branch)
        ensure_remote(root, args.repo)
        commit_if_needed(root, args.commit_message)
        try:
            push(root, args.branch)
        except RuntimeError as exc:
            if not _is_non_fast_forward(exc):
                raise
            print("Push rejected (remote ahead). Rebasing local branch and retrying...")
            sync_branch(root, args.branch)
            push(root, args.branch)
    except RuntimeError as exc:
        print(f"\nDeploy helper failed: {exc}", file=sys.stderr)
        print("Tip: if push fails, ensure GitHub auth is valid and resolve any rebase conflicts, then rerun.", file=sys.stderr)
        return 1

    print("\nGitHub push completed.")
    print("Use these values in Streamlit Community Cloud:")
    print(f"- Repo: {args.repo}")
    print(f"- Branch: {args.branch}")
    print(f"- App file: {args.app_file}")

    if not args.no_open:
        open_streamlit_cloud()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
