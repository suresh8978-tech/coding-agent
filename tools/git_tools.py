"""
Git Tools
Handles git operations: status, diff, add, commit, push.
Push requires admin access and user confirmation.
"""

import os
import subprocess
import logging
from typing import Dict

logger = logging.getLogger("agent.git")


def _run_git(repo_path: str, args: list) -> Dict:
    """Run a git command and return the result."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=60
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "exit_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Git command timed out."}
    except FileNotFoundError:
        return {"success": False, "error": "Git not found on server."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def git_status(repo_path: str) -> Dict:
    """Show the current git status — modified, untracked, staged files."""
    result = _run_git(repo_path, ["status", "--porcelain"])
    if not result.get("success"):
        return result

    output = result.get("stdout", "")
    if not output:
        return {"result": "Working tree is clean. No changes to commit."}

    # Parse status
    modified = []
    untracked = []
    staged = []

    for line in output.splitlines():
        if len(line) < 3:
            continue
        status = line[:2]
        filename = line[3:]

        if status[0] in ("M", "A", "D", "R"):
            staged.append(filename)
        if status[1] == "M":
            modified.append(filename)
        if status == "??":
            untracked.append(filename)

    logger.info(f"Git status: {len(modified)} modified, {len(untracked)} untracked, {len(staged)} staged")
    return {
        "modified": modified,
        "untracked": untracked,
        "staged": staged,
        "raw": output
    }


def git_diff(repo_path: str, file_path: str = "") -> Dict:
    """Show git diff for a specific file or all changes."""
    args = ["diff"]
    if file_path:
        args.append(file_path)

    result = _run_git(repo_path, args)
    if not result.get("success"):
        return result

    diff = result.get("stdout", "")
    if not diff:
        return {"result": "No unstaged changes." + (" Try git_diff_staged." if not file_path else "")}

    logger.info(f"Git diff: {'all files' if not file_path else file_path}")
    return {"diff": diff}


def git_add(repo_path: str, file_path: str = ".") -> Dict:
    """Stage files for commit. Use '.' to stage all changes."""
    result = _run_git(repo_path, ["add", file_path])
    if result.get("success"):
        logger.info(f"Git add: {file_path}")
        return {"result": f"Staged: {file_path}"}
    return result


def git_commit(repo_path: str, message: str) -> Dict:
    """Commit staged changes with a message."""
    if not message:
        return {"error": "Commit message is required."}

    # Set git user config if not set
    git_name = os.getenv("GIT_USER_NAME", "Terraform Agent")
    git_email = os.getenv("GIT_USER_EMAIL", "agent@example.com")
    _run_git(repo_path, ["config", "user.name", git_name])
    _run_git(repo_path, ["config", "user.email", git_email])

    result = _run_git(repo_path, ["commit", "-m", message])
    if result.get("success"):
        logger.info(f"Git commit: {message}")
        return {"result": f"Committed: {message}", "output": result.get("stdout", "")}
    return result


def git_push(repo_path: str, branch: str = "") -> Dict:
    """
    Push commits to remote repository.
    This is a privileged operation — requires admin access.
    Returns a confirmation request if not yet confirmed.
    """
    args = ["push"]
    if branch:
        args.extend(["origin", branch])

    result = _run_git(repo_path, args)
    if result.get("success"):
        logger.info(f"Git push: {'origin ' + branch if branch else 'default'}")
        return {
            "result": "Successfully pushed to remote.",
            "output": result.get("stdout", "") + result.get("stderr", "")
        }
    return result


def git_log(repo_path: str, count: int = 10) -> Dict:
    """Show recent git commits."""
    result = _run_git(repo_path, [
        "log", f"-{count}",
        "--pretty=format:%h | %an | %ar | %s"
    ])
    if result.get("success"):
        commits = result.get("stdout", "").splitlines()
        logger.info(f"Git log: {len(commits)} commits")
        return {"commits": commits, "count": len(commits)}
    return result


def git_branch(repo_path: str) -> Dict:
    """Show current and available branches."""
    result = _run_git(repo_path, ["branch", "-a"])
    if result.get("success"):
        branches = result.get("stdout", "").splitlines()
        current = ""
        for b in branches:
            if b.startswith("*"):
                current = b[2:].strip()
        return {"current_branch": current, "branches": [b.strip().lstrip("* ") for b in branches]}
    return result
