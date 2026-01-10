"""Git operation tools for the coding agent."""

import os
import subprocess
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool


def _run_git_command(args: list[str], cwd: Optional[str] = None) -> tuple[bool, str]:
    """Run a git command and return success status and output.
    
    Args:
        args: List of git command arguments (without 'git' prefix).
        cwd: Working directory for the command.
        
    Returns:
        Tuple of (success, output_or_error).
    """
    repo_path = cwd or os.environ.get("REPO_PATH", ".")
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, result.stderr.strip() or result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "Git command timed out."
    except FileNotFoundError:
        return False, "Git is not installed or not in PATH."
    except Exception as e:
        return False, f"Error running git command: {str(e)}"


@tool
def git_fetch_all() -> str:
    """Fetch all branches from all remotes.
    
    Returns:
        Success message or error description.
    """
    success, output = _run_git_command(["fetch", "--all"])
    if success:
        return f"Successfully fetched all branches.\n{output}" if output else "Successfully fetched all branches."
    return f"Error fetching: {output}"


@tool
def git_create_branch(name: str) -> str:
    """Create a new branch with 'agent-' prefix.
    
    Args:
        name: The branch name (will be prefixed with 'agent-' automatically).
        
    Returns:
        Success message or error description.
    """
    # Ensure the branch name has the agent- prefix
    branch_name = name if name.startswith("agent-") else f"agent-{name}"
    
    # Sanitize branch name: replace spaces and special chars
    branch_name = branch_name.replace(" ", "-").lower()
    branch_name = "".join(c for c in branch_name if c.isalnum() or c in "-_/")
    
    success, output = _run_git_command(["checkout", "-b", branch_name])
    if success:
        return f"Successfully created and switched to branch '{branch_name}'."
    return f"Error creating branch: {output}"


@tool
def git_checkout(branch: str) -> str:
    """Switch to an existing branch.
    
    Args:
        branch: The name of the branch to switch to.
        
    Returns:
        Success message or error description.
    """
    success, output = _run_git_command(["checkout", branch])
    if success:
        return f"Successfully switched to branch '{branch}'."
    return f"Error switching branch: {output}"


@tool
def git_add(files: str) -> str:
    """Stage files for commit.
    
    Args:
        files: Comma-separated list of file paths to stage, or '.' for all files.
        
    Returns:
        Success message or error description.
    """
    file_list = [f.strip() for f in files.split(",")]
    success, output = _run_git_command(["add"] + file_list)
    if success:
        return f"Successfully staged files: {', '.join(file_list)}"
    return f"Error staging files: {output}"


@tool
def git_commit(message: str) -> str:
    """Commit staged changes.
    
    Args:
        message: The commit message.
        
    Returns:
        Success message or error description.
    """
    success, output = _run_git_command(["commit", "-m", message])
    if success:
        return f"Successfully committed changes.\n{output}"
    return f"Error committing: {output}"


@tool
def git_push(remote: str = "origin", branch: str = "") -> str:
    """Push commits to remote repository.
    
    Args:
        remote: The remote name (default: 'origin').
        branch: The branch to push. If empty, pushes current branch.
        
    Returns:
        Success message or error description.
    """
    if branch:
        success, output = _run_git_command(["push", remote, branch])
    else:
        success, output = _run_git_command(["push", "-u", remote, "HEAD"])
    
    if success:
        return f"Successfully pushed to {remote}.\n{output}" if output else f"Successfully pushed to {remote}."
    return f"Error pushing: {output}"


@tool
def git_diff(staged: bool = False) -> str:
    """Get the diff of changes.
    
    Args:
        staged: If True, show diff of staged changes. Otherwise show unstaged.
        
    Returns:
        The diff output or message if no changes.
    """
    args = ["diff", "--staged"] if staged else ["diff"]
    success, output = _run_git_command(args)
    if success:
        return output if output else "No changes to show."
    return f"Error getting diff: {output}"


@tool
def git_status() -> str:
    """Get the current repository status.
    
    Returns:
        The git status output.
    """
    success, output = _run_git_command(["status", "--short"])
    if success:
        return output if output else "Working tree clean, no changes."
    return f"Error getting status: {output}"


def get_current_branch() -> str:
    """Get the name of the current branch.
    
    Returns:
        The current branch name or error message.
    """
    success, output = _run_git_command(["branch", "--show-current"])
    if success:
        return output
    return f"Error getting current branch: {output}"
