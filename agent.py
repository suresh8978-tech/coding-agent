"""
Terraform Coding Agent v1.0
─────────────────────────────────────
An AI-powered coding agent that manages Terraform infrastructure files.

Capabilities:
  • Analyze repository structure
  • List, view, create, modify, delete .tf files
  • Run safe Terraform CLI commands
  • Git operations (status, diff, commit, push)
  • User access control via user_config.yaml
  • Integrates with Open WebUI as an OpenAPI Tool Server

Architecture:
  agent.py          → FastAPI server (this file)
  tools/
    file_manager.py → File operations (read, write, delete)
    terraform_tools → Terraform CLI commands
    git_tools.py    → Git operations (status, commit, push)
    access_control  → User permission management
  user_config.yaml  → Access control configuration
  .env              → Environment variables (repo path, API keys)
"""

import os
import logging
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from tools import file_manager, terraform_tools, git_tools, access_control

# ── Load Environment ─────────────────────────────────
load_dotenv()

REPO_PATH = os.getenv("TERRAFORM_REPO_PATH", "")
LOG_FILE = os.getenv("LOG_FILE", "agent.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── Logging Setup ────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("agent")


# ── Helper: Get User Email from Open WebUI Headers ───
def get_email(request: Request) -> str:
    """Extract user email from Open WebUI request headers."""
    email = request.headers.get("X-OpenWebUI-User-Email", "")
    if not email:
        email = request.headers.get("x-openwebui-user-email", "")
    return email.strip().lower()


def get_repo_path() -> str:
    """Get the configured repository path."""
    if not REPO_PATH:
        raise ValueError("TERRAFORM_REPO_PATH not set in .env")
    if not Path(REPO_PATH).exists():
        raise ValueError(f"Repo path not found: {REPO_PATH}")
    return REPO_PATH


# ── FastAPI App ──────────────────────────────────────
app = FastAPI(
    title="Terraform Coding Agent",
    description=(
        "AI-powered Terraform infrastructure management agent. "
        "Analyze, create, modify, and manage .tf files. "
        "Run Terraform CLI commands and manage git operations. "
        "User access is controlled via user_config.yaml."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Models ───────────────────────────────────

class FileRequest(BaseModel):
    file_path: str = Field(..., description="Relative path to the file, e.g. main.tf")
    content: str = Field(..., description="The complete file content")
    reason: str = Field("", description="Brief explanation of the change")

class TerraformRequest(BaseModel):
    command: str = Field(..., description="Terraform subcommand: init, plan, validate, fmt, etc.")

class GitCommitRequest(BaseModel):
    message: str = Field(..., description="Commit message describing the changes")

class GitPushRequest(BaseModel):
    branch: str = Field("", description="Branch to push. Leave empty for current branch.")
    confirm: bool = Field(False, description="Set to true to confirm push. First call without confirm to preview.")


# ── FILE OPERATION ENDPOINTS ─────────────────────────

@app.get("/analyze_repo", operation_id="analyze_terraform_repo",
         summary="Analyze the Terraform repository structure")
def analyze_repo(request: Request):
    """Analyze the repository: list all .tf files with their providers, resources, variables, and modules. Call this first to understand the repo."""
    email = get_email(request)
    allowed, reason = access_control.check_permission(email, "read")
    if not allowed:
        return {"error": reason}

    try:
        repo = get_repo_path()
        result = file_manager.analyze_repo(repo)
        result["user"] = email
        return result
    except Exception as e:
        return {"error": str(e)}


@app.get("/list_files", operation_id="list_terraform_files",
         summary="List all files in the repository")
def list_files(request: Request):
    """List all Terraform (.tf) and related files in the repository."""
    email = get_email(request)
    allowed, reason = access_control.check_permission(email, "read")
    if not allowed:
        return {"error": reason}

    try:
        repo = get_repo_path()
        result = file_manager.list_files(repo)
        result["user"] = email
        return result
    except Exception as e:
        return {"error": str(e)}


@app.get("/show_file", operation_id="show_terraform_file",
         summary="Display the contents of a file")
def show_file(request: Request,
              file_path: str = Query(..., description="Relative path, e.g. main.tf")):
    """Read and display the contents of a specific file from the repository."""
    email = get_email(request)
    allowed, reason = access_control.check_permission(email, "read")
    if not allowed:
        return {"error": reason}

    try:
        repo = get_repo_path()
        result = file_manager.show_file(repo, file_path)
        result["user"] = email
        return result
    except Exception as e:
        return {"error": str(e)}


@app.post("/create_file", operation_id="create_terraform_file",
          summary="Create a new Terraform file")
def create_file(req: FileRequest, request: Request):
    """Create a new .tf file in the repository. Will not overwrite existing files. Read-write access required."""
    email = get_email(request)
    allowed, reason = access_control.check_permission(email, "write")
    if not allowed:
        return {"error": reason}

    try:
        repo = get_repo_path()
        result = file_manager.create_file(repo, req.file_path, req.content, req.reason)
        result["user"] = email
        logger.info(f"User {email} created: {req.file_path}")
        return result
    except Exception as e:
        return {"error": str(e)}


@app.post("/modify_file", operation_id="modify_terraform_file",
          summary="Modify an existing file with backup")
def modify_file(req: FileRequest, request: Request):
    """Modify an existing file. Creates a .bak backup before changes. Shows a diff of what changed. Read-write access required."""
    email = get_email(request)
    allowed, reason = access_control.check_permission(email, "write")
    if not allowed:
        return {"error": reason}

    try:
        repo = get_repo_path()
        result = file_manager.modify_file(repo, req.file_path, req.content, req.reason)
        result["user"] = email
        logger.info(f"User {email} modified: {req.file_path}")
        return result
    except Exception as e:
        return {"error": str(e)}


@app.delete("/delete_file", operation_id="delete_terraform_file",
            summary="Delete a file with backup")
def delete_file(request: Request,
                file_path: str = Query(..., description="File to delete")):
    """Delete a file from the repository. Creates a backup before deletion. Read-write access required."""
    email = get_email(request)
    allowed, reason = access_control.check_permission(email, "write")
    if not allowed:
        return {"error": reason}

    try:
        repo = get_repo_path()
        result = file_manager.delete_file(repo, file_path)
        result["user"] = email
        logger.info(f"User {email} deleted: {file_path}")
        return result
    except Exception as e:
        return {"error": str(e)}


# ── TERRAFORM ENDPOINTS ──────────────────────────────

@app.post("/run_terraform", operation_id="run_terraform_command",
          summary="Run a safe Terraform CLI command")
def run_terraform(req: TerraformRequest, request: Request):
    """Run a Terraform command. Allowed: init, plan, validate, fmt, state list, output, providers, version. Blocked: apply, destroy (must be run manually for safety). Read-only users can only run plan, validate, output."""
    email = get_email(request)
    allowed, reason = access_control.check_permission(email, "terraform")
    if not allowed:
        return {"error": reason}

    # Check if user can run this specific command
    user_commands = access_control.get_terraform_allowed_commands(email)
    cmd_first = req.command.strip().split()[0] if req.command.strip() else ""
    if cmd_first not in user_commands:
        return {
            "error": f"You don't have permission to run 'terraform {req.command}'.",
            "your_allowed_commands": user_commands
        }

    try:
        repo = get_repo_path()
        result = terraform_tools.run_command(repo, req.command)
        result["user"] = email
        return result
    except Exception as e:
        return {"error": str(e)}


# ── GIT ENDPOINTS ────────────────────────────────────

@app.get("/git_status", operation_id="git_status",
         summary="Show git status — modified, untracked, staged files")
def git_status_endpoint(request: Request):
    """Show the current git status of the repository. Shows modified, untracked, and staged files."""
    email = get_email(request)
    allowed, reason = access_control.check_permission(email, "read")
    if not allowed:
        return {"error": reason}

    try:
        repo = get_repo_path()
        result = git_tools.git_status(repo)
        result["user"] = email
        return result
    except Exception as e:
        return {"error": str(e)}


@app.get("/git_diff", operation_id="git_diff",
         summary="Show git diff for changes")
def git_diff_endpoint(request: Request,
                      file_path: str = Query("", description="File to diff. Empty for all changes.")):
    """Show what has changed in the repository since the last commit."""
    email = get_email(request)
    allowed, reason = access_control.check_permission(email, "read")
    if not allowed:
        return {"error": reason}

    try:
        repo = get_repo_path()
        result = git_tools.git_diff(repo, file_path)
        result["user"] = email
        return result
    except Exception as e:
        return {"error": str(e)}


@app.get("/git_log", operation_id="git_log",
         summary="Show recent git commits")
def git_log_endpoint(request: Request,
                     count: int = Query(10, description="Number of commits to show")):
    """Show the recent commit history of the repository."""
    email = get_email(request)
    allowed, reason = access_control.check_permission(email, "read")
    if not allowed:
        return {"error": reason}

    try:
        repo = get_repo_path()
        result = git_tools.git_log(repo, count)
        result["user"] = email
        return result
    except Exception as e:
        return {"error": str(e)}


@app.post("/git_commit", operation_id="git_commit",
          summary="Stage all changes and commit")
def git_commit_endpoint(req: GitCommitRequest, request: Request):
    """Stage all modified files and create a git commit. Read-write access required."""
    email = get_email(request)
    allowed, reason = access_control.check_permission(email, "write")
    if not allowed:
        return {"error": reason}

    try:
        repo = get_repo_path()

        # Stage all changes
        add_result = git_tools.git_add(repo, ".")
        if "error" in add_result:
            return add_result

        # Commit
        result = git_tools.git_commit(repo, req.message)
        result["user"] = email
        logger.info(f"User {email} committed: {req.message}")
        return result
    except Exception as e:
        return {"error": str(e)}


@app.post("/git_push", operation_id="git_push",
          summary="Push commits to remote repository (admin only)")
def git_push_endpoint(req: GitPushRequest, request: Request):
    """Push commits to the remote Git repository. Admin access required. Call first without confirm=true to preview, then call again with confirm=true to actually push."""
    email = get_email(request)
    allowed, reason = access_control.check_permission(email, "push")
    if not allowed:
        return {"error": reason}

    try:
        repo = get_repo_path()

        if not req.confirm:
            # Preview mode — show what would be pushed
            status = git_tools.git_status(repo)
            log = git_tools.git_log(repo, 5)
            branch = git_tools.git_branch(repo)

            return {
                "action": "preview",
                "message": "Ready to push. Call again with confirm=true to push.",
                "current_branch": branch.get("current_branch", "unknown"),
                "recent_commits": log.get("commits", [])[:5],
                "status": status
            }

        # Actually push
        result = git_tools.git_push(repo, req.branch)
        result["user"] = email
        logger.info(f"User {email} pushed to remote")
        return result
    except Exception as e:
        return {"error": str(e)}


@app.get("/git_branch", operation_id="git_branch",
         summary="Show current and available branches")
def git_branch_endpoint(request: Request):
    """Show the current branch and list all available branches."""
    email = get_email(request)
    allowed, reason = access_control.check_permission(email, "read")
    if not allowed:
        return {"error": reason}

    try:
        repo = get_repo_path()
        result = git_tools.git_branch(repo)
        result["user"] = email
        return result
    except Exception as e:
        return {"error": str(e)}


# ── INFO ENDPOINT ────────────────────────────────────

@app.get("/agent_info", operation_id="get_agent_info",
         summary="Get agent information and your access level")
def agent_info(request: Request):
    """Get information about the agent and your current access level."""
    email = get_email(request)
    access_level, user_name = access_control.get_user_access(email)
    tf_commands = access_control.get_terraform_allowed_commands(email)

    capabilities = {
        "read-only": ["view files", "analyze repo", "terraform plan/validate", "git status/log"],
        "read-write": ["view files", "analyze repo", "create/modify/delete files", "all terraform commands", "git commit"],
        "admin": ["everything", "including git push to remote"],
        "deny": []
    }

    return {
        "agent": "Terraform Coding Agent",
        "version": "1.0.0",
        "user": email,
        "user_name": user_name,
        "access_level": access_level,
        "capabilities": capabilities.get(access_level, []),
        "terraform_commands": tf_commands,
        "repo_path": REPO_PATH
    }


# ── Run ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8001"))

    print(f"""
╔══════════════════════════════════════════════╗
║     Terraform Coding Agent v1.0.0            ║
╠══════════════════════════════════════════════╣
║  Repo:   {REPO_PATH or 'NOT SET!': <35}║
║  Port:   {port: <35}║
║  Docs:   http://0.0.0.0:{port}/docs{' ' * (28 - len(str(port)))}║
║  Log:    {LOG_FILE: <35}║
╚══════════════════════════════════════════════╝
""")

    if not REPO_PATH:
        print("⚠️  WARNING: TERRAFORM_REPO_PATH not set in .env!")
        print("   Set it before using the agent.\n")

    uvicorn.run(app, host="0.0.0.0", port=port)
