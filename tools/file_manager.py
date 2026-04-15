"""
File Manager Tool
Handles all file operations: list, read, create, modify, delete Terraform files.
Creates backups before any modification.
"""

import os
import re
import difflib
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger("agent.file_manager")

IGNORE_DIRS = {".git", ".terraform", ".venv", "node_modules", "__pycache__", ".idea", ".vscode"}
ALLOWED_EXTS = {".tf", ".tfvars", ".hcl", ".md", ".txt", ".yml", ".yaml", ".json", ".sh"}


def safe_resolve(root: Path, relpath: str) -> Path:
    """Resolve a relative path safely, blocking path traversal attacks."""
    p = (root / relpath).resolve()
    if root.resolve() not in p.parents and p != root.resolve():
        raise ValueError(f"Blocked path traversal: {relpath}")
    return p


def list_files(repo_path: str) -> Dict:
    """List all Terraform and related files in the repo."""
    root = Path(repo_path).resolve()
    if not root.exists():
        return {"error": f"Repo path not found: {repo_path}"}

    files = []
    for p in root.rglob("*"):
        if len(files) >= 500:
            break
        if any(part in IGNORE_DIRS for part in p.parts):
            continue
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            try:
                files.append(str(p.resolve().relative_to(root)))
            except:
                pass
    files.sort()

    tf_files = [f for f in files if f.endswith(".tf")]
    other_files = [f for f in files if not f.endswith(".tf")]

    logger.info(f"Listed {len(files)} files ({len(tf_files)} .tf)")
    return {
        "terraform_files": tf_files,
        "other_files": other_files,
        "total": len(files)
    }


def show_file(repo_path: str, file_path: str) -> Dict:
    """Read and return the contents of a file."""
    root = Path(repo_path).resolve()
    try:
        p = safe_resolve(root, file_path)
        if not p.exists():
            return {"error": f"File not found: {file_path}"}

        text = p.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        truncated = len(lines) > 300
        if truncated:
            lines = lines[:300]

        logger.info(f"Showed file: {file_path} ({len(lines)} lines)")
        return {
            "file_path": file_path,
            "content": "\n".join(lines),
            "lines": len(lines),
            "truncated": truncated
        }
    except Exception as e:
        return {"error": str(e)}


def create_file(repo_path: str, file_path: str, content: str, reason: str = "") -> Dict:
    """Create a new file. Will not overwrite existing files."""
    root = Path(repo_path).resolve()
    try:
        p = safe_resolve(root, file_path)
        if p.exists():
            return {"error": f"File already exists: {file_path}. Use modify_file instead."}

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

        logger.info(f"Created file: {file_path} (reason: {reason})")
        return {
            "result": f"Created: {file_path}",
            "reason": reason,
            "content": content
        }
    except Exception as e:
        return {"error": str(e)}


def modify_file(repo_path: str, file_path: str, new_content: str, reason: str = "") -> Dict:
    """Modify an existing file. Creates a .bak backup before changes."""
    root = Path(repo_path).resolve()
    try:
        p = safe_resolve(root, file_path)

        # Read old content
        old_text = ""
        if p.exists():
            old_text = p.read_text(encoding="utf-8", errors="ignore")

        # Generate diff
        diff = "".join(difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"{file_path} (original)",
            tofile=f"{file_path} (modified)"
        ))

        if not diff.strip():
            return {"result": "No changes detected."}

        # Create backup
        if p.exists():
            backup = p.with_suffix(p.suffix + ".bak")
            backup.write_text(old_text, encoding="utf-8")

        # Write new content
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(new_content, encoding="utf-8")

        logger.info(f"Modified file: {file_path} (reason: {reason})")
        return {
            "result": f"Modified: {file_path}",
            "reason": reason,
            "backup": f"{file_path}.bak",
            "diff": diff
        }
    except Exception as e:
        return {"error": str(e)}


def delete_file(repo_path: str, file_path: str) -> Dict:
    """Delete a file. Creates a backup before deletion."""
    root = Path(repo_path).resolve()
    try:
        p = safe_resolve(root, file_path)
        if not p.exists():
            return {"error": f"File not found: {file_path}"}

        # Backup before delete
        backup = p.with_suffix(p.suffix + ".deleted.bak")
        backup.write_text(p.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        p.unlink()

        logger.info(f"Deleted file: {file_path}")
        return {
            "result": f"Deleted: {file_path}",
            "backup": backup.name
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_repo(repo_path: str) -> Dict:
    """Analyze the repo structure: providers, resources, variables, modules."""
    root = Path(repo_path).resolve()
    files = list_files(repo_path)
    tf_files = files.get("terraform_files", [])

    if not tf_files:
        return {"result": "No .tf files found in the repository."}

    summaries = {}
    all_providers = set()
    total_resources = 0
    total_variables = 0
    total_outputs = 0

    for f in tf_files:
        try:
            p = safe_resolve(root, f)
            text = p.read_text(encoding="utf-8", errors="ignore")

            providers = sorted(set(re.findall(r'^\s*provider\s+"([^"]+)"', text, flags=re.MULTILINE)))
            resources = re.findall(r'^\s*resource\s+"([^"]+)"\s+"([^"]+)"', text, flags=re.MULTILINE)
            variables = re.findall(r'^\s*variable\s+"([^"]+)"', text, flags=re.MULTILINE)
            outputs = re.findall(r'^\s*output\s+"([^"]+)"', text, flags=re.MULTILINE)
            modules = re.findall(r'^\s*module\s+"([^"]+)"', text, flags=re.MULTILINE)

            all_providers.update(providers)
            total_resources += len(resources)
            total_variables += len(variables)
            total_outputs += len(outputs)

            parts = [f"{f}:"]
            if providers: parts.append(f"Providers: {', '.join(providers)}")
            if resources: parts.append(f"Resources: {len(resources)}")
            if variables: parts.append(f"Variables: {len(variables)}")
            if outputs: parts.append(f"Outputs: {len(outputs)}")
            if modules: parts.append(f"Modules: {', '.join(modules)}")
            summaries[f] = " | ".join(parts)
        except:
            summaries[f] = f"{f}: Could not read"

    logger.info(f"Analyzed repo: {len(tf_files)} .tf files, {total_resources} resources")
    return {
        "tf_files": len(tf_files),
        "providers": sorted(all_providers),
        "total_resources": total_resources,
        "total_variables": total_variables,
        "total_outputs": total_outputs,
        "file_summaries": summaries
    }
