"""Approval workflow utilities for the coding agent."""

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from langchain_core.tools import tool


@tool
def apply_pending_change(file_path: str, new_content: str) -> str:
    """Apply an approved change by writing the new content to the file.
    
    This tool should be used after a modification has been approved.
    Use the 'modified' content from the pending change dictionary.
    
    Args:
        file_path: Path to the file to write.
        new_content: The new content to write (from the 'modified' field of the pending change).
        
    Returns:
        Success message or error description.
    """
    try:
        path = Path(file_path)
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return f"Successfully applied change to '{file_path}'."
    except PermissionError:
        return f"Error: Permission denied writing to '{file_path}'."
    except Exception as e:
        return f"Error applying change: {str(e)}"


@dataclass
class PendingChange:
    """Represents a pending code modification awaiting approval."""
    file_path: str
    original_content: str
    new_content: str
    diff: str
    description: str
    change_type: str = "modify"  # modify, create, delete


def generate_unified_diff(original: str, modified: str, filename: str) -> str:
    """Generate a unified diff between original and modified content.
    
    Args:
        original: Original file content.
        modified: Modified file content.
        filename: Name of the file for diff header.
        
    Returns:
        Unified diff as a string.
    """
    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)
    
    # Ensure lines end with newlines for proper diff format
    if original_lines and not original_lines[-1].endswith('\n'):
        original_lines[-1] += '\n'
    if modified_lines and not modified_lines[-1].endswith('\n'):
        modified_lines[-1] += '\n'
    
    diff_lines = list(difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm='\n'
    ))
    
    return ''.join(diff_lines)


def create_modification_plan(changes: list[PendingChange]) -> str:
    """Create a formatted modification plan for user review.
    
    Args:
        changes: List of PendingChange objects.
        
    Returns:
        Formatted plan as a string with diffs.
    """
    if not changes:
        return "No changes to apply."
    
    plan_lines = [
        "=" * 60,
        "MODIFICATION PLAN",
        "=" * 60,
        "",
        f"Total files to modify: {len(changes)}",
        "",
    ]
    
    for i, change in enumerate(changes, 1):
        plan_lines.extend([
            "-" * 60,
            f"Change {i}/{len(changes)}: {change.change_type.upper()}",
            f"File: {change.file_path}",
            f"Description: {change.description}",
            "",
            "Diff:",
            change.diff,
            "",
        ])
    
    plan_lines.extend([
        "=" * 60,
        "END OF MODIFICATION PLAN",
        "=" * 60,
    ])
    
    return "\n".join(plan_lines)


def format_changes_for_display(changes: list[dict[str, Any]]) -> str:
    """Format a list of changes for display to the user.
    
    Args:
        changes: List of change dictionaries from modification tools.
        
    Returns:
        Formatted string for display.
    """
    if not changes:
        return "No changes pending."
    
    output_lines = [
        "=" * 60,
        "PROPOSED MODIFICATIONS",
        "=" * 60,
    ]
    
    for i, change in enumerate(changes, 1):
        file_path = change.get("file", "Unknown file")
        description = change.get("description", "No description")
        diff = change.get("diff", "")
        
        output_lines.extend([
            "",
            f"{i}. {file_path}",
            f"   {description}",
            "-" * 60,
        ])
        
        # Add diff preview (first 20 lines)
        diff_lines = diff.split('\n')[:20]
        for line in diff_lines:
            output_lines.append(f"  {line}")
        
        if len(diff.split('\n')) > 20:
            output_lines.append("  ... (diff truncated)")
        
        output_lines.append("-" * 60)
    
    output_lines.extend([
        "",
        "Type 'approve' to apply these changes",
        "Type 'reject' to cancel",
        "Or describe requested changes",
        "=" * 60,
    ])
    
    return "\n".join(output_lines)


def format_push_request(branch: str, commit_count: int, files_changed: list[str]) -> str:
    """Format a push approval request for display.
    
    Args:
        branch: Name of the branch to push.
        commit_count: Number of commits to push.
        files_changed: List of files that were changed.
        
    Returns:
        Formatted push request string.
    """
    output_lines = [
        "=" * 60,
        "PUSH APPROVAL REQUEST",
        "=" * 60,
        f"Branch: {branch}",
        f"Commits to push: {commit_count}",
        "-" * 60,
        "Files Changed:",
    ]
    
    for file in files_changed[:15]:  # Limit to 15 files
        output_lines.append(f"  {file}")
    
    if len(files_changed) > 15:
        output_lines.append(f"  ... and {len(files_changed) - 15} more files")
    
    output_lines.extend([
        "-" * 60,
        "Type 'push' to push to remote",
        "Type 'cancel' to skip pushing",
        "=" * 60,
    ])
    
    return "\n".join(output_lines)
