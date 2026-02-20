"""Shell command execution tool for the coding agent."""

import subprocess
import os
from typing import Optional
from langchain_core.tools import tool
from tools.utils import safe_tool


@tool
@safe_tool
def run_shell_command(command: str, working_dir: str = "") -> str:
    """Execute a shell command and return the output.
    
    Use this tool when no specific tool exists for a task.
    For example: finding files, searching content, running scripts, etc.
    
    Args:
        command: The shell command to execute.
        working_dir: Optional working directory for the command. 
                     If empty, uses REPO_PATH or current directory.
        
    Returns:
        The command output (stdout and stderr combined) or error message.
    """
    cwd = working_dir or os.environ.get("REPO_PATH", ".")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
        )
        
        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"[stderr]: {result.stderr}")
        
        output = "\n".join(output_parts) if output_parts else "(no output)"
        
        # Limit output size to avoid overwhelming the LLM
        max_output = 10000
        if len(output) > max_output:
            output = output[:max_output] + f"\n... (output truncated, {len(output)} total chars)"
        
        return f"Exit code: {result.returncode}\n{output}"
        
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after 120 seconds: {command}"
    except Exception as e:
        return f"Error executing command: {str(e)}"


@tool
@safe_tool
def find_files(pattern: str, directory: str = ".", max_results: int = 50) -> str:
    """Find files matching a pattern using find command.
    
    Args:
        pattern: File name pattern (e.g., "*.yml", "*.py", "playbook*")
        directory: Directory to search in (default: current directory)
        max_results: Maximum number of results to return (default: 50)
        
    Returns:
        List of matching file paths.
    """
    cwd = os.environ.get("REPO_PATH", ".")
    search_dir = os.path.join(cwd, directory) if directory != "." else cwd
    
    try:
        # Use find command for pattern matching
        cmd = f"find {search_dir} -name '{pattern}' -type f 2>/dev/null | head -n {max_results}"
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        files = result.stdout.strip()
        if not files:
            return f"No files found matching pattern '{pattern}' in '{directory}'"
        
        file_list = files.splitlines()
        file_count = len(file_list)
        msg = f"Found {file_count} file(s) matching '{pattern}':\n{files}"
        
        if file_count >= max_results:
            msg += f"\n\n[FILTERED: Results limited to {max_results}. Use a more specific pattern to narrow results.]"
        
        return msg
        
    except subprocess.TimeoutExpired:
        return "Error: Search timed out"
    except Exception as e:
        return f"Error searching files: {str(e)}"


@tool
@safe_tool
def search_in_files(search_term: str, file_pattern: str = "*", directory: str = ".") -> str:
    """Search for a term within files using grep.
    
    Args:
        search_term: The text to search for.
        file_pattern: File pattern to search in (e.g., "*.py", "*.yml")
        directory: Directory to search in (default: current directory)
        
    Returns:
        Matching lines with file names and line numbers.
    """
    cwd = os.environ.get("REPO_PATH", ".")
    search_dir = os.path.join(cwd, directory) if directory != "." else cwd
    
    try:
        # Use grep with recursive search
        cmd = f"grep -rn --include='{file_pattern}' '{search_term}' {search_dir} 2>/dev/null | head -n 50"
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        matches = result.stdout.strip()
        if not matches:
            return f"No matches found for '{search_term}' in {file_pattern} files"
        
        match_lines = matches.splitlines()
        match_count = len(match_lines)
        msg = f"Found {match_count} match(es) for '{search_term}':\n{matches}"
        
        if match_count >= 50:
            msg += "\n\n[FILTERED: Results limited to 50 matches. Use a more specific search term or file pattern.]"
        
        return msg
        
    except subprocess.TimeoutExpired:
        return "Error: Search timed out"
    except Exception as e:
        return f"Error searching: {str(e)}"
