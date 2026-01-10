"""File operation tools for the coding agent."""

import os
from pathlib import Path
from langchain_core.tools import tool


@tool
def read_file(path: str) -> str:
    """Read the contents of a file.
    
    Args:
        path: Absolute or relative path to the file to read.
        
    Returns:
        The contents of the file as a string.
    """
    try:
        file_path = Path(path)
        if not file_path.exists():
            return f"Error: File '{path}' does not exist."
        if not file_path.is_file():
            return f"Error: '{path}' is not a file."
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except PermissionError:
        return f"Error: Permission denied reading '{path}'."
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file.
    
    Args:
        path: Absolute or relative path to the file to write.
        content: The content to write to the file.
        
    Returns:
        Success message or error description.
    """
    try:
        file_path = Path(path)
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to '{path}'."
    except PermissionError:
        return f"Error: Permission denied writing to '{path}'."
    except Exception as e:
        return f"Error writing file: {str(e)}"


@tool
def list_directory(path: str, max_items: int = 50) -> str:
    """List the contents of a directory.
    
    Args:
        path: Absolute or relative path to the directory.
        max_items: Maximum number of items to return (default: 50).
        
    Returns:
        A formatted string listing files and directories.
        Results are automatically limited to prevent overwhelming output.
    """
    try:
        dir_path = Path(path)
        if not dir_path.exists():
            return f"Error: Directory '{path}' does not exist."
        if not dir_path.is_dir():
            return f"Error: '{path}' is not a directory."
        
        all_entries = []
        skipped_dirs = []
        
        for entry in sorted(dir_path.iterdir()):
            # Skip large inventories directories
            if entry.is_dir() and entry.name.lower() == "inventories":
                try:
                    file_count = sum(1 for _ in entry.rglob("*") if _.is_file())
                    if file_count > 30:
                        skipped_dirs.append(f"{entry.name} ({file_count} files)")
                        continue
                except Exception:
                    pass
            
            entry_type = "[DIR]" if entry.is_dir() else "[FILE]"
            all_entries.append(f"{entry_type} {entry.name}")
        
        if not all_entries:
            return f"Directory '{path}' is empty."
        
        # Limit results
        total_count = len(all_entries)
        entries = all_entries[:max_items]
        
        result = f"Contents of '{path}' ({total_count} items):\n" + "\n".join(entries)
        
        if total_count > max_items:
            result += f"\n\n[FILTERED: Showing {max_items} of {total_count} items. Use specific patterns to find more.]"
        
        if skipped_dirs:
            result += f"\n[SKIPPED large directories: {', '.join(skipped_dirs)}]"
        
        return result
    except PermissionError:
        return f"Error: Permission denied accessing '{path}'."
    except Exception as e:
        return f"Error listing directory: {str(e)}"


@tool
def delete_file(path: str) -> str:
    """Delete a file.
    
    Args:
        path: Absolute or relative path to the file to delete.
        
    Returns:
        Success message or error description.
    """
    try:
        file_path = Path(path)
        if not file_path.exists():
            return f"Error: File '{path}' does not exist."
        if not file_path.is_file():
            return f"Error: '{path}' is not a file."
        
        file_path.unlink()
        return f"Successfully deleted '{path}'."
    except PermissionError:
        return f"Error: Permission denied deleting '{path}'."
    except Exception as e:
        return f"Error deleting file: {str(e)}"


@tool
def file_exists(path: str) -> str:
    """Check if a file or directory exists.
    
    Args:
        path: Absolute or relative path to check.
        
    Returns:
        A message indicating whether the path exists and its type.
    """
    try:
        file_path = Path(path)
        if not file_path.exists():
            return f"Path '{path}' does not exist."
        elif file_path.is_file():
            return f"Path '{path}' exists and is a file."
        elif file_path.is_dir():
            return f"Path '{path}' exists and is a directory."
        else:
            return f"Path '{path}' exists but is of unknown type."
    except Exception as e:
        return f"Error checking path: {str(e)}"
