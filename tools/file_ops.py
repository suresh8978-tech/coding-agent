"""File operation tools for the coding agent."""

import os
from pathlib import Path
from langchain_core.tools import tool
from tools.utils import safe_tool


# Lines per chunk when reading large files.
# Override via FILE_CHUNK_SIZE in .env  (optimal for claude-sonnet-4-5: 500).
CHUNK_SIZE = int(os.environ.get("FILE_CHUNK_SIZE", 500))


@tool
@safe_tool
def read_file(path: str, start_line: int = 1, end_line: int = 0) -> str:
    """Read the contents of a file, with optional line range for chunked reading.

    For files with more than 200 lines, the file is automatically read in chunks.
    When no range is specified, the first chunk (lines 1-200) is returned along
    with metadata indicating total line count and available chunks, so you know
    to call this tool again with the next range to read subsequent chunks.

    Args:
        path: Absolute or relative path to the file to read.
        start_line: 1-based line number to start reading from (default: 1).
        end_line: 1-based line number to stop reading at (inclusive).
                  Use 0 (default) to auto-determine based on chunk size.

    Returns:
        The requested lines of the file with chunk metadata for large files.
    """
    try:
        file_path = Path(path)
        if not file_path.exists():
            return f"Error: File '{path}' does not exist."
        if not file_path.is_file():
            return f"Error: '{path}' is not a file."

        with open(file_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()

        total_lines = len(all_lines)

        if total_lines <= CHUNK_SIZE and start_line == 1 and end_line == 0:
            return "".join(all_lines)

        # Resolve effective range
        effective_start = max(1, start_line)
        if end_line == 0:
            effective_end = min(effective_start + CHUNK_SIZE - 1, total_lines)
        else:
            effective_end = min(end_line, total_lines)

        chunk_lines = all_lines[effective_start - 1:effective_end]
        chunk_content = "".join(chunk_lines)

        total_chunks = (total_lines + CHUNK_SIZE - 1) // CHUNK_SIZE
        current_chunk = (effective_start - 1) // CHUNK_SIZE + 1
        has_more = effective_end < total_lines

        header = (
            f"[FILE: {path}]\n"
            f"[Total lines: {total_lines} | Chunk {current_chunk}/{total_chunks} | "
            f"Lines {effective_start}-{effective_end}]\n"
        )
        if has_more:
            next_start = effective_end + 1
            next_end = min(next_start + CHUNK_SIZE - 1, total_lines)
            header += (
                f"[MORE CONTENT AVAILABLE: call read_file(path='{path}', "
                f"start_line={next_start}, end_line={next_end}) for next chunk]\n"
            )
        else:
            header += "[END OF FILE]\n"

        header += "-" * 60 + "\n"
        return header + chunk_content

    except PermissionError:
        return f"Error: Permission denied reading '{path}'."
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
@safe_tool
def write_file(
    path: str,
    content: str,
    mode: str = "write",
    start_line: int = 0,
    end_line: int = 0,
) -> str:
    """Write or edit content in a file, with support for chunked and in-place edits.

    For files larger than 200 lines, content MUST be written in chunks using
    this tool multiple times rather than in a single call:
      - First chunk:      mode='write'  — creates/overwrites the file
      - Later chunks:     mode='append' — appends each subsequent chunk
      - Line-range edit:  mode='patch'  — replaces only the lines start_line..end_line

    Args:
        path: Absolute or relative path to the file to write.
        content: The content to write (chunk text, not the whole file for large files).
        mode: Write mode —
              'write'  : overwrite the entire file (use for first chunk or small files).
              'append' : append content to the end of an existing file (use for subsequent chunks).
              'patch'  : replace lines start_line..end_line (inclusive, 1-based) with content.
        start_line: First line to replace when mode='patch' (1-based, inclusive).
        end_line:   Last line to replace when mode='patch' (1-based, inclusive).

    Returns:
        Success message with current line count, or error description.
    """
    try:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if mode == "write":
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

        elif mode == "append":
            if not file_path.exists():
                return f"Error: Cannot append — file '{path}' does not exist. Use mode='write' first."
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(content)

        elif mode == "patch":
            if not file_path.exists():
                return f"Error: Cannot patch — file '{path}' does not exist."
            if start_line < 1 or end_line < start_line:
                return "Error: patch mode requires start_line >= 1 and end_line >= start_line."
            with open(file_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
            total = len(all_lines)
            if start_line > total:
                return f"Error: start_line {start_line} exceeds file length ({total} lines)."
            patched_end = min(end_line, total)
            patch_lines = content.splitlines(keepends=True)
            if patch_lines and not patch_lines[-1].endswith('\n'):
                patch_lines[-1] += '\n'
            new_lines = all_lines[:start_line - 1] + patch_lines + all_lines[patched_end:]
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)

        else:
            return f"Error: Unknown mode '{mode}'. Use 'write', 'append', or 'patch'."

        final_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
        return f"Successfully wrote to '{path}' (mode='{mode}', file now {final_lines} lines)."

    except PermissionError:
        return f"Error: Permission denied writing to '{path}'."
    except Exception as e:
        return f"Error writing file: {str(e)}"


@tool
@safe_tool
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
@safe_tool
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
