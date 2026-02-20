"""Python code modification tools."""

from pathlib import Path
from typing import Any
from langchain_core.tools import tool
from tools.utils import safe_tool


def _get_sg_root(source: str, language: str = "python"):
    """Get an SgRoot for parsing source code."""
    try:
        from ast_grep_py import SgRoot
        return SgRoot(source, language)
    except ImportError:
        raise ImportError("ast-grep-py not installed. Run: pip install ast-grep-py")


@tool
@safe_tool
def modify_python_code(path: str, pattern: str, replacement: str) -> dict[str, Any]:
    """Modify Python code by replacing patterns using ast-grep.
    
    This tool finds code matching the pattern and generates the modified
    content. It does NOT write to the file - the modification must be
    approved first.
    
    Args:
        path: Path to the Python file.
        pattern: ast-grep pattern to match (e.g., "print($A)").
        replacement: Replacement text for matched code.
        
    Returns:
        Dictionary with original content, modified content, and diff.
    """
    file_path = Path(path)
    if not file_path.exists():
        return {"error": f"File '{path}' does not exist."}
    
    try:
        original = file_path.read_text(encoding='utf-8')
        root = _get_sg_root(original)
        node = root.root()
    except ImportError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Error: {str(e)}"}
    
    matches = node.find_all(pattern=pattern)
    if not matches:
        return {
            "error": f"No matches found for pattern: {pattern}",
            "file": path,
        }
    
    # Generate edits
    edits = []
    for match in matches:
        edit = match.replace(replacement)
        edits.append(edit)
    
    # Apply edits to get new source
    modified = node.commit_edits(edits)
    
    # Generate diff
    from tools.approval import generate_unified_diff
    diff = generate_unified_diff(original, modified, path)
    
    return {
        "file": path,
        "original": original,
        "modified": modified,
        "diff": diff,
        "match_count": len(matches),
        "description": f"Replace {len(matches)} occurrence(s) of '{pattern}' with '{replacement}'",
    }


@tool
@safe_tool
def add_import(path: str, import_stmt: str) -> dict[str, Any]:
    """Add an import statement to a Python file.
    
    The import will be added after existing imports or at the top of the file.
    
    Args:
        path: Path to the Python file.
        import_stmt: The import statement to add (e.g., "import os" or "from typing import List").
        
    Returns:
        Dictionary with original content, modified content, and diff.
    """
    file_path = Path(path)
    if not file_path.exists():
        return {"error": f"File '{path}' does not exist."}
    
    try:
        original = file_path.read_text(encoding='utf-8')
        root = _get_sg_root(original)
        node = root.root()
    except ImportError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Error: {str(e)}"}
    
    lines = original.splitlines(keepends=True)
    
    # Find the last import statement
    last_import_line = 0
    for imp in node.find_all(kind="import_statement"):
        last_import_line = max(last_import_line, imp.range().end.line + 1)
    for imp in node.find_all(kind="import_from_statement"):
        last_import_line = max(last_import_line, imp.range().end.line + 1)
    
    # Check if import already exists
    if import_stmt.strip() in original:
        return {
            "error": f"Import '{import_stmt}' already exists in file.",
            "file": path,
        }
    
    # Insert the import
    import_line = import_stmt.strip() + "\n"
    if last_import_line > 0:
        lines.insert(last_import_line, import_line)
    else:
        # No existing imports, add at the beginning (after any docstring/comments)
        insert_pos = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''"):
                insert_pos = i
                break
        lines.insert(insert_pos, import_line)
    
    modified = "".join(lines)
    
    from tools.approval import generate_unified_diff
    diff = generate_unified_diff(original, modified, path)
    
    return {
        "file": path,
        "original": original,
        "modified": modified,
        "diff": diff,
        "description": f"Add import: {import_stmt}",
    }


@tool
@safe_tool
def add_function(path: str, func_code: str, after: str = "") -> dict[str, Any]:
    """Add a new function to a Python file.
    
    Args:
        path: Path to the Python file.
        func_code: The complete function code to add.
        after: Name of function/class after which to insert. Empty = end of file.
        
    Returns:
        Dictionary with original content, modified content, and diff.
    """
    file_path = Path(path)
    if not file_path.exists():
        return {"error": f"File '{path}' does not exist."}
    
    try:
        original = file_path.read_text(encoding='utf-8')
        root = _get_sg_root(original)
        node = root.root()
    except ImportError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Error: {str(e)}"}
    
    lines = original.splitlines(keepends=True)
    insert_line = len(lines)  # Default to end of file
    
    if after:
        # Find the function or class to insert after
        found = False
        for func in node.find_all(kind="function_definition"):
            name_node = func.find(kind="identifier")
            if name_node and name_node.text() == after:
                insert_line = func.range().end.line + 1
                found = True
                break
        
        if not found:
            for cls in node.find_all(kind="class_definition"):
                name_node = cls.find(kind="identifier")
                if name_node and name_node.text() == after:
                    insert_line = cls.range().end.line + 1
                    found = True
                    break
        
        if not found:
            return {
                "error": f"Could not find function or class named '{after}'.",
                "file": path,
            }
    
    # Ensure proper formatting
    func_to_insert = "\n\n" + func_code.strip() + "\n"
    
    # Insert the function
    if insert_line >= len(lines):
        modified = original.rstrip() + func_to_insert
    else:
        lines.insert(insert_line, func_to_insert)
        modified = "".join(lines)
    
    from tools.approval import generate_unified_diff
    diff = generate_unified_diff(original, modified, path)
    
    # Extract function name for description
    func_name = "new function"
    try:
        func_root = _get_sg_root(func_code)
        func_node = func_root.root().find(kind="function_definition")
        if func_node:
            name_node = func_node.find(kind="identifier")
            if name_node:
                func_name = name_node.text()
    except Exception:
        pass
    
    return {
        "file": path,
        "original": original,
        "modified": modified,
        "diff": diff,
        "description": f"Add function '{func_name}'" + (f" after '{after}'" if after else " at end of file"),
    }
