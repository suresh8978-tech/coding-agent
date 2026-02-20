"""Python code analysis tools using ast-grep."""

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
def analyze_python_file(path: str) -> dict[str, Any]:
    """Analyze a Python file's structure using ast-grep.
    
    Args:
        path: Path to the Python file to analyze.
        
    Returns:
        Dictionary containing file structure analysis:
        - functions: List of function definitions
        - classes: List of class definitions
        - imports: List of import statements
        - global_variables: List of global variable assignments
    """
    file_path = Path(path)
    if not file_path.exists():
        return {"error": f"File '{path}' does not exist."}
    if not file_path.suffix == '.py':
        return {"error": f"File '{path}' is not a Python file."}
    
    try:
        source = file_path.read_text(encoding='utf-8')
    except Exception as e:
        return {"error": f"Error reading file: {str(e)}"}
    
    try:
        root = _get_sg_root(source)
        node = root.root()
    except ImportError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Error parsing Python code: {str(e)}"}
    
    result = {
        "file": path,
        "functions": [],
        "classes": [],
        "imports": [],
        "global_variables": [],
        "line_count": len(source.splitlines()),
    }
    
    # Find function definitions
    func_nodes = node.find_all(kind="function_definition")
    for func in func_nodes:
        name_node = func.find(kind="identifier")
        if name_node:
            result["functions"].append({
                "name": name_node.text(),
                "start_line": func.range().start.line + 1,
                "end_line": func.range().end.line + 1,
                "text": func.text()[:200] + "..." if len(func.text()) > 200 else func.text(),
            })
    
    # Find class definitions
    class_nodes = node.find_all(kind="class_definition")
    for cls in class_nodes:
        name_node = cls.find(kind="identifier")
        if name_node:
            # Find methods in class
            methods = []
            method_nodes = cls.find_all(kind="function_definition")
            for method in method_nodes:
                method_name = method.find(kind="identifier")
                if method_name:
                    methods.append(method_name.text())
            
            result["classes"].append({
                "name": name_node.text(),
                "start_line": cls.range().start.line + 1,
                "end_line": cls.range().end.line + 1,
                "methods": methods,
            })
    
    # Find imports
    import_nodes = node.find_all(kind="import_statement")
    for imp in import_nodes:
        result["imports"].append({
            "text": imp.text(),
            "line": imp.range().start.line + 1,
        })
    
    import_from_nodes = node.find_all(kind="import_from_statement")
    for imp in import_from_nodes:
        result["imports"].append({
            "text": imp.text(),
            "line": imp.range().start.line + 1,
        })
    
    return result


@tool
@safe_tool
def find_python_pattern(path: str, pattern: str) -> list[dict[str, Any]]:
    """Find code matching a pattern in a Python file using ast-grep.
    
    Args:
        path: Path to the Python file.
        pattern: ast-grep pattern to search for (e.g., "print($A)" to find print calls).
        
    Returns:
        List of matches with their location and text.
    """
    file_path = Path(path)
    if not file_path.exists():
        return [{"error": f"File '{path}' does not exist."}]
    
    try:
        source = file_path.read_text(encoding='utf-8')
        root = _get_sg_root(source)
        node = root.root()
    except ImportError as e:
        return [{"error": str(e)}]
    except Exception as e:
        return [{"error": f"Error: {str(e)}"}]
    
    matches = node.find_all(pattern=pattern)
    results = []
    
    for match in matches:
        results.append({
            "text": match.text(),
            "start_line": match.range().start.line + 1,
            "end_line": match.range().end.line + 1,
            "start_col": match.range().start.column,
            "end_col": match.range().end.column,
        })
    
    return results if results else [{"message": f"No matches found for pattern: {pattern}"}]


@tool
@safe_tool
def find_functions(path: str) -> list[dict[str, Any]]:
    """Find all function definitions in a Python file.
    
    Args:
        path: Path to the Python file.
        
    Returns:
        List of function definitions with name, signature, and location.
    """
    file_path = Path(path)
    if not file_path.exists():
        return [{"error": f"File '{path}' does not exist."}]
    
    try:
        source = file_path.read_text(encoding='utf-8')
        root = _get_sg_root(source)
        node = root.root()
    except ImportError as e:
        return [{"error": str(e)}]
    except Exception as e:
        return [{"error": f"Error: {str(e)}"}]
    
    func_nodes = node.find_all(kind="function_definition")
    results = []
    
    for func in func_nodes:
        name_node = func.find(kind="identifier")
        params_node = func.find(kind="parameters")
        
        if name_node:
            results.append({
                "name": name_node.text(),
                "parameters": params_node.text() if params_node else "()",
                "start_line": func.range().start.line + 1,
                "end_line": func.range().end.line + 1,
            })
    
    return results if results else [{"message": "No functions found in file."}]


@tool
@safe_tool
def find_classes(path: str) -> list[dict[str, Any]]:
    """Find all class definitions in a Python file.
    
    Args:
        path: Path to the Python file.
        
    Returns:
        List of class definitions with name, methods, and location.
    """
    file_path = Path(path)
    if not file_path.exists():
        return [{"error": f"File '{path}' does not exist."}]
    
    try:
        source = file_path.read_text(encoding='utf-8')
        root = _get_sg_root(source)
        node = root.root()
    except ImportError as e:
        return [{"error": str(e)}]
    except Exception as e:
        return [{"error": f"Error: {str(e)}"}]
    
    class_nodes = node.find_all(kind="class_definition")
    results = []
    
    for cls in class_nodes:
        name_node = cls.find(kind="identifier")
        if name_node:
            # Find base classes
            bases = []
            arg_list = cls.find(kind="argument_list")
            if arg_list:
                for arg in arg_list.find_all(kind="identifier"):
                    bases.append(arg.text())
            
            # Find methods
            methods = []
            for method in cls.find_all(kind="function_definition"):
                method_name = method.find(kind="identifier")
                if method_name:
                    methods.append(method_name.text())
            
            results.append({
                "name": name_node.text(),
                "bases": bases,
                "methods": methods,
                "start_line": cls.range().start.line + 1,
                "end_line": cls.range().end.line + 1,
            })
    
    return results if results else [{"message": "No classes found in file."}]


@tool
@safe_tool
def find_imports(path: str) -> list[dict[str, Any]]:
    """Find all import statements in a Python file.
    
    Args:
        path: Path to the Python file.
        
    Returns:
        List of import statements with their type and details.
    """
    file_path = Path(path)
    if not file_path.exists():
        return [{"error": f"File '{path}' does not exist."}]
    
    try:
        source = file_path.read_text(encoding='utf-8')
        root = _get_sg_root(source)
        node = root.root()
    except ImportError as e:
        return [{"error": str(e)}]
    except Exception as e:
        return [{"error": f"Error: {str(e)}"}]
    
    results = []
    
    # Regular imports
    for imp in node.find_all(kind="import_statement"):
        results.append({
            "type": "import",
            "text": imp.text(),
            "line": imp.range().start.line + 1,
        })
    
    # From imports
    for imp in node.find_all(kind="import_from_statement"):
        results.append({
            "type": "from_import",
            "text": imp.text(),
            "line": imp.range().start.line + 1,
        })
    
    return results if results else [{"message": "No imports found in file."}]
