"""Ansible content analysis tools using ansible-content-capture."""

import os
import sys
from pathlib import Path
from typing import Any
from langchain_core.tools import tool
from tools.utils import safe_tool


def _format_tree_node(node: Any, indent: int = 0) -> list[str]:
    """Format a tree node for display."""
    lines = []
    prefix = "  " * indent
    
    node_info = {
        "type": getattr(node, "type", "unknown"),
    }
    
    # Add common attributes if they exist
    if hasattr(node, "filepath"):
        node_info["filepath"] = node.filepath
    if hasattr(node, "name") and node.name:
        node_info["name"] = node.name
    if hasattr(node, "fqcn"):
        node_info["fqcn"] = node.fqcn
    
    lines.append(f"{prefix}{node_info}")
    
    # Recursively format children
    if hasattr(node, "children"):
        for child in node.children:
            lines.extend(_format_tree_node(child, indent + 1))
    
    return lines


@tool
@safe_tool
def scan_ansible_project(path: str) -> dict[str, Any]:
    """Scan an Ansible project directory and generate its AST.
    
    Uses ansible-content-capture to parse the project structure including
    playbooks, roles, tasks, and variables.
    
    Args:
        path: Path to the Ansible project directory.
        
    Returns:
        Dictionary containing the project structure and analysis.
    """
    project_path = Path(path)
    if not project_path.exists():
        return {"error": f"Path '{path}' does not exist."}
    if not project_path.is_dir():
        return {"error": f"Path '{path}' is not a directory."}
    
    try:
        from ansible_content_capture.scanner import AnsibleScanner
    except ImportError:
        return {"error": "ansible-content-capture not installed. Run: pip install ansible-content-capture"}
    
    try:
        scanner = AnsibleScanner()
        result = scanner.run(target_dir=str(project_path))
    except Exception as e:
        return {"error": f"Error scanning project: {str(e)}"}
    
    # Parse the result tree
    output = {
        "path": str(project_path),
        "playbooks": [],
        "roles": [],
        "taskfiles": [],
        "tree": [],
    }
    
    def traverse(node, parent_type=None):
        node_type = getattr(node, "type", "unknown")
        node_data = {
            "type": node_type,
            "filepath": getattr(node, "filepath", None),
            "name": getattr(node, "name", None),
        }
        
        if node_type == "playbook":
            output["playbooks"].append(node_data)
        elif node_type == "role":
            node_data["default_variables"] = getattr(node, "default_variables", {})
            output["roles"].append(node_data)
        elif node_type == "taskfile":
            output["taskfiles"].append(node_data)
        
        if hasattr(node, "children"):
            for child in node.children:
                traverse(child, node_type)
    
    if hasattr(result, "root") and result.root:
        traverse(result.root)
        output["tree"] = _format_tree_node(result.root)
    elif hasattr(result, "children"):
        for child in result.children:
            traverse(child)
    
    output["summary"] = {
        "playbook_count": len(output["playbooks"]),
        "role_count": len(output["roles"]),
        "taskfile_count": len(output["taskfiles"]),
    }
    
    return output


@tool
@safe_tool
def analyze_playbook(path: str) -> dict[str, Any]:
    """Analyze a single Ansible playbook file.
    
    Args:
        path: Path to the playbook YAML file.
        
    Returns:
        Dictionary containing playbook structure with plays and tasks.
    """
    file_path = Path(path)
    if not file_path.exists():
        return {"error": f"File '{path}' does not exist."}
    if not file_path.suffix in ['.yml', '.yaml']:
        return {"error": f"File '{path}' is not a YAML file."}
    
    try:
        import yaml
    except ImportError:
        return {"error": "PyYAML not installed. Run: pip install pyyaml"}
    
    try:
        with open(file_path, 'r') as f:
            content = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return {"error": f"YAML parsing error: {str(e)}"}
    except Exception as e:
        return {"error": f"Error reading file: {str(e)}"}
    
    if not isinstance(content, list):
        return {"error": "Invalid playbook format. Expected a list of plays."}
    
    result = {
        "file": path,
        "plays": [],
    }
    
    for play in content:
        if not isinstance(play, dict):
            continue
        
        play_data = {
            "name": play.get("name", "Unnamed play"),
            "hosts": play.get("hosts", ""),
            "tasks": [],
            "roles": [],
            "vars": list(play.get("vars", {}).keys()) if play.get("vars") else [],
            "handlers": [],
        }
        
        # Extract tasks
        for task in play.get("tasks", []):
            if isinstance(task, dict):
                task_info = {
                    "name": task.get("name", "Unnamed task"),
                }
                # Find the module used
                for key in task:
                    if key not in ["name", "when", "register", "loop", "with_items", "become", "tags", "notify"]:
                        task_info["module"] = key
                        break
                play_data["tasks"].append(task_info)
        
        # Extract roles
        for role in play.get("roles", []):
            if isinstance(role, str):
                play_data["roles"].append({"name": role})
            elif isinstance(role, dict):
                play_data["roles"].append({"name": role.get("role", role.get("name", "unknown"))})
        
        # Extract handlers
        for handler in play.get("handlers", []):
            if isinstance(handler, dict):
                play_data["handlers"].append({
                    "name": handler.get("name", "Unnamed handler"),
                })
        
        result["plays"].append(play_data)
    
    result["summary"] = {
        "play_count": len(result["plays"]),
        "total_tasks": sum(len(p["tasks"]) for p in result["plays"]),
        "total_roles": sum(len(p["roles"]) for p in result["plays"]),
    }
    
    return result


@tool
@safe_tool
def analyze_role(path: str) -> dict[str, Any]:
    """Analyze an Ansible role directory structure.
    
    Args:
        path: Path to the role directory.
        
    Returns:
        Dictionary containing role structure, tasks, variables, and handlers.
    """
    role_path = Path(path)
    if not role_path.exists():
        return {"error": f"Path '{path}' does not exist."}
    if not role_path.is_dir():
        return {"error": f"Path '{path}' is not a directory."}
    
    try:
        import yaml
    except ImportError:
        return {"error": "PyYAML not installed. Run: pip install pyyaml"}
    
    result = {
        "name": role_path.name,
        "path": str(role_path),
        "tasks": [],
        "handlers": [],
        "defaults": {},
        "vars": {},
        "meta": {},
        "templates": [],
        "files": [],
    }
    
    # Check for standard role directories
    tasks_dir = role_path / "tasks"
    handlers_dir = role_path / "handlers"
    defaults_dir = role_path / "defaults"
    vars_dir = role_path / "vars"
    meta_dir = role_path / "meta"
    templates_dir = role_path / "templates"
    files_dir = role_path / "files"
    
    def load_yaml_file(file_path: Path) -> Any:
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception:
            return None
    
    # Load tasks/main.yml
    if tasks_dir.exists():
        main_tasks = tasks_dir / "main.yml"
        if main_tasks.exists():
            tasks_content = load_yaml_file(main_tasks)
            if isinstance(tasks_content, list):
                for task in tasks_content:
                    if isinstance(task, dict):
                        result["tasks"].append({
                            "name": task.get("name", "Unnamed"),
                            "module": next((k for k in task if k not in ["name", "when", "register"]), None),
                        })
    
    # Load handlers/main.yml
    if handlers_dir.exists():
        main_handlers = handlers_dir / "main.yml"
        if main_handlers.exists():
            handlers_content = load_yaml_file(main_handlers)
            if isinstance(handlers_content, list):
                for handler in handlers_content:
                    if isinstance(handler, dict):
                        result["handlers"].append(handler.get("name", "Unnamed"))
    
    # Load defaults/main.yml
    if defaults_dir.exists():
        main_defaults = defaults_dir / "main.yml"
        if main_defaults.exists():
            result["defaults"] = load_yaml_file(main_defaults) or {}
    
    # Load vars/main.yml
    if vars_dir.exists():
        main_vars = vars_dir / "main.yml"
        if main_vars.exists():
            result["vars"] = load_yaml_file(main_vars) or {}
    
    # Load meta/main.yml
    if meta_dir.exists():
        main_meta = meta_dir / "main.yml"
        if main_meta.exists():
            result["meta"] = load_yaml_file(main_meta) or {}
    
    # List templates
    if templates_dir.exists():
        result["templates"] = [f.name for f in templates_dir.iterdir() if f.is_file()]
    
    # List files
    if files_dir.exists():
        result["files"] = [f.name for f in files_dir.iterdir() if f.is_file()]
    
    return result


@tool
@safe_tool
def find_tasks_using_module(path: str, module: str) -> list[dict[str, Any]]:
    """Find all tasks that use a specific Ansible module.
    
    Args:
        path: Path to search (playbook file or directory).
        module: The module name to search for (e.g., "yum", "apt", "copy").
        
    Returns:
        List of tasks using the specified module.
    """
    search_path = Path(path)
    if not search_path.exists():
        return [{"error": f"Path '{path}' does not exist."}]
    
    try:
        import yaml
    except ImportError:
        return [{"error": "PyYAML not installed. Run: pip install pyyaml"}]
    
    results = []
    yaml_files = []
    
    if search_path.is_file():
        yaml_files = [search_path]
    else:
        yaml_files = list(search_path.rglob("*.yml")) + list(search_path.rglob("*.yaml"))
    
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r') as f:
                content = yaml.safe_load(f)
        except Exception:
            continue
        
        def find_module_in_tasks(tasks, file_path, context=""):
            if not isinstance(tasks, list):
                return
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                if module in task or f"ansible.builtin.{module}" in task:
                    results.append({
                        "file": str(file_path),
                        "task_name": task.get("name", "Unnamed"),
                        "module": module,
                        "context": context,
                    })
        
        # Handle playbook format
        if isinstance(content, list):
            for play in content:
                if isinstance(play, dict):
                    find_module_in_tasks(play.get("tasks", []), yaml_file, f"play: {play.get('name', 'unnamed')}")
                    find_module_in_tasks(play.get("handlers", []), yaml_file, "handlers")
        # Handle task file format
        elif isinstance(content, dict) and "tasks" in content:
            find_module_in_tasks(content["tasks"], yaml_file)
    
    return results if results else [{"message": f"No tasks found using module '{module}'"}]


@tool
@safe_tool
def get_variable_usage(path: str) -> dict[str, Any]:
    """Analyze variable definitions and usage in Ansible content.
    
    Args:
        path: Path to the Ansible project or playbook.
        
    Returns:
        Dictionary with variable definitions, defaults, and usage locations.
    """
    search_path = Path(path)
    if not search_path.exists():
        return {"error": f"Path '{path}' does not exist."}
    
    try:
        import yaml
        import re
    except ImportError:
        return {"error": "Required packages not installed."}
    
    result = {
        "definitions": {},
        "defaults": {},
        "usage": [],
    }
    
    yaml_files = []
    if search_path.is_file():
        yaml_files = [search_path]
    else:
        yaml_files = list(search_path.rglob("*.yml")) + list(search_path.rglob("*.yaml"))
    
    # Pattern to find Jinja2 variable references
    var_pattern = re.compile(r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)')
    
    for yaml_file in yaml_files:
        try:
            content = yaml_file.read_text()
            parsed = yaml.safe_load(content)
        except Exception:
            continue
        
        rel_path = str(yaml_file)
        
        # Check for defaults
        if "defaults" in str(yaml_file) and isinstance(parsed, dict):
            for key, value in parsed.items():
                result["defaults"][key] = {
                    "value": value,
                    "file": rel_path,
                }
        
        # Check for vars
        if "vars" in str(yaml_file) and isinstance(parsed, dict):
            for key, value in parsed.items():
                result["definitions"][key] = {
                    "value": value,
                    "file": rel_path,
                }
        
        # Find variable usage in file
        for match in var_pattern.finditer(content):
            var_name = match.group(1)
            result["usage"].append({
                "variable": var_name,
                "file": rel_path,
            })
    
    # Deduplicate usage
    seen = set()
    unique_usage = []
    for item in result["usage"]:
        key = (item["variable"], item["file"])
        if key not in seen:
            seen.add(key)
            unique_usage.append(item)
    result["usage"] = unique_usage
    
    return result
