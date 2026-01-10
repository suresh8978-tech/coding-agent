"""Ansible code modification tools."""

from pathlib import Path
from typing import Any
from langchain_core.tools import tool


@tool
def modify_task(path: str, task_name: str, new_spec: dict) -> dict[str, Any]:
    """Modify an existing task in an Ansible playbook or task file.
    
    Args:
        path: Path to the YAML file containing the task.
        task_name: Name of the task to modify.
        new_spec: Dictionary with the new task specification.
        
    Returns:
        Dictionary with original content, modified content, and diff.
    """
    file_path = Path(path)
    if not file_path.exists():
        return {"error": f"File '{path}' does not exist."}
    
    try:
        import yaml
    except ImportError:
        return {"error": "PyYAML not installed."}
    
    try:
        original = file_path.read_text(encoding='utf-8')
        content = yaml.safe_load(original)
    except Exception as e:
        return {"error": f"Error reading file: {str(e)}"}
    
    # Find and modify the task
    modified_content = content
    found = False
    
    def find_and_modify_task(tasks):
        nonlocal found
        if not isinstance(tasks, list):
            return tasks
        
        for i, task in enumerate(tasks):
            if isinstance(task, dict) and task.get("name") == task_name:
                # Merge new spec into existing task
                for key, value in new_spec.items():
                    tasks[i][key] = value
                found = True
                break
        return tasks
    
    # Handle playbook format
    if isinstance(content, list):
        for play in content:
            if isinstance(play, dict):
                if "tasks" in play:
                    play["tasks"] = find_and_modify_task(play["tasks"])
                if "handlers" in play:
                    play["handlers"] = find_and_modify_task(play["handlers"])
    # Handle task file format
    elif isinstance(content, list):
        modified_content = find_and_modify_task(content)
    
    if not found:
        return {"error": f"Task '{task_name}' not found in file."}
    
    # Convert back to YAML
    modified = yaml.dump(modified_content, default_flow_style=False, sort_keys=False)
    
    from tools.approval import generate_unified_diff
    diff = generate_unified_diff(original, modified, path)
    
    return {
        "file": path,
        "original": original,
        "modified": modified,
        "diff": diff,
        "description": f"Modified task '{task_name}'",
    }


@tool
def add_task(path: str, task_spec: dict, after_task: str = "") -> dict[str, Any]:
    """Add a new task to an Ansible playbook or task file.
    
    Args:
        path: Path to the YAML file.
        task_spec: Dictionary with the task specification.
        after_task: Name of task after which to insert. Empty = end.
        
    Returns:
        Dictionary with original content, modified content, and diff.
    """
    file_path = Path(path)
    if not file_path.exists():
        return {"error": f"File '{path}' does not exist."}
    
    try:
        import yaml
    except ImportError:
        return {"error": "PyYAML not installed."}
    
    try:
        original = file_path.read_text(encoding='utf-8')
        content = yaml.safe_load(original)
    except Exception as e:
        return {"error": f"Error reading file: {str(e)}"}
    
    def insert_task(tasks, task, after=""):
        if not isinstance(tasks, list):
            return [task]
        
        if not after:
            tasks.append(task)
        else:
            insert_idx = len(tasks)
            for i, t in enumerate(tasks):
                if isinstance(t, dict) and t.get("name") == after:
                    insert_idx = i + 1
                    break
            tasks.insert(insert_idx, task)
        return tasks
    
    # Handle playbook format (list of plays)
    if isinstance(content, list):
        if content and isinstance(content[0], dict) and "hosts" in content[0]:
            # This is a playbook with plays
            if content[0].get("tasks") is None:
                content[0]["tasks"] = []
            content[0]["tasks"] = insert_task(content[0]["tasks"], task_spec, after_task)
        else:
            # This is a task file (list of tasks)
            content = insert_task(content, task_spec, after_task)
    else:
        return {"error": "Unexpected file format."}
    
    modified = yaml.dump(content, default_flow_style=False, sort_keys=False)
    
    from tools.approval import generate_unified_diff
    diff = generate_unified_diff(original, modified, path)
    
    return {
        "file": path,
        "original": original,
        "modified": modified,
        "diff": diff,
        "description": f"Added task '{task_spec.get('name', 'Unnamed')}'",
    }


@tool
def modify_variable(path: str, var_name: str, new_value: Any) -> dict[str, Any]:
    """Modify a variable in an Ansible vars or defaults file.
    
    Args:
        path: Path to the YAML file containing variables.
        var_name: Name of the variable to modify.
        new_value: New value for the variable.
        
    Returns:
        Dictionary with original content, modified content, and diff.
    """
    file_path = Path(path)
    if not file_path.exists():
        return {"error": f"File '{path}' does not exist."}
    
    try:
        import yaml
    except ImportError:
        return {"error": "PyYAML not installed."}
    
    try:
        original = file_path.read_text(encoding='utf-8')
        content = yaml.safe_load(original) or {}
    except Exception as e:
        return {"error": f"Error reading file: {str(e)}"}
    
    if not isinstance(content, dict):
        return {"error": "File does not contain a variable dictionary."}
    
    old_value = content.get(var_name)
    content[var_name] = new_value
    
    modified = yaml.dump(content, default_flow_style=False, sort_keys=False)
    
    from tools.approval import generate_unified_diff
    diff = generate_unified_diff(original, modified, path)
    
    return {
        "file": path,
        "original": original,
        "modified": modified,
        "diff": diff,
        "old_value": old_value,
        "new_value": new_value,
        "description": f"Modified variable '{var_name}': {old_value} -> {new_value}",
    }


@tool
def modify_yaml_file(path: str, modifications: dict) -> dict[str, Any]:
    """Make arbitrary modifications to a YAML file.
    
    Args:
        path: Path to the YAML file.
        modifications: Dictionary of key paths and values to set.
                      Use dot notation for nested keys (e.g., "play.vars.my_var").
        
    Returns:
        Dictionary with original content, modified content, and diff.
    """
    file_path = Path(path)
    if not file_path.exists():
        return {"error": f"File '{path}' does not exist."}
    
    try:
        import yaml
    except ImportError:
        return {"error": "PyYAML not installed."}
    
    try:
        original = file_path.read_text(encoding='utf-8')
        content = yaml.safe_load(original)
    except Exception as e:
        return {"error": f"Error reading file: {str(e)}"}
    
    def set_nested_value(obj, key_path, value):
        """Set a value in a nested dictionary using dot notation."""
        keys = key_path.split(".")
        current = obj
        for key in keys[:-1]:
            if key.isdigit():
                key = int(key)
            if isinstance(current, list) and isinstance(key, int):
                current = current[key]
            elif isinstance(current, dict):
                if key not in current:
                    current[key] = {}
                current = current[key]
            else:
                raise KeyError(f"Cannot navigate to {key_path}")
        
        final_key = keys[-1]
        if final_key.isdigit():
            final_key = int(final_key)
        
        if isinstance(current, list):
            current[final_key] = value
        else:
            current[final_key] = value
    
    try:
        for key_path, value in modifications.items():
            set_nested_value(content, key_path, value)
    except Exception as e:
        return {"error": f"Error applying modifications: {str(e)}"}
    
    modified = yaml.dump(content, default_flow_style=False, sort_keys=False)
    
    from tools.approval import generate_unified_diff
    diff = generate_unified_diff(original, modified, path)
    
    return {
        "file": path,
        "original": original,
        "modified": modified,
        "diff": diff,
        "description": f"Modified YAML file with {len(modifications)} change(s)",
    }
