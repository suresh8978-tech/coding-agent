"""Tools package for the coding agent."""

from tools.file_ops import read_file, write_file, list_directory, file_exists
from tools.git_ops import (
    git_fetch_all,
    git_create_branch,
    git_checkout,
    git_add,
    git_commit,
    git_push,
    git_diff,
    git_status,
    get_current_branch,
)
from tools.mop_parser import read_mop_document
from tools.python_analysis import (
    analyze_python_file,
    find_python_pattern,
    find_functions,
    find_classes,
    find_imports,
)
from tools.python_coding import modify_python_code, add_import, add_function
from tools.ansible_analysis import (
    scan_ansible_project,
    analyze_playbook,
    analyze_role,
    find_tasks_using_module,
    get_variable_usage,
)
from tools.ansible_coding import modify_task, add_task, modify_variable, modify_yaml_file
from tools.shell_ops import run_shell_command, find_files, search_in_files
from tools.approval import (
    PendingChange,
    create_modification_plan,
    generate_unified_diff,
)

__all__ = [
    # File operations
    "read_file",
    "write_file",
    "list_directory",
    "file_exists",
    # Shell operations
    "run_shell_command",
    "find_files",
    "search_in_files",
    # Git operations
    "git_fetch_all",
    "git_create_branch",
    "git_checkout",
    "git_add",
    "git_commit",
    "git_push",
    "git_diff",
    "git_status",
    "get_current_branch",
    # MOP parsing
    "read_mop_document",
    # Python analysis
    "analyze_python_file",
    "find_python_pattern",
    "find_functions",
    "find_classes",
    "find_imports",
    # Python coding
    "modify_python_code",
    "add_import",
    "add_function",
    # Ansible analysis
    "scan_ansible_project",
    "analyze_playbook",
    "analyze_role",
    "find_tasks_using_module",
    "get_variable_usage",
    # Ansible coding
    "modify_task",
    "add_task",
    "modify_variable",
    "modify_yaml_file",
    # Approval
    "PendingChange",
    "create_modification_plan",
    "generate_unified_diff",
]
