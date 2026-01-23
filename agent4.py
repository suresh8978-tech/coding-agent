#!/usr/bin/env python3
"""
Coding Agent with Ansible and Python capabilities.

A LangGraph-based agent that can analyze and modify Ansible and Python codebases
with an approval-based workflow for all modifications.

Usage:
    python agent.py                          # Interactive mode
    python agent.py --query "your question"  # Non-interactive mode
    python agent.py --mop path/to/mop.docx   # Load MOP document
    python agent.py --mop mop.docx --query "implement step 1"
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Import all tools
from tools.file_ops import read_file, write_file, list_directory, delete_file, file_exists
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
    format_changes_for_display,
    format_push_request,
)

# Load environment variables
load_dotenv()

# =============================================================================
# Logging setup
# =============================================================================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("coding-agent")


# =============================================================================
# Agent State Definition
# =============================================================================

class AgentState(TypedDict):
    """State for the coding agent."""
    # Conversation messages
    messages: Annotated[list, add_messages]

    # Pending changes awaiting approval
    pending_changes: list[dict[str, Any]]

    # Approval flags
    awaiting_modification_approval: bool
    awaiting_push_approval: bool
    modification_approved: bool
    push_approved: bool

    # Git state
    current_branch: str | None
    original_branch: str | None
    branch_created: bool

    # MOP content (if loaded)
    mop_content: dict | None

    # User feedback for revisions
    user_feedback: str | None

    # Repo/workdir context
    repo_path: str | None
    cwd_before: str | None
    cwd_after: str | None


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are an intelligent coding agent specialized in Ansible and Python development.

## Your Capabilities:
1. **File Operations**: Read, write, list, and delete files
2. **Git Operations**: Fetch, create branches (with 'agent-' prefix), commit, push, diff, status
3. **MOP Analysis**: Read and parse Method of Procedure documents (DOCX format, up to 90 pages)
4. **Python Analysis**: Analyze Python code structure using ast-grep (functions, classes, imports, patterns)
5. **Ansible Analysis**: Analyze Ansible projects, playbooks, roles, tasks, and variables
6. **Python Coding**: Modify Python code, add imports, add functions
7. **Ansible Coding**: Modify tasks, add tasks, update variables, modify YAML files

## Workflow Rules (CRITICAL):

### Before Making ANY Code Modifications:
1. ALWAYS run `git_fetch_all` first to ensure you have the latest code
2. Create a new branch using `git_create_branch` - provide a descriptive name (it will be auto-prefixed with 'agent-')
3. The branch name should describe what changes you're making

### Modification Approval Process:
1. When you want to modify code, use the appropriate coding tools (modify_python_code, add_function, modify_task, etc.)
2. These tools return a diff showing the proposed changes - DO NOT apply them yet
3. Present the modification plan with ALL diffs to the user and wait for approval
4. If the user requests changes, incorporate their feedback and present the updated plan
5. Only after explicit approval ("approve", "yes", "proceed", etc.), apply the changes using write_file

### Push Approval Process:
1. After changes are applied and committed, ask for push approval
2. Show the branch name, commit summary, and files changed
3. If the user requests changes to the commit, amend and ask again
4. Only after explicit push approval, execute git_push

### When Reading MOPs:
1. Use read_mop_document to load the entire document
2. Analyze all procedures, steps, and requirements
3. Create a comprehensive modification plan covering ALL changes from the MOP
4. Present the full plan and wait for approval before making any changes

## Response Format:
- Be clear and concise
- Always show diffs when proposing changes
- Explain what each change does and why
- Group related changes together
- Number the changes for easy reference
- If no specific tool exists for a task, use run_shell_command to execute shell commands

## MOP Document Handling:
When a MOP (Method of Procedure) document is loaded:
- Prioritize answering questions based on the MOP content
- Reference specific sections, steps, or procedures from the MOP
- If asked to implement changes, follow the MOP procedures exactly
- Cite the MOP section when explaining actions

## Result Filtering (CRITICAL):
When dealing with large result sets:
- If a directory listing returns more than 30 items, focus on the most relevant ones
- If a search returns more than 20 matches, summarize patterns and show only key examples
- If file content is too long (>500 lines), analyze in sections
- NEVER try to process all items when there are too many - filter and prioritize
- When results are filtered, clearly state what was included and what was skipped
- For large codebases, work incrementally: analyze structure first, then dive into specifics

Remember: NEVER apply changes without explicit user approval!"""


# =============================================================================
# Tool Definitions
# =============================================================================

ALL_TOOLS = [
    # File operations
    read_file,
    write_file,
    list_directory,
    delete_file,
    file_exists,
    # Git operations
    git_fetch_all,
    git_create_branch,
    git_checkout,
    git_add,
    git_commit,
    git_push,
    git_diff,
    git_status,
    # MOP parsing
    read_mop_document,
    # Python analysis
    analyze_python_file,
    find_python_pattern,
    find_functions,
    find_classes,
    find_imports,
    # Python coding
    modify_python_code,
    add_import,
    add_function,
    # Ansible analysis
    scan_ansible_project,
    analyze_playbook,
    analyze_role,
    find_tasks_using_module,
    get_variable_usage,
    # Ansible coding
    modify_task,
    add_task,
    modify_variable,
    modify_yaml_file,
    # Shell operations
    run_shell_command,
    find_files,
    search_in_files,
]


# =============================================================================
# Graph Nodes
# =============================================================================

def setup_node(state: AgentState) -> dict:
    """
    Set working directory for the agent run.
    Keep ALL directory changes inside the graph (not __main__).
    """
    repo_path = state.get("repo_path")

    cwd_before = os.getcwd()
    logger.info("Current working directory (before): %s", cwd_before)

    if repo_path:
        p = Path(repo_path).expanduser().resolve()
        if not p.exists() or not p.is_dir():
            logger.warning("REPO_PATH does not exist or is not a directory: %s", str(p))
        else:
            os.chdir(str(p))
            logger.info("Changed working directory to: %s", str(p))

    cwd_after = os.getcwd()
    logger.info("Current working directory (after): %s", cwd_after)

    return {
        "cwd_before": cwd_before,
        "cwd_after": cwd_after,
    }


def create_agent(model_name: str | None = None):
    """Create the LLM agent with tools bound."""
    llm_name = model_name or os.getenv("LLM_NAME", "claude-3-5-haiku-latest")
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    llm = ChatAnthropic(
        model=llm_name,
        api_key=api_key,
        max_tokens=8192,
    )

    return llm.bind_tools(ALL_TOOLS)


def agent_node(state: AgentState) -> dict:
    """Main agent node that processes messages and decides actions."""
    messages = state["messages"]

    # Add system prompt if not present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    # Check if we're awaiting approval
    if state.get("awaiting_modification_approval"):
        pending = state.get("pending_changes", [])
        if pending:
            context = "\n\n[SYSTEM: You have pending changes awaiting user approval. "
            context += "Wait for the user to approve, reject, or request modifications.]\n"
            context += format_changes_for_display(pending)
            messages = messages + [SystemMessage(content=context)]

    if state.get("awaiting_push_approval"):
        branch = state.get("current_branch", "unknown")
        context = f"\n\n[SYSTEM: Changes have been applied. Awaiting push approval for branch '{branch}'.]\n"
        messages = messages + [SystemMessage(content=context)]

    agent = create_agent()
    response = agent.invoke(messages)

    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "approval_check", "end"]:
    """Determine the next step based on current state."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    if state.get("awaiting_modification_approval") or state.get("awaiting_push_approval"):
        return "approval_check"

    return "end"


def approval_check_node(state: AgentState) -> dict:
    """Handle approval checking and user feedback processing."""
    messages = state["messages"]
    last_user_message = None

    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content.lower().strip()
            break

    if not last_user_message:
        return {}

    if state.get("awaiting_modification_approval"):
        if last_user_message in ["approve", "yes", "proceed", "ok", "go ahead", "lgtm"]:
            return {
                "modification_approved": True,
                "awaiting_modification_approval": False,
            }
        elif last_user_message in ["reject", "no", "cancel", "abort"]:
            return {
                "modification_approved": False,
                "awaiting_modification_approval": False,
                "pending_changes": [],
            }
        else:
            return {"user_feedback": last_user_message}

    if state.get("awaiting_push_approval"):
        if last_user_message in ["push", "yes", "proceed", "ok", "go ahead"]:
            return {
                "push_approved": True,
                "awaiting_push_approval": False,
            }
        elif last_user_message in ["cancel", "no", "skip", "abort"]:
            return {
                "push_approved": False,
                "awaiting_push_approval": False,
            }
        else:
            return {"user_feedback": last_user_message}

    return {}


def _truncate_str(s: str, max_len: int) -> str:
    """Truncate a string to max_len characters."""
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


def tools_node(state: AgentState) -> dict:
    """Execute tool calls and process results with verbose logging."""
    messages = state["messages"]
    last_message = messages[-1]

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}

    logger.info("-" * 60)
    logger.info("TOOL EXECUTION")
    logger.info("-" * 60)

    for tc in last_message.tool_calls:
        logger.info("Tool: %s", tc["name"])
        logger.info("  Input: %s", _truncate_str(str(tc.get("args", {})), 200))

    tool_node = ToolNode(ALL_TOOLS)
    result = tool_node.invoke(state)

    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage):
            tool_name = "unknown"
            for tc in last_message.tool_calls:
