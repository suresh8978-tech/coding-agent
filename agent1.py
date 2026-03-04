#!/usr/bin/env python3
"""
Coding Agent with Ansible and Python capabilities.

A LangGraph-based agent that can analyze and modify Ansible and Python codebases
with an approval-based workflow for all modifications.

Includes fixes for:
- Token overflow (tool result capping, improved summarization)
- Infinite retry loops (consecutive error tracking)
- write_file errors (size limits in system prompt)
- Runaway loops (recursion_limit reduced to 100)

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
import time
from functools import wraps
from typing import Annotated, Any, Literal, Optional, TypedDict
import subprocess

from dotenv import load_dotenv
from langchain_litellm import ChatLiteLLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, RemoveMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# Import all tools
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
    parse_ansible_log,
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
repo_path = os.getenv("REPO_PATH")
os.chdir(repo_path)

# Configure logger
logger = logging.getLogger(__name__)

def setup_logging():
    agent_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(agent_dir, 'agent.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_file_path,
        filemode='a'
    )


# =============================================================================
# FIX 1: Tool Result Capping — prevents single large tool results from
# blowing up the context window
# =============================================================================

MAX_TOOL_RESULT_CHARS = 15000  # ~3,750 tokens per tool result


def cap_tool_result(msg, max_chars=MAX_TOOL_RESULT_CHARS):
    """Truncate an oversized tool result message."""
    if not isinstance(msg, ToolMessage):
        return msg
    if not isinstance(msg.content, str) or len(msg.content) <= max_chars:
        return msg

    original_len = len(msg.content)
    head_size = max_chars - 500
    tail_size = 300

    truncated_content = (
        msg.content[:head_size]
        + f"\n\n... [TRUNCATED: Original was {original_len:,} chars. "
        + "Use read_file with start_line/end_line for specific sections.] ...\n"
        + msg.content[-tail_size:]
    )

    return ToolMessage(
        content=truncated_content,
        tool_call_id=msg.tool_call_id,
        name=getattr(msg, 'name', None),
    )


# =============================================================================
# Agent State Definition
# =============================================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    pending_changes: list[dict[str, Any]]
    awaiting_modification_approval: bool
    awaiting_push_approval: bool
    modification_approved: bool
    push_approved: bool
    pending_push_call: dict | None
    current_branch: str | None
    original_branch: str | None
    branch_created: bool
    mop_content: dict | None
    agent_md_content: str | None
    repo_path: str | None
    mop_path: str | None
    user_feedback: str | None
    summary: str | None
    non_interactive: bool
    error: Optional[str]
    consecutive_tool_errors: int


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
6. **Ansible Log Analysis**: Parse and analyze Ansible execution logs to find failed tasks, affected hosts, and execution summaries using `parse_ansible_log`
7. **Python Coding**: Modify Python code, add imports, add functions
8. **Ansible Coding**: Modify tasks, add tasks, update variables, modify YAML files

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

## Large File Handling (CRITICAL):
Files with more than 200 lines MUST be read and written in chunks — never in a single operation.

### Reading large files:
- The `read_file` tool automatically returns the first 200-line chunk with metadata when a large file is detected
- The metadata header tells you: total lines, current chunk number, total chunks, and the exact `read_file` call needed for the next chunk
- **Always read ALL chunks before drawing conclusions** about a large file's content or making modifications
- After each chunk, immediately call `read_file` again with the indicated `start_line` and `end_line` for the next chunk until `[END OF FILE]` is shown
- Summarize findings incrementally as you process each chunk; do not wait until all chunks are done to report progress

### Forbidden Commands:
- **NEVER** use `run_shell_command` with `grep`, `cat`, `head`, `tail`, `sed`, `awk`, or similar to read file content.
- **ALWAYS** use the `read_file` tool. It is the ONLY allowed way to read files.
- You may use `grep` ONLY for searching for file names (e.g. `find . -name ...`) or checking for existence of a pattern across MANY files (e.g. `grep -l "pattern" -R .`), but NOT for reading content or analyzing code structure of a specific file.

### Writing / editing large files:
The `write_file` tool supports three modes for chunked edits:
- **`mode='write'`** (default): creates or fully overwrites the file — use for the **first chunk only** or for small files (≤200 lines)
- **`mode='append'`**: appends content to the end of the file — use for **every subsequent chunk** after the first
- **`mode='patch'`**: replaces only lines `start_line`..`end_line` (1-based, inclusive) — use to **surgically edit a section** of a large file without rewriting it entirely

Rules:
- For new/rewritten files >200 lines: write the first 200-line chunk with `mode='write'`, then each additional chunk with `mode='append'`
- For targeted edits to existing large files: prefer `mode='patch'` with the exact line range instead of rewriting the whole file
- Never pass more than ~200 lines of content in a single `write_file` call

## Writing Files (CRITICAL):
- NEVER write files larger than 500 lines in a single write_file call
- If you need to create a large document, write it in sections using mode='append'
- If write_file fails, do NOT retry with the same content — try a different approach
- If a tool call fails, do NOT retry the same call more than once — try a different approach or ask the user

## Result Filtering (CRITICAL):
When dealing with large result sets:
- If a directory listing returns more than 30 items, focus on the most relevant ones
- If a search returns more than 20 matches, summarize patterns and show only key examples
- NEVER try to process all items when there are too many - filter and prioritize
- When results are filtered, clearly state what was included and what was skipped
- For large codebases, work incrementally: analyze structure first, then dive into specifics

Remember: NEVER apply changes without explicit user approval!"""


# =============================================================================
# Tool Definitions
# =============================================================================

ALL_TOOLS = [
    read_file, write_file, list_directory, file_exists,
    git_fetch_all, git_create_branch, git_checkout, git_add,
    git_commit, git_push, git_diff, git_status,
    read_mop_document,
    analyze_python_file, find_python_pattern, find_functions, find_classes, find_imports,
    modify_python_code, add_import, add_function,
    scan_ansible_project, analyze_playbook, analyze_role,
    find_tasks_using_module, get_variable_usage, parse_ansible_log,
    modify_task, add_task, modify_variable, modify_yaml_file,
    run_shell_command, find_files, search_in_files,
]


# =============================================================================
# Agent Node Functions
# =============================================================================

def create_agent(model_name: str | None = None):
    llm_name = model_name or os.getenv("LLM_NAME", "anthropic/bedrock-sonnet-4.6")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    api_url = os.getenv("ANTHROPIC_API_URL")
    llm_kwargs = {
        "model": llm_name, "api_key": api_key,
        "max_tokens": 4096, "drop_params": True,
    }
    if api_url:
        llm_kwargs["api_base"] = api_url
    llm = ChatLiteLLM(**llm_kwargs)
    return llm.bind_tools(ALL_TOOLS)


def _sanitize_messages(messages: list) -> list:
    """Remove orphaned ToolMessages that have no matching tool_use in the preceding AIMessage."""
    sanitized = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            preceding_ai = None
            for j in range(len(sanitized) - 1, -1, -1):
                if isinstance(sanitized[j], AIMessage):
                    preceding_ai = sanitized[j]
                    break
            if preceding_ai and hasattr(preceding_ai, 'tool_calls') and preceding_ai.tool_calls:
                tool_ids = {tc['id'] for tc in preceding_ai.tool_calls}
                if msg.tool_call_id not in tool_ids:
                    logger.warning(f"Dropping orphaned ToolMessage with tool_call_id={msg.tool_call_id}")
                    continue
            elif preceding_ai is None:
                logger.warning(f"Dropping orphaned ToolMessage (no preceding AIMessage) with tool_call_id={msg.tool_call_id}")
                continue
        sanitized.append(msg)
    return sanitized


def agent_node(state: AgentState) -> dict:
    messages = state["messages"]

    if not messages or not isinstance(messages[0], SystemMessage):
        system_content = SYSTEM_PROMPT
        agent_md_content = state.get("agent_md_content")
        if agent_md_content:
            system_content += build_agent_md_context(agent_md_content)
        mop_content = state.get("mop_content")
        if mop_content:
            system_content += build_context_message(mop_content)
        summary = state.get("summary")
        if summary:
            system_content += f"\n\n[SYSTEM: Conversation Summary of earlier messages:]\n{summary}\n"
        messages = [SystemMessage(content=system_content)] + list(messages)

    if state.get("awaiting_modification_approval"):
        pending = state.get("pending_changes", [])
        if pending:
            context = "\n\n[SYSTEM: You have pending changes awaiting user approval. "
            context += "Wait for the user to approve, reject, or request modifications.]\n"
            context += format_changes_for_display(pending)
            messages = messages + [HumanMessage(content=context)]

    if state.get("awaiting_push_approval"):
        branch = state.get("current_branch", "unknown")
        context = f"\n\n[SYSTEM: Changes have been applied. Awaiting push approval for branch '{branch}'.]\n"
        messages = messages + [HumanMessage(content=context)]

    if state.get("non_interactive"):
        hint = (
            "\n\n[SYSTEM: You are running in non-interactive mode (single query). "
            "You MUST be exhaustive: use multiple tools and search strategies before "
            "concluding that something does not exist. Do NOT give up after one or two tool calls.]\n"
        )
        messages = messages + [HumanMessage(content=hint)]

    # Break retry loops
    max_consecutive_errors = 3
    error_count = state.get("consecutive_tool_errors", 0)
    if error_count >= max_consecutive_errors:
        logger.warning(f"Breaking retry loop after {error_count} consecutive tool errors")
        return {
            "messages": [AIMessage(content=(
                f"I've encountered repeated tool errors ({error_count} consecutive failures). "
                "To avoid an infinite loop I'm stopping here. "
                "Please check the error details above and try rephrasing your request."
            ))],
            "consecutive_tool_errors": 0,
        }

    messages = _sanitize_messages(messages)

    agent = create_agent()
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = agent.invoke(messages)
            break
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                if attempt < max_retries - 1:
                    wait = 60 * (2 ** attempt)
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}), waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
            else:
                raise

    return {"messages": [response]}


def setup_node(state: AgentState) -> dict:
    repo_path = state.get("repo_path")
    if repo_path:
        if os.path.abspath(repo_path) != os.getcwd():
            try:
                os.chdir(repo_path)
                logger.info(f"Working directory set to: {repo_path}")
            except Exception as e:
                logger.error(f"Error changing directory to {repo_path}: {e}")

    updates = {}
    if not state.get("agent_md_content"):
        agent_md_content = load_agent_md(repo_path)
        if agent_md_content:
            updates["agent_md_content"] = agent_md_content

    mop_path = state.get("mop_path")
    if mop_path and not state.get("mop_content"):
        try:
            mop_content = load_mop_content(mop_path)
            if mop_content:
                updates["mop_content"] = mop_content
                logger.info(f"Loaded MOP content from {mop_path}")
        except Exception as e:
            logger.error(f"Failed to load MOP from {mop_path}: {e}")

    return updates


def should_continue(state: AgentState) -> Literal["tools", "approval_check", "push_approval", "end"]:
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    if state.get("awaiting_modification_approval"):
        return "approval_check"
    return "end"


def should_continue_after_push_approval(state: AgentState) -> Literal["execute_push", "agent"]:
    return "execute_push" if state.get("push_approved") else "agent"


def approval_check_node(state: AgentState) -> dict:
    last_user_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content.lower().strip()
            break
    if not last_user_message:
        return {}

    if state.get("awaiting_modification_approval"):
        if last_user_message in ["approve", "yes", "proceed", "ok", "go ahead", "lgtm"]:
            return {"modification_approved": True, "awaiting_modification_approval": False}
        elif last_user_message in ["reject", "no", "cancel", "abort"]:
            return {"modification_approved": False, "awaiting_modification_approval": False, "pending_changes": []}
        else:
            return {"user_feedback": last_user_message}

    if state.get("awaiting_push_approval"):
        if last_user_message in ["push", "yes", "proceed", "ok", "go ahead"]:
            return {"push_approved": True, "awaiting_push_approval": False}
        elif last_user_message in ["cancel", "no", "skip", "abort"]:
            return {"push_approved": False, "awaiting_push_approval": False}
        else:
            return {"user_feedback": last_user_message}

    return {}


def tools_node(state: AgentState) -> dict:
    """Execute tool calls with capping, error tracking, and push handling."""
    messages = state["messages"]
    last_message = messages[-1]

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}

    logger.info("-" * 60)
    logger.info("TOOL EXECUTION")
    logger.info("-" * 60)

    for tc in last_message.tool_calls:
        logger.info(f"Tool: {tc['name']}")
        logger.info(f"  Input: {_truncate_str(str(tc.get('args', {})), 200)}")

    # Separate git_push
    allowed_tool_calls = []
    blocked_push_call = None
    blocked_messages = []

    for tc in last_message.tool_calls:
        if tc["name"] == "git_push" and not state.get("push_approved"):
            logger.info("git_push requires approval, prompting user")
            args = tc.get("args", {})
            remote = args.get("remote", "origin")
            branch = args.get("branch", "")

            print(f"\n{'='*50}")
            print(f"PUSH APPROVAL REQUIRED")
            print(f"Remote: {remote}")
            print(f"Branch: {branch or 'current (HEAD)'}")
            print(f"{'='*50}")

            try:
                approval = input("Push to remote? (yes/no): ").strip().lower()
            except EOFError:
                approval = "no"

            if approval in ["yes", "y", "push", "ok", "proceed"]:
                rp = os.environ.get("REPO_PATH", ".")
                try:
                    cmd = ["git", "push", remote, branch] if branch else ["git", "push", "-u", remote, "HEAD"]
                    result = subprocess.run(cmd, cwd=rp, capture_output=True, text=True, timeout=60)
                    output = result.stdout.strip() or result.stderr.strip()
                    push_msg = f"Successfully pushed to {remote}.\n{output}" if result.returncode == 0 and output else (
                        f"Successfully pushed to {remote}." if result.returncode == 0 else f"Error pushing: {output}"
                    )
                except subprocess.TimeoutExpired:
                    push_msg = "Error pushing: Git push timed out."
                except Exception as e:
                    push_msg = f"Error pushing: {str(e)}"
                logger.info(f"Push result: {push_msg}")
                print(f"\n{push_msg}\n")
            else:
                push_msg = "Push cancelled by user."
                print(f"\n{push_msg}\n")

            blocked_messages.append(ToolMessage(tool_call_id=tc["id"], content=push_msg, name=tc["name"]))
        else:
            allowed_tool_calls.append(tc)

    # Execute allowed tools
    result_messages = []
    if allowed_tool_calls:
        filtered_message = AIMessage(
            content=last_message.content,
            tool_calls=allowed_tool_calls,
            id=last_message.id,
            additional_kwargs=last_message.additional_kwargs,
            response_metadata=last_message.response_metadata
        )
        temp_messages = list(messages)
        temp_messages[-1] = filtered_message
        temp_state = dict(state)
        temp_state["messages"] = temp_messages
        tool_node = ToolNode(ALL_TOOLS, handle_tool_errors=True)
        node_result = tool_node.invoke(temp_state)
        result_messages = node_result.get("messages", [])

    # =========================================================================
    # FIX 1+3: Cap oversized tool results BEFORE they enter message history
    # =========================================================================
    capped_result_messages = []
    for msg in result_messages:
        if isinstance(msg, ToolMessage):
            msg = cap_tool_result(msg)
        capped_result_messages.append(msg)
    result_messages = capped_result_messages

    all_result_messages = result_messages + blocked_messages

    # Detect errors
    tool_errors: list[str] = []
    for msg in all_result_messages:
        if isinstance(msg, ToolMessage):
            tool_name = "unknown"
            for tc in last_message.tool_calls:
                if tc["id"] == msg.tool_call_id:
                    tool_name = tc["name"]
                    break
            output = msg.content if isinstance(msg.content, str) else str(msg.content)
            logger.info(f"Output from {tool_name}:")
            logger.info(f"  {_truncate_str(output, 500)}")

            _ERROR_PREFIXES = (
                "Error in tool", "Error invoking tool", "Error:",
                "Error reading file:", "Error writing file:",
                "Error listing directory:", "Error checking path:",
                "Error fetching:", "Error creating branch:",
                "Error switching branch:", "Error staging files:",
                "Error committing:", "Error pushing:",
                "Error getting diff:", "Error getting status:",
                "Error getting current branch:", "Error executing command:",
                "Error searching",
            )
            if not output.startswith("[BLOCKED]") and output.startswith(_ERROR_PREFIXES):
                if tool_name != "write_file":
                    logger.warning(f"Tool error from '{tool_name}': {output[:200]}")
                    tool_errors.append(f"**{tool_name}**: {output}")

    # FIX 6: Capped error injection
    if tool_errors:
        error_summary = "\n".join(tool_errors)
        if len(error_summary) > 2000:
            error_summary = error_summary[:2000] + "\n... (additional errors truncated)"
        all_result_messages.append(HumanMessage(content=(
            "[SYSTEM: Tool error occurred. Report it to the user and try a DIFFERENT approach. "
            "Do NOT retry the same tool call.]\n\n" + error_summary
        )))

    logger.info("-" * 60)

    updates: dict[str, Any] = {"messages": all_result_messages}

    if tool_errors:
        updates["consecutive_tool_errors"] = state.get("consecutive_tool_errors", 0) + 1
    else:
        updates["consecutive_tool_errors"] = 0

    if blocked_push_call:
        updates["pending_push_call"] = blocked_push_call
        updates["awaiting_push_approval"] = True

    if state.get("push_approved"):
        for tc in allowed_tool_calls:
            if tc["name"] == "git_push":
                updates["push_approved"] = False
                updates["pending_push_call"] = None
                break

    modification_tools = {
        "modify_python_code", "add_import", "add_function",
        "modify_task", "add_task", "modify_variable", "modify_yaml_file"
    }
    pending_changes = list(state.get("pending_changes", []))
    for msg in result_messages:
        if isinstance(msg, ToolMessage):
            tool_name = None
            for tc in allowed_tool_calls:
                if tc["id"] == msg.tool_call_id:
                    tool_name = tc["name"]
                    break
            if tool_name in modification_tools:
                try:
                    import json
                    tool_result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    if isinstance(tool_result, dict) and "diff" in tool_result:
                        pending_changes.append(tool_result)
                except Exception:
                    pass

    if pending_changes != state.get("pending_changes", []):
        updates["pending_changes"] = pending_changes
        updates["awaiting_modification_approval"] = True

    return updates


def push_approval_node(state: AgentState) -> dict:
    pending_push = state.get("pending_push_call")
    branch = state.get("current_branch", "unknown")
    push_request = format_push_request(branch, 1, [])
    logger.info("Push approval node triggered - calling interrupt")
    user_response = interrupt(push_request)
    logger.info(f"Interrupt resumed with user response: {user_response}")
    response_lower = str(user_response).lower().strip()

    if response_lower in ["push", "yes", "proceed", "ok", "go ahead", "y"]:
        return {"push_approved": True, "awaiting_push_approval": False}
    elif response_lower in ["cancel", "no", "skip", "abort", "n"]:
        return {
            "push_approved": False, "awaiting_push_approval": False, "pending_push_call": None,
            "messages": [AIMessage(content="Push cancelled by user.")]
        }
    else:
        return {
            "push_approved": False, "awaiting_push_approval": False, "pending_push_call": None,
            "user_feedback": user_response,
            "messages": [AIMessage(content=f"Push deferred. User feedback: {user_response}")]
        }


def execute_push_node(state: AgentState) -> dict:
    pending_push = state.get("pending_push_call")
    if not pending_push:
        return {"messages": [AIMessage(content="No pending push to execute.")],
                "push_approved": False, "pending_push_call": None, "awaiting_push_approval": False}

    args = pending_push.get("args", {})
    remote = args.get("remote", "origin")
    branch = args.get("branch", "")
    rp = os.environ.get("REPO_PATH", ".")
    try:
        cmd = ["git", "push", remote, branch] if branch else ["git", "push", "-u", remote, "HEAD"]
        result = subprocess.run(cmd, cwd=rp, capture_output=True, text=True, timeout=60)
        output = result.stdout.strip() or result.stderr.strip()
        msg = f"Successfully pushed to {remote}.\n{output}" if result.returncode == 0 else f"Error pushing: {output}"
    except subprocess.TimeoutExpired:
        msg = "Error pushing: Git push timed out."
    except Exception as e:
        msg = f"Error pushing: {str(e)}"

    logger.info(f"git_push result: {msg}")
    return {"messages": [AIMessage(content=msg)],
            "push_approved": False, "pending_push_call": None, "awaiting_push_approval": False}


def _truncate_str(s: str, max_len: int) -> str:
    return s if len(s) <= max_len else s[:max_len] + "..."


# =============================================================================
# FIX 2: Improved summarize_node
# =============================================================================

def summarize_node(state: AgentState) -> dict:
    """Summarize older messages if the context gets too long."""
    messages = state["messages"]

    # Cap any oversized tool results still in history
    capped_messages = []
    changed = False
    for m in messages:
        if isinstance(m, ToolMessage) and isinstance(m.content, str) and len(m.content) > MAX_TOOL_RESULT_CHARS:
            capped_messages.append(cap_tool_result(m))
            changed = True
        else:
            capped_messages.append(m)
    if changed:
        messages = capped_messages

    total_tokens = sum(
        len(str(m.content)) // 4
        for m in messages
        if hasattr(m, 'content') and isinstance(m.content, str)
    )

    # FIX: Raised from 10000 to 80000
    TOKEN_LIMIT = 80000

    if total_tokens > TOKEN_LIMIT and len(messages) > 10:
        keep_at_least = 10  # FIX: was 6
        idx = len(messages) - keep_at_least

        while idx > 0:
            prev_msg = messages[idx - 1]
            curr_msg = messages[idx]
            is_prev_ai_with_tools = isinstance(prev_msg, AIMessage) and getattr(prev_msg, "tool_calls", None)
            is_curr_tool = isinstance(curr_msg, ToolMessage)
            if is_prev_ai_with_tools or is_curr_tool:
                idx -= 1
            else:
                break

        if idx > 0:
            msgs_to_summarize = messages[0:idx]

            summary_prompt = "Summarize this conversation history concisely. Focus on key decisions, files analyzed, and tasks completed:\n\n"
            for m in msgs_to_summarize:
                role = m.type
                content = str(m.content)
                if len(content) > 300:
                    content = content[:300] + "..."
                summary_prompt += f"{role}: {content}\n"

            # FIX: Cap the summary prompt
            if len(summary_prompt) > 15000:
                summary_prompt = summary_prompt[:15000] + "\n... (truncated)"

            summary_messages = [
                SystemMessage(content="Summarize this conversation concisely in 200 words or less."),
                HumanMessage(content=summary_prompt)
            ]

            llm_name = os.getenv("LLM_NAME", "anthropic/bedrock-sonnet-4.6")
            api_key = os.getenv("ANTHROPIC_API_KEY")
            api_url = os.getenv("ANTHROPIC_API_URL")
            llm_kwargs = {"model": llm_name, "api_key": api_key, "max_tokens": 1024, "drop_params": True}
            if api_url:
                llm_kwargs["api_base"] = api_url
            llm = ChatLiteLLM(**llm_kwargs)

            try:
                response = llm.invoke(summary_messages)
                new_summary_text = response.content
                old_summary = state.get("summary")
                if old_summary:
                    combined = f"{old_summary}\n\n[Update]:\n{new_summary_text}"
                    if len(combined) > 5000:
                        combined = combined[-5000:]
                    new_summary = combined
                else:
                    new_summary = new_summary_text

                removals = [RemoveMessage(id=m.id) for m in msgs_to_summarize if m.id]
                logger.info(f"Summarized {len(msgs_to_summarize)} messages (~{total_tokens} tokens)")
                return {"summary": new_summary, "messages": removals}
            except Exception as e:
                logger.error(f"Error during summarization: {e}")
                new_summary = state.get("summary", "") or ""
                new_summary += f"\n[Auto-trimmed {len(msgs_to_summarize)} old messages]"
                removals = [RemoveMessage(id=m.id) for m in msgs_to_summarize if m.id]
                return {"summary": new_summary, "messages": removals}

    return {}


# =============================================================================
# Exception Handling
# =============================================================================

def catch_exceptions(node_func):
    @wraps(node_func)
    def wrapper(state: AgentState) -> AgentState:
        try:
            result = node_func(state)
            if isinstance(result, dict):
                result.setdefault("error", None)
            return result
        except Exception as exc:
            from langgraph.errors import GraphInterrupt
            if isinstance(exc, GraphInterrupt):
                raise
            logger.error(f"Unhandled exception in {node_func.__name__}: {exc}", exc_info=True)
            return {**state, "error": str(exc)}
    return wrapper


def error_handler_node(state: AgentState) -> dict:
    error_message = state.get("error", "An unexpected error occurred.")
    return {
        **state,
        "messages": state["messages"] + [
            AIMessage(content=f":warning: An error occurred:\n\n```\n{error_message}\n```\n\nPlease try again.")
        ],
        "error": None,
    }


def route_on_error(next_node: str):
    def router(state: AgentState) -> str:
        return "error_handler" if state.get("error") else next_node
    return router


# =============================================================================
# Graph Construction
# =============================================================================

def create_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("setup",          catch_exceptions(setup_node))
    workflow.add_node("agent",          catch_exceptions(agent_node))
    workflow.add_node("tools",          catch_exceptions(tools_node))
    workflow.add_node("approval_check", catch_exceptions(approval_check_node))
    workflow.add_node("summarize",      catch_exceptions(summarize_node))
    workflow.add_node("error_handler",  error_handler_node)

    workflow.set_entry_point("setup")

    workflow.add_conditional_edges("setup", route_on_error("summarize"),
        {"summarize": "summarize", "error_handler": "error_handler"})

    def agent_router(state: AgentState) -> str:
        if state.get("error"):
            return "error_handler"
        return should_continue(state)

    workflow.add_conditional_edges("agent", agent_router,
        {"tools": "tools", "approval_check": "approval_check", "end": END, "error_handler": "error_handler"})

    workflow.add_conditional_edges("tools", route_on_error("summarize"),
        {"summarize": "summarize", "error_handler": "error_handler"})
    workflow.add_conditional_edges("approval_check", route_on_error("summarize"),
        {"summarize": "summarize", "error_handler": "error_handler"})
    workflow.add_conditional_edges("summarize", route_on_error("agent"),
        {"agent": "agent", "error_handler": "error_handler"})

    workflow.add_edge("error_handler", END)
    return workflow.compile(checkpointer=MemorySaver())


# =============================================================================
# Runner Functions
# =============================================================================

def load_mop_content(mop_path):
    if not mop_path:
        return None
    from tools.mop_parser import read_mop_document
    result = read_mop_document.invoke({"path": mop_path})
    if isinstance(result, dict) and "error" in result:
        return None
    return result


def load_agent_md(repo_path=None):
    base = repo_path or os.getcwd()
    path = os.path.join(base, "AGENT.md")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception:
        return None


def create_initial_state(repo_path=None, mop_path=None):
    return {
        "messages": [], "pending_changes": [],
        "awaiting_modification_approval": False, "awaiting_push_approval": False,
        "modification_approved": False, "push_approved": False,
        "pending_push_call": None, "current_branch": None, "original_branch": None,
        "branch_created": False, "mop_content": None, "agent_md_content": None,
        "user_feedback": None, "summary": None,
        "repo_path": repo_path, "mop_path": mop_path,
        "non_interactive": False, "error": None, "consecutive_tool_errors": 0,
    }


def build_agent_md_context(content):
    if not content:
        return ""
    ctx = "\n\n" + "=" * 60 + "\nPROJECT-SPECIFIC INSTRUCTIONS (AGENT.md)\n" + "=" * 60
    ctx += "\nThese instructions MUST be followed. AGENT.md takes precedence.\n"
    ctx += "\n--- AGENT.md ---\n" + content[:15000]
    if len(content) > 15000:
        ctx += "\n... (truncated)"
    ctx += "\n--- END AGENT.md ---\n" + "=" * 60 + "\n"
    return ctx


def build_context_message(mop):
    if not mop:
        return ""
    ctx = f"\n\n[MOP DOCUMENT LOADED]\nTitle: {mop.get('title', 'Untitled')}\n"
    ctx += f"Sections: {mop.get('stats', {}).get('section_count', 0)}\n"
    ctx += f"Tables: {mop.get('stats', {}).get('table_count', 0)}\n"
    ctx += "\n--- MOP FULL CONTENT ---\n" + mop.get("full_text", "")[:30000]
    if len(mop.get("full_text", "")) > 30000:
        ctx += "\n... (truncated)"
    ctx += "\n--- END MOP CONTENT ---\n"
    return ctx


def run_single_query(query, repo_path=None, mop_path=None):
    import uuid
    from langgraph.errors import GraphInterrupt
    setup_logging()
    graph = create_graph()
    state = create_initial_state(repo_path, mop_path)
    state["non_interactive"] = True
    state["messages"] = [HumanMessage(content=query)]
    # FIX 4: Reduced from 10000 to 100
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 100}
    try:
        result = graph.invoke(state, config)
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                return msg.content
        return "No response generated."
    except GraphInterrupt as e:
        return f"PUSH APPROVAL REQUIRED\n\n{e.args[0] if e.args else ''}"


def run_interactive(repo_path=None, mop_path=None):
    import uuid
    from langgraph.errors import GraphInterrupt
    setup_logging()
    graph = create_graph()
    state = create_initial_state(repo_path, mop_path)
    # FIX 4: Reduced from 10000 to 100
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 100}

    print("=" * 60)
    print("Coding Agent with Ansible & Python Capabilities")
    print("=" * 60)
    print("Commands: approve | reject | push | cancel | quit")
    print("=" * 60 + "\n")

    interrupted = False
    first_invocation = True

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            try:
                if interrupted:
                    try:
                        result = graph.invoke(Command(resume=user_input), config)
                        interrupted = False
                    except GraphInterrupt as e:
                        print(f"\n{e.args[0] if e.args else 'Approval required'}")
                        print("\nType 'push' or 'cancel'.\n")
                        continue
                else:
                    if first_invocation:
                        state["messages"] = [HumanMessage(content=user_input)]
                        try:
                            result = graph.invoke(state, config)
                        except GraphInterrupt as e:
                            interrupted = True
                            print(f"\n{e.args[0] if e.args else 'Push approval required'}")
                            print("\nType 'push' or 'cancel'.\n")
                            first_invocation = False
                            continue
                        first_invocation = False
                    else:
                        try:
                            result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)
                        except GraphInterrupt as e:
                            interrupted = True
                            print(f"\n{e.args[0] if e.args else 'Push approval required'}")
                            print("\nType 'push' or 'cancel'.\n")
                            continue

                state = result
                for msg in reversed(state["messages"]):
                    if isinstance(msg, AIMessage) and msg.content:
                        print(f"\nAgent: {msg.content}\n")
                        break

                if state.get("awaiting_modification_approval") and state.get("pending_changes"):
                    print("\n" + format_changes_for_display(state["pending_changes"]))
                    print("\nType 'approve' or describe changes.\n")

            except Exception as e:
                print(f"\nError: {e}\n")
                import traceback
                traceback.print_exc()
                interrupted = False
    except KeyboardInterrupt:
        print("\n\nGoodbye!")


def parse_args():
    parser = argparse.ArgumentParser(description="Coding Agent with Ansible & Python capabilities")
    parser.add_argument("--mop", type=str, help="Path to MOP document")
    parser.add_argument("--query", "-q", type=str, help="Non-interactive query")
    parser.add_argument("--repo", type=str, help="Override REPO_PATH")
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    repo_path = args.repo or os.getenv("REPO_PATH")
    if args.query:
        print(run_single_query(args.query, repo_path, args.mop))
    else:
        run_interactive(repo_path, args.mop)
