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
                if tc["id"] == msg.tool_call_id:
                    tool_name = tc["name"]
                    break

            output = msg.content if isinstance(msg.content, str) else str(msg.content)
            logger.info("Output from %s:", tool_name)
            logger.info("  %s", _truncate_str(output, 500))

    logger.info("-" * 60)

    modification_tools = {
        "modify_python_code", "add_import", "add_function",
        "modify_task", "add_task", "modify_variable", "modify_yaml_file"
    }

    pending_changes = state.get("pending_changes", [])

    for tool_call in last_message.tool_calls:
        if tool_call["name"] in modification_tools:
            for msg in result.get("messages", []):
                if isinstance(msg, ToolMessage) and msg.tool_call_id == tool_call["id"]:
                    try:
                        import json
                        tool_result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                        if isinstance(tool_result, dict) and "diff" in tool_result:
                            pending_changes.append(tool_result)
                    except Exception:
                        pass

    updates = result
    if pending_changes != state.get("pending_changes", []):
        updates["pending_changes"] = pending_changes
        updates["awaiting_modification_approval"] = True

    return updates


# =============================================================================
# Graph Construction
# =============================================================================

def create_graph():
    """Create the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("setup", setup_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("approval_check", approval_check_node)

    workflow.set_entry_point("setup")
    workflow.add_edge("setup", "agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "approval_check": "approval_check",
            "end": END,
        }
    )

    workflow.add_edge("tools", "agent")
    workflow.add_edge("approval_check", "agent")

    return workflow.compile()


# =============================================================================
# Agent Runner Functions
# =============================================================================

def load_mop_content(mop_path: str) -> dict | None:
    """Load MOP document content."""
    if not mop_path:
        return None

    print(f"Loading MOP document: {mop_path}")
    result = read_mop_document.invoke({"path": mop_path})

    if isinstance(result, dict) and "error" in result:
        print(f"Error loading MOP: {result['error']}")
        return None

    print(f"MOP loaded successfully. {result.get('stats', {}).get('word_count', 0)} words.")
    return result


def create_initial_state(mop_content: dict | None = None, repo_path: str | None = None) -> AgentState:
    """Create initial agent state."""
    return {
        "messages": [],
        "pending_changes": [],
        "awaiting_modification_approval": False,
        "awaiting_push_approval": False,
        "modification_approved": False,
        "push_approved": False,
        "current_branch": None,
        "original_branch": None,
        "branch_created": False,
        "mop_content": mop_content,
        "user_feedback": None,
        "repo_path": repo_path,
        "cwd_before": None,
        "cwd_after": None,
    }


def build_context_message(mop_content: dict | None) -> str:
    """Build context message with MOP content if available."""
    if not mop_content:
        return ""

    context = "\n\n[MOP DOCUMENT LOADED]\n"
    context += f"Title: {mop_content.get('title', 'Untitled')}\n"
    context += f"Sections: {mop_content.get('stats', {}).get('section_count', 0)}\n"
    context += f"Tables: {mop_content.get('stats', {}).get('table_count', 0)}\n"
    context += "\n--- MOP FULL CONTENT ---\n"
    context += mop_content.get("full_text", "")[:50000]
    if len(mop_content.get("full_text", "")) > 50000:
        context += "\n... (content truncated)"
    context += "\n--- END MOP CONTENT ---\n"
    context += "\nPrioritize responses based on this MOP document when applicable.\n"

    return context


def run_single_query(query: str, mop_content: dict | None = None, repo_path: str | None = None) -> str:
    """Run a single query in non-interactive mode."""
    graph = create_graph()
    state = create_initial_state(mop_content, repo_path)

    full_query = query
    if mop_content:
        context = build_context_message(mop_content)
        full_query = context + "\n\nUser Query: " + query

    state["messages"] = [HumanMessage(content=full_query)]

    try:
        result = graph.invoke(state, {"recursion_limit": 100})

        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                return msg.content

        return "No response generated."

    except Exception as e:
        error_msg = str(e)
        if "rate_limit" in error_msg.lower():
            return "Error: Rate limit reached. Please wait and try again."
        return f"Error: {e}"
    finally:
        # ✅ Print working directory after execution (success OR failure)
        print(f"\nWorking directory (current): {os.getcwd()}\n")


def run_interactive(mop_content: dict | None = None, repo_path: str | None = None):
    """Run the agent in interactive mode."""
    graph = create_graph()
    state = create_initial_state(mop_content, repo_path)

    print("=" * 60)
    print("Coding Agent with Ansible & Python Capabilities")
    print("=" * 60)
    if mop_content:
        print(f"MOP loaded: {mop_content.get('title', 'Untitled')}")
    print("Commands:")
    print("  - Type your request to interact with the agent")
    print("  - 'approve' to approve pending changes")
    print("  - 'reject' to reject pending changes")
    print("  - 'push' to approve pushing to remote")
    print("  - 'quit' or 'exit' to exit")
    print("=" * 60)
    print()

    mop_context = build_context_message(mop_content) if mop_content else ""

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

            if mop_context and len(state["messages"]) == 0:
                full_input = mop_context + "\n\nUser Query: " + user_input
            else:
                full_input = user_input

            state["messages"] = list(state["messages"]) + [HumanMessage(content=full_input)]

            try:
                result = graph.invoke(state, {"recursion_limit": 100})
                state = result

                for msg in reversed(state["messages"]):
                    if isinstance(msg, AIMessage):
                        print(f"\nAgent: {msg.content}\n")
                        break

                if state.get("awaiting_modification_approval") and state.get("pending_changes"):
                    print("\n" + format_changes_for_display(state["pending_changes"]))
                    print("\nType 'approve' to apply changes or describe what you'd like to change.\n")

                if state.get("awaiting_push_approval"):
                    branch = state.get("current_branch", "unknown")
                    print("\n" + format_push_request(branch, 1, []))
                    print()

            except Exception as e:
                error_msg = str(e)
                if "rate_limit" in error_msg.lower():
                    print("\nRate limit reached. Please wait a moment and try again.\n")
                else:
                    print(f"\nError: {e}\n")
                    import traceback
                    traceback.print_exc()
            finally:
                # ✅ Print working directory after each request (success OR error)
                print(f"\nWorking directory (current): {os.getcwd()}\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Coding Agent with Ansible & Python capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py                              # Interactive mode
  python agent.py --query "list ansible files" # Non-interactive query
  python agent.py --mop procedure.docx         # Load MOP, interactive mode
  python agent.py --mop procedure.docx --query "implement step 1"
        """
    )
    parser.add_argument(
        "--mop",
        type=str,
        help="Path to MOP (Method of Procedure) DOCX document"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Query to run in non-interactive mode"
    )
    parser.add_argument(
        "--repo",
        type=str,
        help="Override REPO_PATH from .env"
    )
    return parser.parse_args()


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    load_dotenv()
    args = parse_args()

    # Resolve repo path (DO NOT chdir here; setup_node handles it)
    repo_path = args.repo or os.getenv("REPO_PATH")

    # Load MOP if provided
    mop_content = None
    if args.mop:
        mop_content = load_mop_content(args.mop)

    # Run in appropriate mode
    if args.query:
        response = run_single_query(args.query, mop_content, repo_path=repo_path)
        print(response)
    else:
        run_interactive(mop_content, repo_path=repo_path)
