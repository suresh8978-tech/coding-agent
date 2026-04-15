"""
Terraform Tools
Runs safe Terraform CLI commands: init, plan, validate, fmt, etc.
Blocks dangerous commands (apply, destroy) for safety.
"""

import subprocess
import logging
from typing import Dict

logger = logging.getLogger("agent.terraform")

# Commands that are safe to run
ALLOWED_COMMANDS = [
    "init", "plan", "validate", "fmt",
    "state list", "state show",
    "output", "providers", "version"
]

# Commands that are blocked for safety
BLOCKED_COMMANDS = ["apply", "destroy", "import", "taint", "untaint"]


def run_command(repo_path: str, command: str) -> Dict:
    """
    Run a safe Terraform CLI command.
    
    Allowed: init, plan, validate, fmt, state list, state show, output, providers, version
    Blocked: apply, destroy (must be run manually via SSH for safety)
    """
    cmd_parts = command.strip().split()
    if not cmd_parts:
        return {"error": "No command provided."}

    # Check if command is blocked
    cmd_check = " ".join(cmd_parts[:2]) if len(cmd_parts) > 1 else cmd_parts[0]
    
    for blocked in BLOCKED_COMMANDS:
        if cmd_check == blocked or cmd_parts[0] == blocked:
            return {
                "error": f"Command 'terraform {command}' is BLOCKED for safety.",
                "reason": "This command modifies real infrastructure and could cost money or cause outages.",
                "suggestion": f"Run 'terraform {command}' manually via SSH after reviewing the plan.",
                "allowed_commands": ALLOWED_COMMANDS
            }

    # Check if command is allowed
    if cmd_check not in ALLOWED_COMMANDS and cmd_parts[0] not in ALLOWED_COMMANDS:
        return {
            "error": f"Command 'terraform {command}' is not recognized.",
            "allowed_commands": ALLOWED_COMMANDS
        }

    try:
        logger.info(f"Running: terraform {command}")
        result = subprocess.run(
            ["terraform"] + cmd_parts,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=120
        )

        response = {
            "command": f"terraform {command}",
            "exit_code": result.returncode,
            "success": result.returncode == 0
        }

        if result.stdout:
            response["stdout"] = result.stdout
        if result.stderr:
            response["stderr"] = result.stderr

        logger.info(f"terraform {command} → exit code {result.returncode}")
        return response

    except subprocess.TimeoutExpired:
        logger.error(f"terraform {command} timed out")
        return {"error": "Command timed out after 120 seconds."}
    except FileNotFoundError:
        logger.error("Terraform CLI not found")
        return {"error": "Terraform CLI not found. Make sure terraform is installed on the server."}
    except Exception as e:
        logger.error(f"terraform {command} failed: {e}")
        return {"error": str(e)}
