"""
Access Control
Reads user_config.yaml and enforces access levels.
Reloads config on every request — no restart needed.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger("agent.access")

# Try to import yaml
try:
    import yaml
except ImportError:
    yaml = None


def _load_yaml_config() -> Dict:
    """Load user_config.yaml. Reloads every time for live updates."""
    config_path = os.getenv("USER_CONFIG_PATH", "user_config.yaml")
    
    # Also check relative to the app directory
    if not os.path.exists(config_path):
        app_dir = Path(__file__).parent.parent
        config_path = str(app_dir / "user_config.yaml")
    
    if not os.path.exists(config_path):
        logger.warning(f"Config not found: {config_path}")
        return {"admins": [], "users": [], "default_access": "deny"}
    
    try:
        if yaml:
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        else:
            # Fallback: simple parser if yaml not installed
            return _parse_simple_yaml(config_path)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {"admins": [], "users": [], "default_access": "deny"}


def _parse_simple_yaml(path: str) -> Dict:
    """Simple YAML-like parser as fallback when PyYAML not installed."""
    import json
    # Try loading as JSON first (for config.json fallback)
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        pass
    
    # Return empty config
    return {"admins": [], "users": [], "default_access": "deny"}


def get_user_access(email: str) -> Tuple[str, str]:
    """
    Get user's access level from config.
    Returns (access_level, user_name).
    
    Access levels:
        "admin"      → Full access: read, write, push to remote
        "read-write" → Can view and modify files, run terraform
        "read-only"  → Can only view files and run terraform plan/validate
        "deny"       → No access
    """
    if not email:
        return "deny", "Unknown"
    
    email = email.lower().strip()
    config = _load_yaml_config()
    
    # Check admins
    for admin in config.get("admins", []):
        admin_email = admin.get("email", "").lower().strip() if isinstance(admin, dict) else str(admin).lower().strip()
        if admin_email == email:
            name = admin.get("name", email.split("@")[0]) if isinstance(admin, dict) else email.split("@")[0]
            return "admin", name
    
    # Check regular users
    for user in config.get("users", []):
        user_email = user.get("email", "").lower().strip() if isinstance(user, dict) else ""
        if user_email == email:
            access = user.get("access", "read-write")
            name = user.get("name", email.split("@")[0])
            return access, name
    
    # Default access
    default = config.get("default_access", "deny")
    return default, email.split("@")[0]


def check_permission(email: str, action: str) -> Tuple[bool, str]:
    """
    Check if a user has permission for a specific action.
    
    Actions and required access levels:
        "read"      → read-only, read-write, admin
        "write"     → read-write, admin
        "push"      → admin only
        "terraform" → read-only (plan/validate only), read-write, admin
    """
    access, name = get_user_access(email)
    
    if access == "deny":
        return False, f"Access denied for {email}. Contact your admin."
    
    permission_map = {
        "read":      ["read-only", "read-write", "admin"],
        "write":     ["read-write", "admin"],
        "push":      ["admin"],
        "terraform": ["read-only", "read-write", "admin"],
    }
    
    required = permission_map.get(action, [])
    
    if access in required:
        return True, ""
    
    return False, f"Action '{action}' requires {' or '.join(required)} access. You have: {access}"


def get_terraform_allowed_commands(email: str) -> list:
    """Get which terraform commands a user can run based on their access level."""
    access, _ = get_user_access(email)
    
    if access == "admin":
        return ["init", "plan", "validate", "fmt", "state list", "state show", "output", "providers", "version"]
    elif access == "read-write":
        return ["init", "plan", "validate", "fmt", "state list", "state show", "output", "providers", "version"]
    elif access == "read-only":
        return ["plan", "validate", "output", "providers", "version"]
    else:
        return []
