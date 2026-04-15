# Terraform Coding Agent

An AI-powered agent that manages Terraform infrastructure files. Integrates with Open WebUI as a Tool Server.

## Project Structure

```
coding-agent/
├── tools/                    # Agent capabilities
│   ├── __init__.py
│   ├── file_manager.py       # File operations (list, read, create, modify, delete)
│   ├── terraform_tools.py    # Terraform CLI commands (init, plan, validate, fmt)
│   ├── git_tools.py          # Git operations (status, diff, commit, push)
│   └── access_control.py     # User permission management
├── .env                      # Environment config (repo path, API keys)
├── .env.example              # Template for .env
├── .gitignore
├── agent.py                  # Main agent — FastAPI server
├── agent.log                 # Runtime logs
├── Dockerfile                # Docker build instructions
├── requirements.txt          # Python dependencies
├── user_config.yaml          # User access control config
└── README.md
```

## Capabilities

| Feature | Description |
|---------|-------------|
| Analyze Repo | Scan all .tf files — show providers, resources, variables, modules |
| List Files | List all Terraform and related files |
| Show File | Display file contents |
| Create File | Create new .tf files with Terraform code |
| Modify File | Edit existing files (creates .bak backup) |
| Delete File | Remove files (creates backup) |
| Terraform CLI | Run init, plan, validate, fmt, state list, output |
| Git Status | Show modified/untracked/staged files |
| Git Commit | Stage and commit changes |
| Git Push | Push to remote (admin only, asks for confirmation) |

## Access Levels

Configured in `user_config.yaml`:

| Level | Can Do |
|-------|--------|
| `read-only` | View files, terraform plan/validate, git status |
| `read-write` | Everything above + create/modify/delete files, git commit |
| `admin` | Everything above + git push to remote |

## Quick Start

### Local Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure .env
cp .env.example .env
# Edit .env with your repo path

# Run
python agent.py
# Open: http://localhost:8001/docs
```

### Docker Deployment
```bash
docker build -t coding-agent .
docker run -d \
  --name coding-agent \
  --restart unless-stopped \
  --network host \
  -v /path/to/terraform/repo:/repo \
  -v $(pwd)/user_config.yaml:/app/user_config.yaml \
  -e TERRAFORM_REPO_PATH=/repo \
  coding-agent
```

### Open WebUI Integration
1. Deploy the agent (Docker or direct)
2. In Open WebUI: Admin → Settings → Integrations → Tool Servers → "+"
3. Enter URL: `http://localhost:8001`
4. Create an Agent: Workspace → Models → "+"
5. Set system prompt to use the Terraform tools
6. Assign access to users
