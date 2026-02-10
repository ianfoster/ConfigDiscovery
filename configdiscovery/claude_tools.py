"""Tool definitions for Claude during discovery process."""

from typing import Any

import httpx

from .compute import ComputeClient, ExecutionResult


TOOL_DEFINITIONS = [
    {
        "name": "run_command",
        "description": "Execute a shell command on the remote HPC system. Use this to test commands, check software availability, run installations, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 300)",
                    "default": 300,
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "probe_environment",
        "description": "Gather information about the remote HPC environment including available tools, scheduler type, module system, etc.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "check_module",
        "description": "Check if an HPC module is available and get information about it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "module_name": {
                    "type": "string",
                    "description": "The module name to check (e.g., 'python/3.11')",
                },
            },
            "required": ["module_name"],
        },
    },
    {
        "name": "load_modules_and_run",
        "description": "Load specified modules and then run a command. Use this to test module combinations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "modules": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of modules to load",
                },
                "command": {
                    "type": "string",
                    "description": "Command to run after loading modules",
                },
            },
            "required": ["modules", "command"],
        },
    },
    {
        "name": "test_config",
        "description": "Test a complete configuration setup with modules, environment variables, setup commands, and a test command.",
        "input_schema": {
            "type": "object",
            "properties": {
                "modules": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Modules to load",
                },
                "env_vars": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Environment variables to set",
                },
                "setup_commands": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Setup commands to run",
                },
                "test_command": {
                    "type": "string",
                    "description": "Command to verify the setup works",
                },
            },
            "required": ["test_command"],
        },
    },
    {
        "name": "fetch_documentation",
        "description": "Fetch and read documentation from a URL. Use this to read installation guides, user manuals, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL of the documentation to fetch",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "save_discovery_note",
        "description": "Save a note about what you learned during discovery. These notes will be included in the final configuration.",
        "input_schema": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "Note about what was learned or tried",
                },
            },
            "required": ["note"],
        },
    },
    {
        "name": "propose_config",
        "description": "Propose a final configuration. Call this when you have successfully tested a working setup.",
        "input_schema": {
            "type": "object",
            "properties": {
                "modules": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Required modules to load",
                },
                "conda_env": {
                    "type": "string",
                    "description": "Conda environment name (if needed)",
                },
                "conda_packages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Packages to install in conda",
                },
                "pip_packages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Packages to install via pip",
                },
                "env_vars": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Environment variables to set",
                },
                "setup_commands": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Setup commands to run",
                },
                "installation_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One-time installation commands",
                },
                "verification_command": {
                    "type": "string",
                    "description": "Command to verify the software works",
                },
                "execution_function": {
                    "type": "string",
                    "description": "Python function code for running the software. Must be valid Python code starting with 'def function_name(...):' and ending with the function body. Do NOT include any JSON braces or other wrapper syntax - just the raw Python function definition.",
                },
                "resources": {
                    "type": "object",
                    "properties": {
                        "nodes": {"type": "integer"},
                        "cores_per_node": {"type": "integer"},
                        "memory_gb": {"type": "integer"},
                        "gpus": {"type": "integer"},
                        "walltime": {"type": "string"},
                        "queue": {"type": "string"},
                    },
                    "description": "Resource requirements",
                },
            },
            "required": ["execution_function"],
        },
    },
]


class ToolExecutor:
    """Execute tools during the discovery process."""

    def __init__(self, compute_client: ComputeClient):
        self.compute = compute_client
        self.notes: list[str] = []
        self.docs_consulted: list[str] = []
        self.attempts = 0
        self.proposed_config: dict | None = None

    def execute(self, tool_name: str, tool_input: dict) -> dict[str, Any]:
        """Execute a tool and return the result."""
        self.attempts += 1

        if tool_name == "run_command":
            return self._run_command(tool_input)
        elif tool_name == "probe_environment":
            return self._probe_environment()
        elif tool_name == "check_module":
            return self._check_module(tool_input)
        elif tool_name == "load_modules_and_run":
            return self._load_modules_and_run(tool_input)
        elif tool_name == "test_config":
            return self._test_config(tool_input)
        elif tool_name == "fetch_documentation":
            return self._fetch_documentation(tool_input)
        elif tool_name == "save_discovery_note":
            return self._save_note(tool_input)
        elif tool_name == "propose_config":
            return self._propose_config(tool_input)
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def _result_to_dict(self, result: ExecutionResult) -> dict:
        """Convert ExecutionResult to dict for tool response."""
        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
        }

    def _run_command(self, input: dict) -> dict:
        command = input["command"]
        timeout = input.get("timeout", 300)
        result = self.compute.run_command(command, timeout=timeout)
        return self._result_to_dict(result)

    def _probe_environment(self) -> dict:
        return self.compute.probe_environment()

    def _check_module(self, input: dict) -> dict:
        module_name = input["module_name"]
        result = self.compute.check_module(module_name)
        return self._result_to_dict(result)

    def _load_modules_and_run(self, input: dict) -> dict:
        modules = input["modules"]
        command = input["command"]
        result = self.compute.load_modules_and_run(modules, command)
        return self._result_to_dict(result)

    def _test_config(self, input: dict) -> dict:
        modules = input.get("modules", [])
        env_vars = input.get("env_vars", {})
        setup_commands = input.get("setup_commands", [])
        test_command = input["test_command"]
        result = self.compute.test_config(
            modules, env_vars, setup_commands, test_command
        )
        return self._result_to_dict(result)

    def _fetch_documentation(self, input: dict) -> dict:
        url = input["url"]
        self.docs_consulted.append(url)
        try:
            response = httpx.get(url, follow_redirects=True, timeout=30)
            response.raise_for_status()

            # Basic HTML to text conversion
            content = response.text
            if "<html" in content.lower():
                # Very basic HTML stripping - in production use a proper parser
                import re
                content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE)
                content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL | re.IGNORECASE)
                content = re.sub(r"<[^>]+>", " ", content)
                content = re.sub(r"\s+", " ", content)
                content = content[:10000]  # Truncate

            return {"success": True, "content": content}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _save_note(self, input: dict) -> dict:
        note = input["note"]
        self.notes.append(note)
        return {"success": True, "message": f"Note saved. Total notes: {len(self.notes)}"}

    def _propose_config(self, input: dict) -> dict:
        self.proposed_config = input
        return {
            "success": True,
            "message": "Configuration proposal recorded. The discovery process will now complete.",
        }
