"""Discovery engine - orchestrates Claude to discover HPC configurations."""

import re
from datetime import date

import anthropic


def extract_function_name(code: str) -> str:
    """Extract the first function name from Python code."""
    match = re.search(r"def\s+(\w+)\s*\(", code)
    if match:
        return match.group(1)
    return "run"  # fallback default


def validate_and_fix_function_code(code: str) -> str:
    """Validate Python function code and attempt to fix common issues.

    Args:
        code: Python function code string

    Returns:
        Fixed code string

    Raises:
        SyntaxError: If code cannot be fixed
    """
    # First, try to compile as-is
    try:
        compile(code, "<string>", "exec")
        return code
    except SyntaxError:
        pass

    # Common fix: remove trailing braces (Claude sometimes adds JSON-style closing)
    fixed = code.rstrip()
    while fixed.endswith("}"):
        fixed = fixed[:-1].rstrip()
        try:
            compile(fixed, "<string>", "exec")
            return fixed
        except SyntaxError:
            continue

    # Common fix: remove leading/trailing backticks (markdown code blocks)
    fixed = code.strip()
    if fixed.startswith("```"):
        fixed = re.sub(r"^```\w*\n?", "", fixed)
        fixed = re.sub(r"\n?```$", "", fixed)
        try:
            compile(fixed, "<string>", "exec")
            return fixed
        except SyntaxError:
            pass

    # If we can't fix it, raise the original error
    compile(code, "<string>", "exec")  # This will raise SyntaxError
    return code  # Never reached


from rich.console import Console
from rich.panel import Panel

from .claude_tools import TOOL_DEFINITIONS, ToolExecutor
from .compute import get_compute_client
from .schema import (
    DiscoveryLog,
    Environment,
    Execution,
    HPCConfig,
    Installation,
    Resources,
)


SYSTEM_PROMPT = """You are an expert at configuring and running software on HPC (High Performance Computing) systems. Your goal is to discover how to run a specific piece of software on a remote HPC system.

You have access to tools that let you:
1. Probe the remote environment to understand what's available
2. Run commands to test configurations
3. Check and load HPC modules
4. Fetch documentation from URLs
5. Test complete configurations
6. Propose a final working configuration

## Discovery Process

1. First, probe the environment to understand what tools and modules are available
2. If documentation URLs are provided, fetch and read them
3. Try to find the software or relevant modules using commands like `module avail | grep <name>`
4. Experiment with loading modules and running test commands
5. If the software needs installation, figure out the steps
6. Once you have a working setup, test it with `test_config`
7. When everything works, call `propose_config` with the complete configuration

## Important Guidelines

- Be methodical - check one thing at a time
- Save notes about what you learn using `save_discovery_note`
- If something fails, analyze the error and try alternatives
- Prefer using existing modules over installing from source
- Consider dependencies - some modules may need others loaded first
- Test your configuration before proposing it

When you call `propose_config`, include:
- All required modules
- Any environment variables needed
- Setup commands that run each time
- Installation steps (if any, these run once)
- A verification command to check the setup works
- A Python function that ACTUALLY RUNS the software (see below)
- Resource requirements (nodes, memory, walltime, etc.)

## CRITICAL: The Execution Function

The `execution_function` you provide must ACTUALLY RUN the software, not just print status or instructions. This function will be called via Globus Compute to execute real work.

BAD example (don't do this):
```python
def run_cesm():
    print("CESM is installed at /path/to/cesm")
    print("To run it, do X, Y, Z...")  # Just prints instructions!
    return {"status": "ready"}
```

GOOD example (do this):
```python
def run_cesm(compset="X", resolution="f19_g17", case_name="test_case"):
    import subprocess
    import os

    cesm_root = os.path.expanduser("~/cesm/my_cesm_sandbox")
    scripts_dir = os.path.join(cesm_root, "cime", "scripts")
    case_dir = os.path.expanduser(f"~/cesm_cases/{case_name}")

    # Create the case
    subprocess.run([
        os.path.join(scripts_dir, "create_newcase"),
        "--case", case_dir,
        "--compset", compset,
        "--res", resolution
    ], check=True)

    # Setup, build, and run
    os.chdir(case_dir)
    subprocess.run(["./case.setup"], check=True)
    subprocess.run(["./case.build"], check=True)
    subprocess.run(["./case.submit"], check=True)

    return {"case_dir": case_dir, "status": "submitted"}
```

The function should:
1. Accept parameters for common options (input files, output dirs, configurations)
2. Actually execute the software using subprocess or Python APIs
3. Return meaningful results (output paths, computed values, status)
4. Handle errors appropriately

## IMPORTANT: Shell command compatibility

When using subprocess with shell=True, use `executable='/bin/bash'` to ensure bash features work:

```python
# WRONG - 'source' may not work in /bin/sh
subprocess.run("source activate myenv && python script.py", shell=True)

# CORRECT - explicitly use bash
subprocess.run("source activate myenv && python script.py", shell=True, executable='/bin/bash')
```

This ensures commands like `source`, bash arrays, and other bash-specific features work correctly.

## IMPORTANT: Complete the full workflow

DO NOT stop halfway and return "next_steps" or instructions. The function must attempt the COMPLETE workflow:

BAD (stops early):
```python
def run_cesm(...):
    create_newcase(...)
    case_setup(...)
    return {"status": "configured", "next_steps": ["build", "submit"]}  # WRONG!
```

GOOD (completes the workflow):
```python
def run_cesm(...):
    create_newcase(...)
    case_setup(...)
    case_build(...)      # Actually build
    case_submit(...)     # Actually submit
    return {"status": "submitted", "output_dir": output_dir}  # CORRECT!
```

If a step might fail or take a long time, add a parameter to skip it (defaulting to run):
```python
def run_cesm(..., skip_build=False, skip_submit=False):
    create_newcase(...)
    case_setup(...)
    if not skip_build:
        case_build(...)
    if not skip_submit:
        case_submit(...)
    return {"status": "complete", ...}
```

The user expects to call `configdiscovery run config.yaml` and have the software actually execute, not receive instructions for manual steps.
"""


class DiscoveryEngine:
    """Orchestrates Claude to discover HPC configurations."""

    def __init__(
        self,
        endpoint_id: str,
        model: str = "claude-sonnet-4-20250514",
        max_iterations: int = 50,
        verbose: bool = True,
    ):
        self.endpoint_id = endpoint_id
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.console = Console()
        self.client = anthropic.Anthropic()

    def discover(
        self,
        software_name: str,
        hpc_system: str,
        docs_urls: list[str] | None = None,
        additional_context: str | None = None,
    ) -> HPCConfig:
        """Run the discovery process for a software package.

        Args:
            software_name: Name of the software to configure
            hpc_system: Name of the HPC system
            docs_urls: Optional URLs to documentation
            additional_context: Additional context about requirements

        Returns:
            The discovered configuration
        """
        # Build the initial user message
        user_message = f"I need to configure and run **{software_name}** on the **{hpc_system}** HPC system."

        if docs_urls:
            user_message += "\n\nHere are some documentation URLs that may help:\n"
            for url in docs_urls:
                user_message += f"- {url}\n"

        if additional_context:
            user_message += f"\n\nAdditional context: {additional_context}"

        user_message += "\n\nPlease discover how to set up and run this software, then propose a configuration."

        # Initialize the compute client and tool executor
        with get_compute_client(self.endpoint_id) as compute:
            executor = ToolExecutor(compute)

            # Track docs URLs provided upfront
            if docs_urls:
                executor.docs_consulted.extend(docs_urls)

            # Run the conversation loop
            messages = [{"role": "user", "content": user_message}]

            for iteration in range(self.max_iterations):
                if self.verbose:
                    self.console.print(
                        f"[dim]Iteration {iteration + 1}/{self.max_iterations}[/dim]"
                    )

                # Call Claude
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=TOOL_DEFINITIONS,
                    messages=messages,
                )

                # Process the response
                assistant_content = []
                tool_results = []

                for block in response.content:
                    if block.type == "text":
                        if self.verbose:
                            self.console.print(Panel(block.text, title="Claude"))
                        assistant_content.append(block)
                    elif block.type == "tool_use":
                        if self.verbose:
                            self.console.print(
                                f"[blue]Tool: {block.name}[/blue]"
                            )
                            self.console.print(f"[dim]{block.input}[/dim]")

                        # Execute the tool
                        result = executor.execute(block.name, block.input)

                        if self.verbose:
                            success = result.get("success", False)
                            color = "green" if success else "red"
                            self.console.print(f"[{color}]Result: {result}[/{color}]")

                        assistant_content.append(block)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                        })

                        # Check if a config was proposed
                        if executor.proposed_config is not None:
                            if self.verbose:
                                self.console.print(
                                    "[green]Configuration proposed! Building final config...[/green]"
                                )
                            return self._build_config(
                                software_name,
                                hpc_system,
                                executor,
                            )

                # Add assistant message
                messages.append({"role": "assistant", "content": assistant_content})

                # If there were tool uses, add results and continue
                if tool_results:
                    messages.append({"role": "user", "content": tool_results})
                elif response.stop_reason == "end_turn":
                    # Claude finished without proposing - prompt to propose
                    messages.append({
                        "role": "user",
                        "content": "Please propose a configuration using the `propose_config` tool.",
                    })

            raise RuntimeError(
                f"Discovery did not complete within {self.max_iterations} iterations"
            )

    def _build_config(
        self,
        software_name: str,
        hpc_system: str,
        executor: ToolExecutor,
    ) -> HPCConfig:
        """Build an HPCConfig from the proposed configuration."""
        proposed = executor.proposed_config
        if proposed is None:
            raise ValueError("No configuration was proposed")

        # Build environment
        environment = Environment(
            modules=proposed.get("modules", []),
            conda_env=proposed.get("conda_env"),
            conda_packages=proposed.get("conda_packages", []),
            pip_packages=proposed.get("pip_packages", []),
            env_vars=proposed.get("env_vars", {}),
            setup_commands=proposed.get("setup_commands", []),
        )

        # Build installation
        installation = Installation(
            steps=proposed.get("installation_steps", []),
            verification=proposed.get("verification_command"),
        )

        # Build resources
        res_input = proposed.get("resources", {})
        resources = Resources(
            nodes=res_input.get("nodes", 1),
            cores_per_node=res_input.get("cores_per_node"),
            memory_gb=res_input.get("memory_gb"),
            gpus=res_input.get("gpus"),
            walltime=res_input.get("walltime", "01:00:00"),
            queue=res_input.get("queue"),
        )

        # Build execution - validate and fix function code, extract name
        function_code = proposed["execution_function"]
        try:
            function_code = validate_and_fix_function_code(function_code)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python function code: {e}")

        function_name = extract_function_name(function_code)

        execution = Execution(
            function=function_code,
            function_name=function_name,
            resources=resources,
        )

        # Build discovery log
        discovery_log = DiscoveryLog(
            date=date.today(),
            docs_consulted=executor.docs_consulted,
            attempts=executor.attempts,
            notes="\n".join(executor.notes) if executor.notes else None,
            claude_model=self.model,
        )

        return HPCConfig(
            name=software_name,
            hpc_system=hpc_system,
            endpoint_id=self.endpoint_id,
            environment=environment,
            installation=installation,
            execution=execution,
            discovery_log=discovery_log,
        )

    def refine(
        self,
        existing_config: HPCConfig,
        instructions: str | None = None,
    ) -> HPCConfig:
        """Refine an existing configuration.

        Args:
            existing_config: The configuration to improve
            instructions: Optional specific instructions for refinement

        Returns:
            A refined configuration
        """
        # Build context from existing config
        config_summary = f"""
## Existing Configuration for {existing_config.name} on {existing_config.hpc_system}

### Environment
- Modules: {existing_config.environment.modules}
- Conda env: {existing_config.environment.conda_env}
- Conda packages: {existing_config.environment.conda_packages}
- Pip packages: {existing_config.environment.pip_packages}
- Env vars: {existing_config.environment.env_vars}
- Setup commands: {existing_config.environment.setup_commands}

### Installation
- Steps: {existing_config.installation.steps}
- Verification: {existing_config.installation.verification}

### Current Execution Function
```python
{existing_config.execution.function}
```

### Previous Discovery Notes
{existing_config.discovery_log.notes or "No notes"}

### What was learned
- Attempts: {existing_config.discovery_log.attempts}
- Docs consulted: {existing_config.discovery_log.docs_consulted}
"""

        default_instructions = """
Improve this configuration, especially the execution function. The current function
may stop early and return "next_steps" instead of completing the full workflow.

Make the execution function actually run the software to completion. If steps might
fail or take a long time, add skip parameters (defaulting to False) rather than
stopping early.
"""

        user_message = f"""I have an existing configuration that needs improvement.

{config_summary}

## Instructions
{instructions or default_instructions}

Please test and refine this configuration, then propose an improved version using `propose_config`.
Keep what works, fix what doesn't, and make the execution function complete the full workflow.
"""

        # Run discovery with existing config as context
        with get_compute_client(self.endpoint_id) as compute:
            executor = ToolExecutor(compute)
            executor.docs_consulted = list(existing_config.discovery_log.docs_consulted)

            messages = [{"role": "user", "content": user_message}]

            for iteration in range(self.max_iterations):
                if self.verbose:
                    self.console.print(
                        f"[dim]Refine iteration {iteration + 1}/{self.max_iterations}[/dim]"
                    )

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=TOOL_DEFINITIONS,
                    messages=messages,
                )

                assistant_content = []
                tool_results = []

                for block in response.content:
                    if block.type == "text":
                        if self.verbose:
                            self.console.print(Panel(block.text, title="Claude"))
                        assistant_content.append(block)
                    elif block.type == "tool_use":
                        if self.verbose:
                            self.console.print(f"[blue]Tool: {block.name}[/blue]")
                            self.console.print(f"[dim]{block.input}[/dim]")

                        result = executor.execute(block.name, block.input)

                        if self.verbose:
                            success = result.get("success", False)
                            color = "green" if success else "red"
                            self.console.print(f"[{color}]Result: {result}[/{color}]")

                        assistant_content.append(block)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                        })

                        if executor.proposed_config is not None:
                            if self.verbose:
                                self.console.print(
                                    "[green]Refined configuration proposed![/green]"
                                )
                            return self._build_config(
                                existing_config.name,
                                existing_config.hpc_system,
                                executor,
                            )

                messages.append({"role": "assistant", "content": assistant_content})

                if tool_results:
                    messages.append({"role": "user", "content": tool_results})
                elif response.stop_reason == "end_turn":
                    messages.append({
                        "role": "user",
                        "content": "Please propose the refined configuration using `propose_config`.",
                    })

            raise RuntimeError(
                f"Refinement did not complete within {self.max_iterations} iterations"
            )


class ConfigRunner:
    """Run software using a saved configuration."""

    def __init__(self, config: HPCConfig):
        self.config = config

    def test_config(self) -> dict:
        """Test the configuration by running the verification command.

        Returns:
            dict with 'success', 'output', and optionally 'error'
        """
        verification = self.config.installation.verification
        if not verification:
            return {"success": True, "output": "No verification command defined"}

        with get_compute_client(self.config.endpoint_id) as compute:
            result = compute.test_config(
                modules=self.config.environment.modules,
                env_vars=self.config.environment.env_vars,
                setup_commands=self.config.environment.setup_commands,
                test_command=verification,
            )

            if result.success:
                return {"success": True, "output": result.output}
            else:
                return {"success": False, "error": result.error, "output": result.output}

    def install_dependencies(self, force: bool = False) -> dict:
        """Install all dependencies specified in the configuration.

        This includes:
        - Creating conda environments if specified
        - Installing conda packages
        - Installing pip packages
        - Running installation steps

        Args:
            force: If True, reinstall even if already present

        Returns:
            dict with 'success', 'details', and optionally 'error'
        """
        details = []
        env = self.config.environment

        with get_compute_client(self.config.endpoint_id) as compute:
            # Determine base conda path
            conda_path = "/soft/applications/conda/2025-09-25/mconda3"
            conda_bin = f"{conda_path}/condabin/conda"
            pip_bin = f"{conda_path}/bin/pip"
            python_bin = f"{conda_path}/bin/python"

            # Step 1: Create conda environment if specified
            if env.conda_env:
                env_name = env.conda_env
                env_pip = f"{conda_path}/envs/{env_name}/bin/pip"

                # Check if env exists AND has pip
                check_cmd = f"test -x {env_pip}"
                result = compute.run_command(check_cmd)

                if not result.success or force:
                    # Check if env exists but is broken (no pip)
                    check_env_cmd = f"{conda_bin} env list | grep -q '^{env_name} '"
                    env_exists = compute.run_command(check_env_cmd)

                    if env_exists.success and not force:
                        # Env exists but pip missing - try to install pip
                        details.append(f"Conda environment {env_name} exists but incomplete, fixing...")
                        fix_cmd = f"source {conda_path}/etc/profile.d/conda.sh && conda activate {env_name} && conda install -y pip"
                        result = compute.run_command(fix_cmd, timeout=300)
                        if not result.success:
                            # Can't fix, recreate
                            details.append(f"Recreating conda environment: {env_name}")
                            remove_cmd = f"{conda_bin} env remove -n {env_name} -y"
                            compute.run_command(remove_cmd, timeout=120)
                            create_cmd = f"{conda_bin} create -n {env_name} python=3.11 pip -y"
                            result = compute.run_command(create_cmd, timeout=600)
                            if not result.success:
                                return {"success": False, "error": f"Failed to create conda env: {result.error}"}
                    else:
                        # Create new environment
                        create_cmd = f"{conda_bin} create -n {env_name} python=3.11 pip -y"
                        result = compute.run_command(create_cmd, timeout=600)
                        if result.success:
                            details.append(f"Created conda environment: {env_name}")
                        else:
                            return {"success": False, "error": f"Failed to create conda env: {result.error}"}
                else:
                    details.append(f"Conda environment ready: {env_name}")

                # Update paths for the conda env
                pip_bin = f"{conda_path}/envs/{env_name}/bin/pip"
                python_bin = f"{conda_path}/envs/{env_name}/bin/python"

            # Step 2: Install conda packages
            if env.conda_packages:
                conda_activate = f"source {conda_path}/etc/profile.d/conda.sh && conda activate {env.conda_env}" if env.conda_env else ""
                for pkg in env.conda_packages:
                    if conda_activate:
                        install_cmd = f"{conda_activate} && conda install -y {pkg}"
                    else:
                        install_cmd = f"{conda_bin} install -y {pkg}"

                    result = compute.run_command(install_cmd, timeout=600)
                    if result.success:
                        details.append(f"Installed conda package: {pkg}")
                    else:
                        # Try with conda-forge
                        if conda_activate:
                            install_cmd = f"{conda_activate} && conda install -y -c conda-forge {pkg}"
                        else:
                            install_cmd = f"{conda_bin} install -y -c conda-forge {pkg}"
                        result = compute.run_command(install_cmd, timeout=600)
                        if result.success:
                            details.append(f"Installed conda package (conda-forge): {pkg}")
                        else:
                            return {"success": False, "error": f"Failed to install conda package {pkg}: {result.error}"}

            # Step 3: Install pip packages
            if env.pip_packages:
                # Use conda activate for pip installs when we have a conda env
                if env.conda_env:
                    pip_prefix = f"source {conda_path}/etc/profile.d/conda.sh && conda activate {env.conda_env} && pip"
                else:
                    pip_prefix = pip_bin

                for pkg in env.pip_packages:
                    # For conda envs, always install via activated environment
                    if env.conda_env:
                        install_cmd = f"{pip_prefix} install {pkg}"
                    else:
                        # Check if already installed for non-conda envs
                        check_cmd = f"{python_bin} -c \"import {pkg.split('[')[0].replace('-', '_')}\" 2>/dev/null"
                        result = compute.run_command(check_cmd)
                        if result.success and not force:
                            details.append(f"Pip package already installed: {pkg}")
                            continue
                        install_cmd = f"{pip_bin} install {pkg}"

                    result = compute.run_command(install_cmd, timeout=300)
                    if result.success:
                        details.append(f"Installed pip package: {pkg}")
                    else:
                        return {"success": False, "error": f"Failed to install pip package {pkg}: {result.error}"}

            # Step 4: Run installation steps
            if self.config.installation.steps:
                # Build env var exports prefix
                env_exports = ""
                if env.env_vars:
                    env_exports = " && ".join(f"export {k}={v}" for k, v in env.env_vars.items()) + " && "

                for step in self.config.installation.steps:
                    # Prepend env var exports to each step
                    full_cmd = f"{env_exports}{step}" if env_exports else step
                    result = compute.run_command(full_cmd, timeout=600)
                    if result.success:
                        details.append(f"Ran: {step[:50]}...")
                    else:
                        # Some steps are informational (echo), don't fail on those
                        if step.startswith("echo"):
                            details.append(f"Info: {step}")
                        else:
                            return {"success": False, "error": f"Installation step failed: {step}\n{result.error}"}

            return {"success": True, "details": details}

    def run(self, *args, **kwargs):
        """Execute the configured software."""
        with get_compute_client(self.config.endpoint_id) as compute:
            # First, test/setup the environment
            setup_result = compute.test_config(
                modules=self.config.environment.modules,
                env_vars=self.config.environment.env_vars,
                setup_commands=self.config.environment.setup_commands,
                test_command="echo 'Environment ready'",
            )

            if not setup_result.success:
                raise RuntimeError(f"Failed to setup environment: {setup_result.error}")

            # Run the actual function
            result = compute.register_and_run_function(
                function_code=self.config.execution.function,
                function_name=self.config.execution.function_name,
                *args,
                **kwargs,
            )

            if not result.success:
                raise RuntimeError(f"Execution failed: {result.error}")

            return result.return_value
