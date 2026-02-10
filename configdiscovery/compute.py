"""Globus Compute integration for remote HPC execution."""

from dataclasses import dataclass
from typing import Any, Callable

from globus_compute_sdk import Client, Executor
from globus_compute_sdk.serialize import ComputeSerializer, CombinedCode


# Special endpoint ID that triggers local execution
LOCAL_ENDPOINT = "local"


@dataclass
class ExecutionResult:
    """Result from a remote execution."""

    success: bool
    output: str | None = None
    error: str | None = None
    return_value: Any = None


class ComputeClient:
    """Wrapper around Globus Compute for HPC execution."""

    def __init__(self, endpoint_id: str):
        """Initialize with a Globus Compute endpoint ID."""
        self.endpoint_id = endpoint_id
        self._client: Client | None = None
        self._executor: Executor | None = None

    @property
    def client(self) -> Client:
        """Lazy-initialize the Globus Compute client."""
        if self._client is None:
            self._client = Client()
        return self._client

    @property
    def executor(self) -> Executor:
        """Lazy-initialize the executor."""
        if self._executor is None:
            self._executor = Executor(
                endpoint_id=self.endpoint_id,
                client=self.client,
                serializer=ComputeSerializer(strategy_code=CombinedCode()),
            )
        return self._executor

    def run_function(
        self, func: Callable, *args, timeout: int = 300, **kwargs
    ) -> ExecutionResult:
        """Execute a function on the remote endpoint."""
        try:
            future = self.executor.submit(func, *args, **kwargs)
            result = future.result(timeout=timeout)
            return ExecutionResult(success=True, return_value=result)
        except TimeoutError:
            return ExecutionResult(success=False, error="Execution timed out")
        except Exception as e:
            return ExecutionResult(success=False, error=str(e))

    def run_command(self, command: str, timeout: int = 300) -> ExecutionResult:
        """Execute a shell command on the remote endpoint."""

        def _run_cmd(cmd: str) -> dict:
            import subprocess

            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=timeout - 10
            )
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        result = self.run_function(_run_cmd, command, timeout=timeout)
        if not result.success:
            return result

        cmd_result = result.return_value
        if cmd_result["returncode"] == 0:
            return ExecutionResult(success=True, output=cmd_result["stdout"])
        else:
            return ExecutionResult(
                success=False,
                output=cmd_result["stdout"],
                error=cmd_result["stderr"] or f"Exit code: {cmd_result['returncode']}",
            )

    def probe_environment(self) -> dict[str, Any]:
        """Gather information about the remote environment."""

        def _probe() -> dict:
            import os
            import platform
            import shutil
            import subprocess

            info = {
                "hostname": platform.node(),
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "cwd": os.getcwd(),
                "home": os.path.expanduser("~"),
                "user": os.environ.get("USER", "unknown"),
            }

            # Check for common tools
            tools = ["module", "conda", "spack", "pip", "git"]
            info["available_tools"] = {}
            for tool in tools:
                info["available_tools"][tool] = shutil.which(tool) is not None

            # Try to get module list
            try:
                result = subprocess.run(
                    "module avail 2>&1 | head -50",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                info["modules_sample"] = result.stdout or result.stderr
            except Exception:
                info["modules_sample"] = None

            # Check scheduler
            schedulers = {
                "slurm": "squeue",
                "pbs": "qstat",
                "lsf": "bjobs",
            }
            info["scheduler"] = None
            for name, cmd in schedulers.items():
                if shutil.which(cmd):
                    info["scheduler"] = name
                    break

            return info

        result = self.run_function(_probe, timeout=60)
        if result.success:
            return result.return_value
        else:
            return {"error": result.error}

    def check_module(self, module_name: str) -> ExecutionResult:
        """Check if a module is available and get its info."""
        return self.run_command(f"module show {module_name} 2>&1")

    def load_modules_and_run(
        self, modules: list[str], command: str, timeout: int = 300
    ) -> ExecutionResult:
        """Load modules and run a command."""
        module_loads = " && ".join(f"module load {m}" for m in modules)
        full_command = f"{module_loads} && {command}" if modules else command
        return self.run_command(full_command, timeout=timeout)

    def test_config(
        self,
        modules: list[str],
        env_vars: dict[str, str],
        setup_commands: list[str],
        test_command: str,
        timeout: int = 300,
    ) -> ExecutionResult:
        """Test a full configuration setup."""

        def _test_config(
            mods: list[str],
            env: dict[str, str],
            setup: list[str],
            test_cmd: str,
        ) -> dict:
            import os
            import subprocess

            # Set environment variables
            for key, value in env.items():
                os.environ[key] = value

            # Build command sequence
            commands = []
            if mods:
                commands.extend(f"module load {m}" for m in mods)
            commands.extend(setup)
            commands.append(test_cmd)

            full_command = " && ".join(commands)
            result = subprocess.run(
                full_command, shell=True, capture_output=True, text=True
            )
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": full_command,
            }

        result = self.run_function(
            _test_config, modules, env_vars, setup_commands, test_command, timeout=timeout
        )
        if not result.success:
            return result

        test_result = result.return_value
        if test_result["returncode"] == 0:
            return ExecutionResult(
                success=True,
                output=test_result["stdout"],
            )
        else:
            return ExecutionResult(
                success=False,
                output=test_result["stdout"],
                error=test_result["stderr"] or f"Exit code: {test_result['returncode']}",
            )

    def register_and_run_function(
        self, function_code: str, function_name: str, *args, timeout: int = 300, **kwargs
    ) -> ExecutionResult:
        """Register a function from code string and execute it."""

        def _run_dynamic(code: str, fname: str, fn_args: tuple, fn_kwargs: dict) -> Any:
            namespace = {}
            exec(code, namespace)
            func = namespace[fname]
            return func(*fn_args, **fn_kwargs)

        return self.run_function(
            _run_dynamic, function_code, function_name, args, kwargs, timeout=timeout
        )

    def close(self) -> None:
        """Clean up resources."""
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        self._client = None

    def __enter__(self) -> "ComputeClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class LocalComputeClient(ComputeClient):
    """Local execution client for testing without Globus Compute.

    Runs functions locally instead of on a remote endpoint.
    Use endpoint_id="local" to get this behavior.
    """

    def __init__(self, endpoint_id: str = LOCAL_ENDPOINT):
        self.endpoint_id = endpoint_id
        self._client = None
        self._executor = None

    def run_function(
        self, func: Callable, *args, timeout: int = 300, **kwargs
    ) -> ExecutionResult:
        """Execute a function locally."""
        try:
            result = func(*args, **kwargs)
            return ExecutionResult(success=True, return_value=result)
        except Exception as e:
            return ExecutionResult(success=False, error=str(e))

    def close(self) -> None:
        """No cleanup needed for local execution."""
        pass


def get_compute_client(endpoint_id: str) -> ComputeClient:
    """Factory function to get appropriate compute client.

    Use endpoint_id="local" for local testing.
    """
    if endpoint_id == LOCAL_ENDPOINT:
        return LocalComputeClient(endpoint_id)
    return ComputeClient(endpoint_id)
