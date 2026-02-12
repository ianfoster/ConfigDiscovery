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

        def _run_cmd(cmd: str, cmd_timeout: int) -> dict:
            import subprocess

            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=cmd_timeout
            )
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        result = self.run_function(_run_cmd, command, timeout - 10, timeout=timeout)
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


# Well-known Globus Compute endpoints for HPC systems
HPC_COMPUTE_ENDPOINTS = {
    "polaris": "0554c761-5a62-474d-b26e-df7455682bba",
    "aurora": "608403f7-58df-44f6-a27b-b6084cae113d",
}

# Well-known Globus Transfer endpoints for HPC systems
HPC_TRANSFER_ENDPOINTS = {
    "polaris": "05d2c76a-e867-4f67-aa57-76edeb0beda0",  # ALCF Eagle filesystem
    "aurora": "05d2c76a-e867-4f67-aa57-76edeb0beda0",   # ALCF Eagle filesystem (shared)
    "theta": "08925f04-569f-11e7-bef8-22000b9a448b",    # ALCF Theta (legacy)
    "frontier": "TODO",  # OLCF Frontier
    "perlmutter": "TODO",  # NERSC Perlmutter
}


class TransferClient:
    """Wrapper around Globus Transfer for file transfers to HPC systems."""

    def __init__(self, destination_endpoint: str | None = None, hpc_system: str | None = None):
        """Initialize with a destination endpoint.

        Args:
            destination_endpoint: Globus collection UUID
            hpc_system: HPC system name to look up endpoint (e.g., 'polaris')
        """
        if destination_endpoint:
            self.destination_endpoint = destination_endpoint
        elif hpc_system and hpc_system in HPC_TRANSFER_ENDPOINTS:
            self.destination_endpoint = HPC_TRANSFER_ENDPOINTS[hpc_system]
        else:
            raise ValueError(
                f"Must provide destination_endpoint or valid hpc_system. "
                f"Known systems: {list(HPC_TRANSFER_ENDPOINTS.keys())}"
            )

        self._transfer_client = None
        self._auth_client = None

    @property
    def transfer_client(self):
        """Lazy-initialize the Globus Transfer client."""
        if self._transfer_client is None:
            import globus_sdk
            from globus_sdk import TransferClient as GlobusTransferClient
            from globus_sdk import NativeAppAuthClient
            from globus_sdk.scopes import TransferScopes

            # Use the same auth flow as Globus Compute
            CLIENT_ID = "4cf29807-cf21-49ec-9443-ff9a3fb9f81c"  # Globus CLI client ID

            client = NativeAppAuthClient(CLIENT_ID)
            client.oauth2_start_flow(
                requested_scopes=[TransferScopes.all],
                refresh_tokens=True
            )

            # Check for cached tokens first
            import os
            token_file = os.path.expanduser("~/.globus_transfer_tokens.json")

            if os.path.exists(token_file):
                import json
                with open(token_file) as f:
                    tokens = json.load(f)
                authorizer = globus_sdk.RefreshTokenAuthorizer(
                    tokens["refresh_token"],
                    client,
                    access_token=tokens.get("access_token"),
                    expires_at=tokens.get("expires_at_seconds"),
                )
            else:
                # Need to authenticate
                authorize_url = client.oauth2_get_authorize_url()
                print(f"\nPlease go to this URL and login:\n{authorize_url}\n")
                auth_code = input("Enter the authorization code: ").strip()

                token_response = client.oauth2_exchange_code_for_tokens(auth_code)
                tokens = token_response.by_resource_server["transfer.api.globus.org"]

                # Cache tokens
                import json
                with open(token_file, "w") as f:
                    json.dump({
                        "access_token": tokens["access_token"],
                        "refresh_token": tokens["refresh_token"],
                        "expires_at_seconds": tokens["expires_at_seconds"],
                    }, f)
                os.chmod(token_file, 0o600)

                authorizer = globus_sdk.RefreshTokenAuthorizer(
                    tokens["refresh_token"],
                    client,
                    access_token=tokens["access_token"],
                    expires_at=tokens["expires_at_seconds"],
                )

            self._transfer_client = GlobusTransferClient(authorizer=authorizer)

        return self._transfer_client

    def get_local_endpoint(self) -> str | None:
        """Try to find a local Globus Connect Personal endpoint."""
        try:
            # Check if Globus Connect Personal is running
            import subprocess
            result = subprocess.run(
                ["globus", "endpoint", "local-id"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def transfer_file(
        self,
        source_endpoint: str,
        source_path: str,
        destination_path: str,
        label: str = "ConfigDiscovery transfer",
        wait: bool = True,
        timeout: int = 600,
    ) -> dict:
        """Transfer a file to the HPC system.

        Args:
            source_endpoint: Source Globus collection UUID
            source_path: Path on source endpoint
            destination_path: Path on destination endpoint
            label: Transfer label
            wait: Whether to wait for transfer completion
            timeout: Timeout in seconds when waiting

        Returns:
            dict with transfer status and task_id
        """
        from globus_sdk import TransferData

        transfer_data = TransferData(
            source_endpoint=source_endpoint,
            destination_endpoint=self.destination_endpoint,
            label=label,
        )
        transfer_data.add_item(source_path, destination_path)

        result = self.transfer_client.submit_transfer(transfer_data)
        task_id = result["task_id"]

        response = {
            "task_id": task_id,
            "status": "submitted",
            "source": f"{source_endpoint}:{source_path}",
            "destination": f"{self.destination_endpoint}:{destination_path}",
        }

        if wait:
            import time
            start_time = time.time()
            while time.time() - start_time < timeout:
                task = self.transfer_client.get_task(task_id)
                status = task["status"]

                if status == "SUCCEEDED":
                    response["status"] = "completed"
                    response["bytes_transferred"] = task.get("bytes_transferred", 0)
                    return response
                elif status == "FAILED":
                    response["status"] = "failed"
                    response["error"] = task.get("nice_status_details", "Transfer failed")
                    return response

                time.sleep(5)

            response["status"] = "timeout"
            response["error"] = f"Transfer did not complete within {timeout} seconds"

        return response

    def get_web_transfer_url(
        self,
        source_endpoint: str | None = None,
        destination_path: str = "~",
    ) -> str:
        """Generate a Globus web app URL for manual transfer.

        Useful when user doesn't have Globus Connect Personal set up.
        """
        base_url = "https://app.globus.org/file-manager"
        params = [
            f"destination_id={self.destination_endpoint}",
            f"destination_path={destination_path}",
        ]
        if source_endpoint:
            params.append(f"origin_id={source_endpoint}")

        return f"{base_url}?{'&'.join(params)}"


def get_transfer_client(
    destination_endpoint: str | None = None,
    hpc_system: str | None = None
) -> TransferClient:
    """Factory function to get a transfer client."""
    return TransferClient(destination_endpoint=destination_endpoint, hpc_system=hpc_system)
