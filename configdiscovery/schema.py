"""Pydantic models for HPC configuration schema."""

from datetime import date
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Environment(BaseModel):
    """Environment setup configuration."""

    modules: list[str] = Field(default_factory=list, description="HPC modules to load")
    conda_env: str | None = Field(default=None, description="Conda environment name")
    conda_packages: list[str] = Field(
        default_factory=list, description="Packages to install in conda env"
    )
    pip_packages: list[str] = Field(
        default_factory=list, description="Packages to install via pip"
    )
    env_vars: dict[str, str] = Field(
        default_factory=dict, description="Environment variables to set"
    )
    setup_commands: list[str] = Field(
        default_factory=list, description="Additional setup commands to run"
    )


class Installation(BaseModel):
    """Software installation configuration."""

    steps: list[str] = Field(
        default_factory=list, description="Installation commands"
    )
    verification: str | None = Field(
        default=None, description="Command to verify installation succeeded"
    )
    skip_if_exists: str | None = Field(
        default=None, description="Path to check - skip installation if exists"
    )


class Resources(BaseModel):
    """HPC resource requirements."""

    nodes: int = Field(default=1, description="Number of nodes")
    cores_per_node: int | None = Field(default=None, description="Cores per node")
    memory_gb: int | None = Field(default=None, description="Memory in GB")
    gpus: int | None = Field(default=None, description="Number of GPUs")
    walltime: str = Field(default="01:00:00", description="Walltime limit (HH:MM:SS)")
    queue: str | None = Field(default=None, description="Queue/partition name")
    account: str | None = Field(default=None, description="Allocation account")


class Execution(BaseModel):
    """Execution configuration."""

    function: str = Field(
        ..., description="Python function code to execute the software"
    )
    function_name: str = Field(
        default="run", description="Name of the main function to call"
    )
    resources: Resources = Field(default_factory=Resources)
    pre_commands: list[str] = Field(
        default_factory=list, description="Commands to run before execution"
    )
    post_commands: list[str] = Field(
        default_factory=list, description="Commands to run after execution"
    )


class DiscoveryLog(BaseModel):
    """Metadata about the discovery process."""

    model_config = ConfigDict(populate_by_name=True)

    discovered_date: date = Field(default_factory=date.today, serialization_alias="date")
    docs_consulted: list[str] = Field(
        default_factory=list, description="Documentation URLs that were referenced"
    )
    attempts: int = Field(default=1, description="Number of attempts before success")
    notes: str | None = Field(
        default=None, description="Notes about what was learned during discovery"
    )
    claude_model: str | None = Field(
        default=None, description="Claude model used for discovery"
    )


class HPCConfig(BaseModel):
    """Complete HPC configuration for running software."""

    name: str = Field(..., description="Software/tool name")
    version: str | None = Field(default=None, description="Software version")
    description: str | None = Field(default=None, description="What this config does")
    hpc_system: str = Field(..., description="HPC system name (e.g., polaris, frontier)")
    endpoint_id: str = Field(..., description="Globus Compute endpoint UUID")

    environment: Environment = Field(default_factory=Environment)
    installation: Installation = Field(default_factory=Installation)
    execution: Execution

    discovery_log: DiscoveryLog = Field(default_factory=DiscoveryLog)

    tags: list[str] = Field(
        default_factory=list, description="Tags for categorization"
    )

    def to_yaml(self) -> str:
        """Serialize config to YAML string."""
        data = self.model_dump(mode="json", exclude_none=True)
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "HPCConfig":
        """Parse config from YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.model_validate(data)

    @classmethod
    def from_yaml_file(cls, path: str) -> "HPCConfig":
        """Load config from YAML file."""
        with open(path) as f:
            return cls.from_yaml(f.read())

    def to_yaml_file(self, path: str) -> None:
        """Save config to YAML file."""
        with open(path, "w") as f:
            f.write(self.to_yaml())


class ConfigIndex(BaseModel):
    """Index of available configurations."""

    configs: dict[str, dict[str, str]] = Field(
        default_factory=dict,
        description="Nested dict: software -> system -> config file path",
    )

    def add_config(self, config: HPCConfig, file_path: str) -> None:
        """Add a config to the index."""
        if config.name not in self.configs:
            self.configs[config.name] = {}
        self.configs[config.name][config.hpc_system] = file_path

    def get_config_path(self, software: str, system: str) -> str | None:
        """Get path to a config file."""
        return self.configs.get(software, {}).get(system)

    def list_software(self) -> list[str]:
        """List all software with configurations."""
        return list(self.configs.keys())

    def list_systems(self, software: str) -> list[str]:
        """List all systems with configs for given software."""
        return list(self.configs.get(software, {}).keys())
