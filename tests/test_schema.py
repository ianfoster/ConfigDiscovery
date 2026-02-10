"""Tests for the schema module."""

import pytest
from configdiscovery.schema import (
    ConfigIndex,
    DiscoveryLog,
    Environment,
    Execution,
    HPCConfig,
    Installation,
    Resources,
)


def test_environment_defaults():
    """Test Environment has sensible defaults."""
    env = Environment()
    assert env.modules == []
    assert env.conda_env is None
    assert env.env_vars == {}


def test_resources_defaults():
    """Test Resources has sensible defaults."""
    res = Resources()
    assert res.nodes == 1
    assert res.walltime == "01:00:00"


def test_hpc_config_creation():
    """Test creating a basic HPCConfig."""
    config = HPCConfig(
        name="test-software",
        hpc_system="test-cluster",
        endpoint_id="test-endpoint-123",
        execution=Execution(
            function="def run(): pass",
        ),
    )

    assert config.name == "test-software"
    assert config.hpc_system == "test-cluster"
    assert config.endpoint_id == "test-endpoint-123"


def test_hpc_config_yaml_roundtrip():
    """Test YAML serialization and deserialization."""
    config = HPCConfig(
        name="cesm2",
        version="2.1.3",
        hpc_system="polaris",
        endpoint_id="abc-123",
        environment=Environment(
            modules=["python/3.11", "netcdf"],
            conda_env="cesm-env",
            env_vars={"CESM_ROOT": "/path/to/cesm"},
        ),
        installation=Installation(
            steps=["pip install cesm-tools"],
            verification="cesm --version",
        ),
        execution=Execution(
            function="def run(x): return x * 2",
            resources=Resources(nodes=4, walltime="02:00:00"),
        ),
        discovery_log=DiscoveryLog(
            docs_consulted=["https://example.com/docs"],
            attempts=3,
            notes="Needed specific module version",
        ),
    )

    yaml_str = config.to_yaml()
    assert "cesm2" in yaml_str
    assert "polaris" in yaml_str
    assert "python/3.11" in yaml_str

    # Roundtrip
    loaded = HPCConfig.from_yaml(yaml_str)
    assert loaded.name == config.name
    assert loaded.version == config.version
    assert loaded.environment.modules == config.environment.modules
    assert loaded.execution.resources.nodes == 4


def test_config_index():
    """Test ConfigIndex operations."""
    index = ConfigIndex()

    config1 = HPCConfig(
        name="app1",
        hpc_system="cluster1",
        endpoint_id="ep1",
        execution=Execution(function="def run(): pass"),
    )
    config2 = HPCConfig(
        name="app1",
        hpc_system="cluster2",
        endpoint_id="ep2",
        execution=Execution(function="def run(): pass"),
    )

    index.add_config(config1, "configs/app1/cluster1.yaml")
    index.add_config(config2, "configs/app1/cluster2.yaml")

    assert index.list_software() == ["app1"]
    assert set(index.list_systems("app1")) == {"cluster1", "cluster2"}
    assert index.get_config_path("app1", "cluster1") == "configs/app1/cluster1.yaml"
    assert index.get_config_path("app1", "nonexistent") is None
