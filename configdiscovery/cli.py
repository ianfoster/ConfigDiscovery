"""CLI interface for ConfigDiscovery."""

import os
import sys

import click
from rich.console import Console
from rich.table import Table

from .discovery import ConfigRunner, DiscoveryEngine
from .github import GitHubConfigStore, RepoConfig
from .schema import HPCConfig


console = Console()


@click.group()
@click.version_option()
def main():
    """ConfigDiscovery: LLM-driven HPC configuration discovery via Globus Compute."""
    pass


@main.command()
@click.argument("software")
@click.option(
    "--endpoint", "-e",
    required=True,
    help="Globus Compute endpoint ID",
)
@click.option(
    "--system", "-s",
    required=True,
    help="HPC system name (e.g., polaris, frontier)",
)
@click.option(
    "--docs", "-d",
    multiple=True,
    help="Documentation URL(s) to consult",
)
@click.option(
    "--context", "-c",
    help="Additional context about requirements",
)
@click.option(
    "--model", "-m",
    default="claude-sonnet-4-20250514",
    help="Claude model to use",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output YAML file path (default: ./<software>-<system>.yaml)",
)
@click.option(
    "--github-repo", "-g",
    help="GitHub repo to save config (format: owner/repo)",
)
@click.option(
    "--create-pr",
    is_flag=True,
    help="Create a PR instead of committing directly",
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    help="Suppress verbose output",
)
def discover(
    software: str,
    endpoint: str,
    system: str,
    docs: tuple[str, ...],
    context: str | None,
    model: str,
    output: str | None,
    github_repo: str | None,
    create_pr: bool,
    quiet: bool,
):
    """Discover how to run SOFTWARE on an HPC system.

    Uses Claude to interactively discover the configuration needed to run
    the specified software on the target HPC system via Globus Compute.

    Example:
        configdiscovery discover cesm2 -e <endpoint-id> -s polaris -d https://cesm.ucar.edu/docs
    """
    console.print(f"[bold]Discovering configuration for {software} on {system}[/bold]")
    console.print(f"Endpoint: {endpoint}")
    if docs:
        console.print(f"Documentation: {', '.join(docs)}")

    engine = DiscoveryEngine(
        endpoint_id=endpoint,
        model=model,
        verbose=not quiet,
    )

    try:
        config = engine.discover(
            software_name=software,
            hpc_system=system,
            docs_urls=list(docs) if docs else None,
            additional_context=context,
        )
    except Exception as e:
        console.print(f"[red]Discovery failed: {e}[/red]")
        sys.exit(1)

    console.print("[green]Discovery complete![/green]")

    # Save locally
    if output is None:
        output = f"{software}-{system}.yaml"

    config.to_yaml_file(output)
    console.print(f"Config saved to: {output}")

    # Save to GitHub if requested
    if github_repo:
        try:
            owner, repo = github_repo.split("/")
            store = GitHubConfigStore(RepoConfig(owner=owner, repo=repo))

            if create_pr:
                url = store.create_config_pr(config)
                console.print(f"[green]PR created: {url}[/green]")
            else:
                url = store.save_config(config)
                console.print(f"[green]Config pushed to: {url}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to save to GitHub: {e}[/red]")
            console.print("Config was saved locally.")


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output file path for refined config",
)
@click.option(
    "--instructions", "-i",
    help="Specific instructions for refinement",
)
@click.option(
    "--endpoint", "-e",
    help="Globus Compute endpoint ID (defaults to config's endpoint)",
)
@click.option(
    "--model", "-m",
    default="claude-sonnet-4-20250514",
    help="Claude model to use",
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    help="Suppress verbose output",
)
def refine(
    config_path: str,
    output: str | None,
    instructions: str | None,
    endpoint: str | None,
    model: str,
    quiet: bool,
):
    """Refine an existing configuration.

    Reads an existing config and uses Claude to improve it, especially
    the execution function. Builds on what was previously learned.

    Example:
        configdiscovery refine cesm-laptop-v2.yaml -o cesm-laptop-v3.yaml

        configdiscovery refine cesm-laptop-v2.yaml -i "Make the function actually run case.build and case.submit"
    """
    console.print(f"[bold]Refining configuration from {config_path}[/bold]")

    existing_config = HPCConfig.from_yaml_file(config_path)
    console.print(f"Loaded config for {existing_config.name} on {existing_config.hpc_system}")

    # Use existing endpoint or override
    endpoint_id = endpoint or existing_config.endpoint_id

    engine = DiscoveryEngine(
        endpoint_id=endpoint_id,
        model=model,
        verbose=not quiet,
    )

    try:
        refined_config = engine.refine(
            existing_config=existing_config,
            instructions=instructions,
        )
    except Exception as e:
        console.print(f"[red]Refinement failed: {e}[/red]")
        sys.exit(1)

    console.print("[green]Refinement complete![/green]")

    # Save refined config
    if output is None:
        # Generate versioned output name
        base = config_path.rsplit(".", 1)[0]
        if base.endswith("-v2"):
            output = base[:-1] + "3.yaml"
        elif base[-2:-1] == "-v" and base[-1].isdigit():
            version = int(base[-1]) + 1
            output = base[:-1] + str(version) + ".yaml"
        else:
            output = base + "-refined.yaml"

    refined_config.to_yaml_file(output)
    console.print(f"Refined config saved to: {output}")


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("args", nargs=-1)
@click.option(
    "--kwarg", "-k",
    multiple=True,
    help="Keyword arguments in key=value format",
)
def run(config_path: str, args: tuple, kwarg: tuple[str, ...]):
    """Run software using a saved configuration.

    Example:
        configdiscovery run cesm2-polaris.yaml input.nc --kwarg output_dir=/results
    """
    console.print(f"[bold]Loading config from {config_path}[/bold]")

    config = HPCConfig.from_yaml_file(config_path)
    console.print(f"Running {config.name} on {config.hpc_system}")

    # Parse kwargs
    kwargs = {}
    for kw in kwarg:
        if "=" not in kw:
            console.print(f"[red]Invalid kwarg format: {kw} (expected key=value)[/red]")
            sys.exit(1)
        key, value = kw.split("=", 1)
        kwargs[key] = value

    runner = ConfigRunner(config)
    try:
        result = runner.run(*args, **kwargs)
        console.print("[green]Execution complete![/green]")
        if result is not None:
            console.print(f"Result: {result}")
    except Exception as e:
        console.print(f"[red]Execution failed: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option(
    "--repo", "-r",
    required=True,
    help="GitHub repo (format: owner/repo)",
)
@click.option(
    "--query", "-q",
    help="Search query",
)
def search(repo: str, query: str | None):
    """Search for configurations in a GitHub repository.

    Example:
        configdiscovery search -r myorg/hpc-configs -q cesm
    """
    try:
        owner, repo_name = repo.split("/")
        store = GitHubConfigStore(RepoConfig(owner=owner, repo=repo_name))

        if query:
            results = store.search_configs(query)
            if not results:
                console.print("No matching configurations found.")
                return

            table = Table(title=f"Configs matching '{query}'")
            table.add_column("Software")
            table.add_column("System")

            for software, system in results:
                table.add_row(software, system)

            console.print(table)
        else:
            index = store.list_configs()
            if not index.configs:
                console.print("No configurations found in repository.")
                return

            table = Table(title="Available Configurations")
            table.add_column("Software")
            table.add_column("Systems")

            for software in index.list_software():
                systems = ", ".join(index.list_systems(software))
                table.add_row(software, systems)

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.argument("software")
@click.argument("system")
@click.option(
    "--repo", "-r",
    required=True,
    help="GitHub repo (format: owner/repo)",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output file path",
)
def fetch(software: str, system: str, repo: str, output: str | None):
    """Fetch a configuration from a GitHub repository.

    Example:
        configdiscovery fetch cesm2 polaris -r myorg/hpc-configs
    """
    try:
        owner, repo_name = repo.split("/")
        store = GitHubConfigStore(RepoConfig(owner=owner, repo=repo_name))

        config = store.get_config(software, system)
        if config is None:
            console.print(f"[red]No config found for {software} on {system}[/red]")
            sys.exit(1)

        if output:
            config.to_yaml_file(output)
            console.print(f"Config saved to: {output}")
        else:
            console.print(config.to_yaml())

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
def show(config_path: str):
    """Display details of a configuration file.

    Example:
        configdiscovery show cesm2-polaris.yaml
    """
    config = HPCConfig.from_yaml_file(config_path)

    console.print(f"[bold]{config.name}[/bold]", style="green")
    if config.version:
        console.print(f"Version: {config.version}")
    if config.description:
        console.print(f"Description: {config.description}")

    console.print(f"\n[bold]HPC System:[/bold] {config.hpc_system}")
    console.print(f"[bold]Endpoint:[/bold] {config.endpoint_id}")

    if config.environment.modules:
        console.print(f"\n[bold]Modules:[/bold] {', '.join(config.environment.modules)}")

    if config.environment.conda_env:
        console.print(f"[bold]Conda env:[/bold] {config.environment.conda_env}")

    if config.environment.env_vars:
        console.print("\n[bold]Environment variables:[/bold]")
        for key, value in config.environment.env_vars.items():
            console.print(f"  {key}={value}")

    if config.installation.steps:
        console.print("\n[bold]Installation steps:[/bold]")
        for step in config.installation.steps:
            console.print(f"  - {step}")

    console.print(f"\n[bold]Resources:[/bold]")
    res = config.execution.resources
    console.print(f"  Nodes: {res.nodes}")
    console.print(f"  Walltime: {res.walltime}")
    if res.queue:
        console.print(f"  Queue: {res.queue}")

    console.print(f"\n[bold]Discovery log:[/bold]")
    console.print(f"  Date: {config.discovery_log.discovered_date}")
    console.print(f"  Attempts: {config.discovery_log.attempts}")
    if config.discovery_log.notes:
        console.print(f"  Notes: {config.discovery_log.notes}")


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--skip-install",
    is_flag=True,
    help="Skip installation verification, just run the test",
)
def test(config_path: str, skip_install: bool):
    """Test a configuration by verifying installation and running a calculation.

    This is useful for validating configs before sharing them. It will:
    1. Verify the software is installed (run verification command)
    2. Run the execution function with default parameters
    3. Report success/failure

    Example:
        configdiscovery test configs/polaris/psi4.yaml
    """
    config = HPCConfig.from_yaml_file(config_path)

    console.print(f"[bold]Testing {config.name} on {config.hpc_system}[/bold]")
    console.print(f"Endpoint: {config.endpoint_id}")

    runner = ConfigRunner(config)

    # Step 1: Verify installation
    if not skip_install and config.installation.verification:
        console.print("\n[bold]Step 1: Verifying installation...[/bold]")
        try:
            result = runner.test_config()
            if result.get("success") or result.get("status") == "completed":
                console.print("[green]✓ Installation verified[/green]")
                if result.get("output"):
                    console.print(f"  Output: {result['output'][:200]}")
            else:
                console.print(f"[red]✗ Installation verification failed[/red]")
                console.print(f"  Error: {result.get('error', 'Unknown error')}")
                sys.exit(1)
        except Exception as e:
            console.print(f"[red]✗ Installation verification failed: {e}[/red]")
            sys.exit(1)
    else:
        console.print("\n[bold]Step 1: Skipping installation verification[/bold]")

    # Step 2: Run with default parameters
    console.print("\n[bold]Step 2: Running test calculation...[/bold]")
    try:
        result = runner.run()
        status = result.get("status", "unknown")

        if status == "completed":
            console.print("[green]✓ Test calculation completed successfully[/green]")

            # Show key results
            for key in ["energy_hartree", "total_energy_Ry", "total_energy_hartree",
                       "scf_energy_hartree", "optimized_energy", "initial_pe_kj_mol"]:
                if key in result:
                    console.print(f"  {key}: {result[key]}")

            if result.get("output_dir"):
                console.print(f"  Output directory: {result['output_dir']}")
        else:
            console.print(f"[red]✗ Test calculation failed[/red]")
            if result.get("error"):
                console.print(f"  Error: {result['error'][:500]}")
            sys.exit(1)

    except Exception as e:
        console.print(f"[red]✗ Test calculation failed: {e}[/red]")
        sys.exit(1)

    console.print("\n[bold green]All tests passed![/bold green]")


@main.command(name="list")
@click.option(
    "--path", "-p",
    default="configs",
    help="Path to configs directory",
)
def list_configs(path: str):
    """List all available configurations.

    Example:
        configdiscovery list
        configdiscovery list -p /path/to/configs
    """
    import glob
    from pathlib import Path

    configs_path = Path(path)
    if not configs_path.exists():
        console.print(f"[red]Config directory not found: {path}[/red]")
        sys.exit(1)

    # Find all YAML files
    yaml_files = list(configs_path.glob("**/*.yaml")) + list(configs_path.glob("**/*.yml"))

    if not yaml_files:
        console.print("No configuration files found.")
        return

    # Group by system
    configs_by_system = {}
    for f in yaml_files:
        try:
            config = HPCConfig.from_yaml_file(str(f))
            system = config.hpc_system or "unknown"
            if system not in configs_by_system:
                configs_by_system[system] = []
            configs_by_system[system].append({
                "name": config.name,
                "version": config.version or "-",
                "path": str(f.relative_to(configs_path)),
                "description": (config.description or "")[:60] + "..." if config.description and len(config.description) > 60 else config.description or "-"
            })
        except Exception:
            continue

    for system, configs in sorted(configs_by_system.items()):
        table = Table(title=f"Configs for {system}")
        table.add_column("Name")
        table.add_column("Version")
        table.add_column("Path")

        for c in sorted(configs, key=lambda x: x["name"]):
            table.add_row(c["name"], c["version"], c["path"])

        console.print(table)
        console.print()


if __name__ == "__main__":
    main()
