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
    "--force",
    is_flag=True,
    help="Force reinstall even if already installed",
)
@click.option(
    "--check-download",
    is_flag=True,
    help="Only check if manual download is complete, don't install",
)
def install(config_path: str, force: bool, check_download: bool):
    """Install dependencies for a configuration on the remote system.

    This ensures all pip packages, conda packages, and conda environments
    are set up before running the software.

    For software requiring manual download (e.g., NAMD, ORCA), this command
    will first check if the download is complete and provide instructions
    if not.

    Example:
        configdiscovery install configs/polaris/pyscf.yaml
        configdiscovery install configs/polaris/namd.yaml --check-download
    """
    config = HPCConfig.from_yaml_file(config_path)

    console.print(f"[bold]Installing dependencies for {config.name} on {config.hpc_system}[/bold]")
    console.print(f"Endpoint: {config.endpoint_id}")

    from .discovery import ConfigRunner
    runner = ConfigRunner(config)

    # Check for manual download requirements
    manual_dl = config.installation.manual_download
    if manual_dl and manual_dl.required:
        console.print(f"\n[yellow]⚠ {config.name} requires manual download[/yellow]")

        # Check if download is complete
        download_complete = False
        if manual_dl.expected_path:
            try:
                check_result = runner.compute.run_command(
                    f"test -e {manual_dl.expected_path} && echo 'EXISTS' || echo 'MISSING'"
                )
                download_complete = "EXISTS" in check_result.get("output", "")
            except Exception:
                pass

        if download_complete:
            console.print(f"[green]✓ Download verified at {manual_dl.expected_path}[/green]")
        else:
            console.print(f"\n[bold]Download Instructions:[/bold]")
            if manual_dl.license_type:
                console.print(f"  License: {manual_dl.license_type}")
            if manual_dl.url:
                console.print(f"  URL: {manual_dl.url}")
            console.print("")
            for i, instruction in enumerate(manual_dl.instructions, 1):
                console.print(f"  {i}. {instruction}")

            if manual_dl.expected_path:
                console.print(f"\n  Expected location: {manual_dl.expected_path}")

            console.print(f"\n[yellow]After completing the download, run this command again.[/yellow]")

            if check_download:
                sys.exit(0)
            else:
                console.print(f"[red]✗ Cannot proceed without manual download[/red]")
                sys.exit(1)

        if check_download:
            console.print("[green]Download check complete.[/green]")
            sys.exit(0)

    try:
        result = runner.install_dependencies(force=force)
        if result.get("success"):
            console.print("[green]✓ Dependencies installed successfully[/green]")
            if result.get("details"):
                for detail in result["details"]:
                    console.print(f"  - {detail}")
        else:
            console.print(f"[red]✗ Installation failed[/red]")
            console.print(f"  Error: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    except Exception as e:
        console.print(f"[red]✗ Installation failed: {e}[/red]")
        sys.exit(1)


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("local_file", type=click.Path(exists=True), required=False)
@click.option(
    "--web",
    is_flag=True,
    help="Open Globus web app for transfer instead of CLI transfer",
)
@click.option(
    "--source-endpoint", "-s",
    help="Source Globus endpoint UUID (auto-detected if Globus Connect Personal is running)",
)
def transfer(config_path: str, local_file: str | None, web: bool, source_endpoint: str | None):
    """Transfer downloaded software to the HPC system via Globus.

    For software requiring manual download (NAMD, ORCA, etc.), use this
    command to transfer the downloaded file to the HPC system.

    Examples:
        # Open Globus web app for transfer
        configdiscovery transfer configs/polaris/namd.yaml --web

        # Transfer a specific file (requires Globus Connect Personal)
        configdiscovery transfer configs/polaris/namd.yaml ~/Downloads/NAMD_3.0.2.tar.gz
    """
    config = HPCConfig.from_yaml_file(config_path)

    manual_dl = config.installation.manual_download
    if not manual_dl or not manual_dl.required:
        console.print(f"[yellow]{config.name} does not require manual download[/yellow]")
        return

    from .compute import get_transfer_client, HPC_TRANSFER_ENDPOINTS

    # Determine destination endpoint
    dest_endpoint = None
    if manual_dl.globus_transfer and manual_dl.globus_transfer.destination_endpoint:
        dest_endpoint = manual_dl.globus_transfer.destination_endpoint
    elif config.hpc_system in HPC_TRANSFER_ENDPOINTS:
        dest_endpoint = HPC_TRANSFER_ENDPOINTS[config.hpc_system]
    else:
        console.print(f"[red]No Globus endpoint configured for {config.hpc_system}[/red]")
        console.print("Please specify a destination endpoint in the config or use scp.")
        sys.exit(1)

    # Determine destination path
    dest_path = "~"
    if manual_dl.globus_transfer and manual_dl.globus_transfer.destination_path:
        dest_path = manual_dl.globus_transfer.destination_path
    elif manual_dl.expected_path:
        import os
        dest_path = os.path.dirname(manual_dl.expected_path.replace("~", ""))
        if not dest_path:
            dest_path = "~"

    try:
        transfer_client = get_transfer_client(destination_endpoint=dest_endpoint)
    except Exception as e:
        console.print(f"[red]Failed to initialize Globus Transfer: {e}[/red]")
        sys.exit(1)

    if web or not local_file:
        # Generate web URL
        url = transfer_client.get_web_transfer_url(
            source_endpoint=source_endpoint,
            destination_path=dest_path,
        )
        console.print(f"\n[bold]Globus Web Transfer[/bold]")
        console.print(f"\nOpen this URL to transfer files to {config.hpc_system}:")
        console.print(f"[blue]{url}[/blue]\n")

        if manual_dl.globus_transfer and manual_dl.globus_transfer.source_filename:
            console.print(f"Expected filename pattern: {manual_dl.globus_transfer.source_filename}")
        console.print(f"Destination path: {dest_path}")

        if manual_dl.instructions:
            console.print(f"\n[bold]After transfer, complete these steps on {config.hpc_system}:[/bold]")
            # Show post-transfer instructions (skip download steps)
            for i, instr in enumerate(manual_dl.instructions):
                if "extract" in instr.lower() or "mv " in instr.lower() or "verify" in instr.lower():
                    console.print(f"  - {instr}")
    else:
        # Direct transfer
        console.print(f"[bold]Transferring {local_file} to {config.hpc_system}[/bold]")

        # Try to get local endpoint
        if not source_endpoint:
            source_endpoint = transfer_client.get_local_endpoint()
            if not source_endpoint:
                console.print("[red]Could not detect local Globus endpoint.[/red]")
                console.print("Please install Globus Connect Personal or specify --source-endpoint")
                console.print("Or use --web to transfer via the Globus web app")
                sys.exit(1)

        import os
        filename = os.path.basename(local_file)
        source_path = os.path.abspath(local_file)
        destination_full = f"{dest_path}/{filename}"

        console.print(f"Source: {source_endpoint}:{source_path}")
        console.print(f"Destination: {dest_endpoint}:{destination_full}")
        console.print("\nStarting transfer...")

        try:
            result = transfer_client.transfer_file(
                source_endpoint=source_endpoint,
                source_path=source_path,
                destination_path=destination_full,
                label=f"ConfigDiscovery: {config.name}",
                wait=True,
                timeout=600,
            )

            if result["status"] == "completed":
                console.print(f"[green]✓ Transfer completed![/green]")
                if result.get("bytes_transferred"):
                    mb = result["bytes_transferred"] / (1024 * 1024)
                    console.print(f"  Transferred: {mb:.1f} MB")
            else:
                console.print(f"[red]✗ Transfer {result['status']}[/red]")
                if result.get("error"):
                    console.print(f"  Error: {result['error']}")
                sys.exit(1)

        except Exception as e:
            console.print(f"[red]Transfer failed: {e}[/red]")
            sys.exit(1)

        # Show post-transfer instructions
        if manual_dl.instructions:
            console.print(f"\n[bold]Next steps on {config.hpc_system}:[/bold]")
            for instr in manual_dl.instructions:
                if "extract" in instr.lower() or "mv " in instr.lower() or "verify" in instr.lower():
                    console.print(f"  - {instr}")


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


@main.group()
def skill():
    """Manage and execute computational skills."""
    pass


@skill.command(name="list")
@click.option("--category", "-c", help="Filter by category")
@click.option("--tag", "-t", help="Filter by tag")
def skill_list(category: str | None, tag: str | None):
    """List available skills.

    Example:
        configdiscovery skill list
        configdiscovery skill list --category quantum_chemistry
    """
    from .skills import get_registry

    registry = get_registry()
    skills = registry.list_skills(category=category, tag=tag)

    if not skills:
        console.print("No skills found.")
        return

    # Group by category
    by_category: dict[str, list] = {}
    for s in skills:
        if s.category not in by_category:
            by_category[s.category] = []
        by_category[s.category].append(s)

    for cat, cat_skills in sorted(by_category.items()):
        table = Table(title=f"Category: {cat}")
        table.add_column("Skill")
        table.add_column("Description")
        table.add_column("Systems")

        for s in cat_skills:
            systems = ", ".join(s.available_systems()) or "none"
            table.add_row(s.name, s.description[:50] + "..." if len(s.description) > 50 else s.description, systems)

        console.print(table)
        console.print()


@skill.command(name="show")
@click.argument("skill_name")
def skill_show(skill_name: str):
    """Show details of a specific skill.

    Example:
        configdiscovery skill show molecular_energy
    """
    from .skills import get_registry

    registry = get_registry()
    s = registry.get(skill_name)

    if not s:
        console.print(f"[red]Skill '{skill_name}' not found[/red]")
        sys.exit(1)

    console.print(f"[bold green]{s.name}[/bold green]")
    console.print(f"[italic]{s.description}[/italic]\n")
    console.print(f"Category: {s.category}")
    console.print(f"Tags: {', '.join(s.tags)}\n")

    # Inputs
    if s.inputs:
        console.print("[bold]Inputs:[/bold]")
        for name, spec in s.inputs.items():
            req = "[red]*[/red]" if spec.required else ""
            default = f" (default: {spec.default})" if spec.default else ""
            options = f" [{', '.join(spec.options)}]" if spec.options else ""
            unit = f" [{spec.unit}]" if spec.unit else ""
            console.print(f"  {req}{name}: {spec.type.value}{unit}{options}{default}")
            if spec.description:
                console.print(f"      {spec.description}")
        console.print()

    # Outputs
    if s.outputs:
        console.print("[bold]Outputs:[/bold]")
        for name, spec in s.outputs.items():
            unit = f" [{spec.unit}]" if spec.unit else ""
            console.print(f"  {name}: {spec.type.value}{unit}")
            if spec.description:
                console.print(f"      {spec.description}")
        console.print()

    # Implementations
    console.print("[bold]Implementations:[/bold]")
    for impl in s.implementations:
        console.print(f"  - {impl.system}: {impl.config_path}")


@skill.command(name="run")
@click.argument("skill_name")
@click.option("--system", "-s", help="Run on specific system")
@click.option("--param", "-p", multiple=True, help="Parameters in key=value format")
@click.option("--timeout", "-t", default=600, help="Timeout in seconds")
def skill_run(skill_name: str, system: str | None, param: tuple[str, ...], timeout: int):
    """Execute a skill.

    Example:
        configdiscovery skill run molecular_energy -p molecule="H 0 0 0\\nH 0 0 0.74"
        configdiscovery skill run molecular_energy --system polaris -p method=DFT
    """
    from .skills import get_executor

    # Parse parameters
    kwargs = {}
    for p in param:
        if "=" not in p:
            console.print(f"[red]Invalid parameter format: {p} (expected key=value)[/red]")
            sys.exit(1)
        key, value = p.split("=", 1)
        # Try to parse as JSON for complex types
        try:
            import json
            kwargs[key] = json.loads(value)
        except json.JSONDecodeError:
            kwargs[key] = value

    executor = get_executor()
    if system:
        executor.set_preferred_system(system)

    console.print(f"[bold]Running skill: {skill_name}[/bold]")
    if system:
        console.print(f"System: {system}")
    console.print(f"Parameters: {kwargs}\n")

    try:
        result = executor.run(skill_name, system=system, timeout=timeout, **kwargs)

        status = result.get("status", "completed")
        if status == "failed":
            console.print(f"[red]Skill execution failed[/red]")
            console.print(f"Error: {result.get('error', 'Unknown error')}")
            sys.exit(1)

        console.print("[green]Skill completed successfully[/green]\n")
        console.print("[bold]Results:[/bold]")
        for key, value in result.items():
            if isinstance(value, float):
                console.print(f"  {key}: {value:.6f}")
            elif isinstance(value, str) and len(value) > 100:
                console.print(f"  {key}: {value[:100]}...")
            else:
                console.print(f"  {key}: {value}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@skill.command(name="search")
@click.argument("query")
def skill_search(query: str):
    """Search for skills by name, description, or tags.

    Example:
        configdiscovery skill search energy
        configdiscovery skill search quantum
    """
    from .skills import get_registry

    registry = get_registry()
    results = registry.search(query)

    if not results:
        console.print(f"No skills matching '{query}'")
        return

    table = Table(title=f"Skills matching '{query}'")
    table.add_column("Skill")
    table.add_column("Category")
    table.add_column("Description")

    for s in results:
        table.add_row(s.name, s.category, s.description[:60])

    console.print(table)


if __name__ == "__main__":
    main()
