#!/usr/bin/env python3
"""Test all configurations against a Globus Compute endpoint.

Usage:
    # Set endpoint ID via environment variable
    export GLOBUS_COMPUTE_ENDPOINT=0554c761-5a62-474d-b26e-df7455682bba

    # Test all configs
    python scripts/test_configs.py

    # Test specific configs
    python scripts/test_configs.py configs/polaris/psi4.yaml configs/polaris/ase.yaml

    # Test with explicit endpoint
    python scripts/test_configs.py --endpoint 0554c761-5a62-474d-b26e-df7455682bba

    # Skip installation verification (just run test calculation)
    python scripts/test_configs.py --skip-install

    # Check endpoint status only
    python scripts/test_configs.py --status
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from globus_compute_sdk import Client
from rich.console import Console
from rich.table import Table

from configdiscovery.discovery import ConfigRunner
from configdiscovery.schema import HPCConfig

console = Console()


def get_endpoint_id(args_endpoint: str | None = None) -> str:
    """Get endpoint ID from args or environment variable."""
    endpoint = args_endpoint or os.environ.get("GLOBUS_COMPUTE_ENDPOINT")
    if not endpoint:
        console.print("[red]Error: No endpoint ID provided.[/red]")
        console.print("Set GLOBUS_COMPUTE_ENDPOINT environment variable or use --endpoint flag")
        sys.exit(1)
    return endpoint


def check_endpoint_status(endpoint_id: str) -> dict:
    """Check if endpoint is online and return status details."""
    try:
        client = Client()
        status = client.get_endpoint_status(endpoint_id)
        return status
    except Exception as e:
        return {"status": "error", "error": str(e)}


def print_endpoint_status(endpoint_id: str):
    """Print formatted endpoint status."""
    console.print(f"\n[bold]Checking endpoint: {endpoint_id}[/bold]")

    status = check_endpoint_status(endpoint_id)

    if status.get("status") == "online":
        details = status.get("details", {})
        table = Table(title="Endpoint Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Status", "[bold green]online[/bold green]")
        table.add_row("Total Workers", str(details.get("total_workers", "?")))
        table.add_row("Idle Workers", str(details.get("idle_workers", "?")))
        table.add_row("Pending Tasks", str(details.get("pending_tasks", "?")))
        table.add_row("Total Tasks Run", str(details.get("total_tasks", "?")))
        table.add_row("Queue", details.get("queue", "?"))
        table.add_row("Provider", details.get("provider_type", "?"))

        console.print(table)
    elif status.get("status") == "offline":
        console.print("[yellow]Endpoint is offline[/yellow]")
        console.print("Start it on the HPC system with: globus-compute-endpoint start <name>")
    else:
        console.print(f"[red]Error checking endpoint: {status.get('error', 'Unknown error')}[/red]")


def find_all_configs(config_dir: str = "configs") -> list[Path]:
    """Find all YAML config files."""
    config_path = Path(config_dir)
    if not config_path.exists():
        return []
    return sorted(config_path.glob("**/*.yaml"))


def test_config(config_path: Path, endpoint_id: str, skip_install: bool = False) -> dict:
    """Test a single configuration.

    Returns dict with 'success', 'install_ok', 'run_ok', 'error' keys.
    """
    result = {
        "config": str(config_path),
        "success": False,
        "install_ok": None,
        "run_ok": None,
        "error": None,
    }

    try:
        config = HPCConfig.from_yaml_file(str(config_path))

        # Override endpoint if provided
        if endpoint_id:
            config.endpoint_id = endpoint_id

        runner = ConfigRunner(config)

        # Step 1: Verify installation
        if not skip_install and config.installation.verification:
            console.print("  Verifying installation...", end=" ")
            try:
                install_result = runner.test_config()
                if install_result.get("success") or install_result.get("status") == "completed":
                    console.print("[green]OK[/green]")
                    result["install_ok"] = True
                else:
                    console.print(f"[red]FAILED[/red]")
                    result["install_ok"] = False
                    result["error"] = install_result.get("error", "Installation verification failed")
                    return result
            except Exception as e:
                console.print(f"[red]FAILED[/red]")
                result["install_ok"] = False
                result["error"] = str(e)
                return result
        else:
            result["install_ok"] = True  # Skipped

        # Step 2: Run test calculation
        console.print("  Running test calculation...", end=" ")
        try:
            run_result = runner.run()
            status = run_result.get("status", "unknown")

            if status == "completed":
                console.print("[green]OK[/green]")
                result["run_ok"] = True
                result["success"] = True

                # Show key results
                for key in ["energy_hartree", "total_energy_Ry", "optimized_energy", "initial_pe_kj_mol"]:
                    if key in run_result:
                        console.print(f"    {key}: {run_result[key]}")
            else:
                console.print(f"[red]FAILED[/red]")
                result["run_ok"] = False
                result["error"] = run_result.get("error", "Test calculation failed")
        except Exception as e:
            console.print(f"[red]FAILED[/red]")
            result["run_ok"] = False
            result["error"] = str(e)

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test ConfigDiscovery configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "configs",
        nargs="*",
        help="Config files to test (default: all in configs/)",
    )
    parser.add_argument(
        "--endpoint", "-e",
        help="Globus Compute endpoint ID (or set GLOBUS_COMPUTE_ENDPOINT)",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip installation verification",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Just check endpoint status, don't run tests",
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Directory containing configs (default: configs/)",
    )

    args = parser.parse_args()

    # Get endpoint ID
    endpoint_id = get_endpoint_id(args.endpoint)

    # Status check only
    if args.status:
        print_endpoint_status(endpoint_id)
        return

    # Check endpoint is online first
    console.print(f"[bold]Endpoint:[/bold] {endpoint_id}")
    status = check_endpoint_status(endpoint_id)
    if status.get("status") != "online":
        console.print(f"[red]Endpoint is not online: {status.get('status', 'unknown')}[/red]")
        if status.get("error"):
            console.print(f"Error: {status['error']}")
        sys.exit(1)
    console.print("[green]Endpoint is online[/green]\n")

    # Find configs to test
    if args.configs:
        config_files = [Path(c) for c in args.configs]
    else:
        config_files = find_all_configs(args.config_dir)

    if not config_files:
        console.print("[yellow]No config files found[/yellow]")
        sys.exit(1)

    console.print(f"[bold]Testing {len(config_files)} configuration(s)...[/bold]\n")

    # Run tests
    results = []
    for config_path in config_files:
        console.print(f"[bold cyan]{config_path.stem}[/bold cyan] ({config_path})")
        result = test_config(config_path, endpoint_id, args.skip_install)
        results.append(result)
        console.print()

    # Summary table
    console.print("\n[bold]Summary[/bold]")
    table = Table()
    table.add_column("Config")
    table.add_column("Install")
    table.add_column("Run")
    table.add_column("Status")

    passed = 0
    failed = 0

    for r in results:
        name = Path(r["config"]).stem
        install = "[green]OK[/green]" if r["install_ok"] else "[red]FAIL[/red]" if r["install_ok"] is False else "-"
        run = "[green]OK[/green]" if r["run_ok"] else "[red]FAIL[/red]" if r["run_ok"] is False else "-"

        if r["success"]:
            status = "[bold green]PASS[/bold green]"
            passed += 1
        else:
            status = "[bold red]FAIL[/bold red]"
            failed += 1

        table.add_row(name, install, run, status)

    console.print(table)
    console.print(f"\n[bold]Results: {passed} passed, {failed} failed[/bold]")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
