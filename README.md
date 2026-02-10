# ConfigDiscovery

LLM-driven HPC software configuration discovery via Globus Compute.

ConfigDiscovery uses Claude to automatically discover how to install and run scientific software on HPC systems. It probes the remote system through Globus Compute, figures out what's available (modules, conda, compilers), and generates validated YAML configurations.

## Features

- **Automatic Discovery**: Claude explores the HPC system and discovers working configurations
- **Validated Configs**: Each config is tested on the actual system before saving
- **Shareable YAML**: Configurations are stored as documented YAML files
- **Remote Execution**: Run calculations via Globus Compute without SSH access

## Available Configurations

Tested configurations for **ALCF Polaris**:

| Software | Domain | Status |
|----------|--------|--------|
| [Psi4](configs/polaris/psi4.yaml) | Quantum Chemistry | ✅ |
| [NWChem](configs/polaris/nwchem.yaml) | Quantum Chemistry | ✅ |
| [CP2K](configs/polaris/cp2k.yaml) | DFT (GPW method) | ✅ |
| [Quantum ESPRESSO](configs/polaris/quantum-espresso.yaml) | Plane-wave DFT | ✅ |
| [OpenMM](configs/polaris/openmm.yaml) | Biomolecular MD | ✅ |
| [LAMMPS](configs/polaris/lammps.yaml) | Classical MD | ✅ |
| [ASE](configs/polaris/ase.yaml) | Atomistic Simulations | ✅ |

## Quick Start

### Installation

```bash
git clone https://github.com/ianfoster/ConfigDiscovery.git
cd ConfigDiscovery
pip install -e .
```

### Requirements

- Python 3.10+
- `ANTHROPIC_API_KEY` environment variable (for discovery)
- Globus Compute endpoint access

### Environment Variables

```bash
# Required: Set your Globus Compute endpoint ID
export GLOBUS_COMPUTE_ENDPOINT=your-endpoint-uuid-here

# Optional: For discovering new configurations
export ANTHROPIC_API_KEY=your-api-key
```

The `GLOBUS_COMPUTE_ENDPOINT` variable allows you to use configs without hardcoding the endpoint ID. Configs can omit `endpoint_id` and it will be read from this variable.

### Setting Up a Globus Compute Endpoint

To run configurations on an HPC system, you need a Globus Compute endpoint running there.

#### 1. Install Globus Compute on the HPC system

SSH into the HPC system and install the endpoint software:

```bash
# On the HPC system (e.g., Polaris login node)
pip install --user globus-compute-endpoint
```

#### 2. Configure the endpoint

```bash
# Create a new endpoint configuration
globus-compute-endpoint configure my-endpoint

# This creates ~/.globus_compute/my-endpoint/config.yaml
# Edit if needed (e.g., to specify a conda environment or scheduler settings)
```

#### 3. Start the endpoint

```bash
# Start the endpoint
globus-compute-endpoint start my-endpoint

# The endpoint ID will be displayed, e.g.:
# Endpoint ID: 0554c761-5a62-474d-b26e-df7455682bba
```

#### 4. Check endpoint status

```bash
# Check if the endpoint is running
globus-compute-endpoint list

# Expected output:
# +-------------+--------+--------------------------------------+
# | Endpoint    | Status | Endpoint ID                          |
# +-------------+--------+--------------------------------------+
# | my-endpoint | Running| 0554c761-5a62-474d-b26e-df7455682bba |
# +-------------+--------+--------------------------------------+

# For more details
globus-compute-endpoint status my-endpoint
```

#### 5. Stop the endpoint (when done)

```bash
globus-compute-endpoint stop my-endpoint
```

#### Troubleshooting

```bash
# View endpoint logs
globus-compute-endpoint logs my-endpoint

# Restart if having issues
globus-compute-endpoint restart my-endpoint

# Check if endpoint is reachable from your local machine
python -c "from globus_compute_sdk import Client; c = Client(); print(c.get_endpoint_status('YOUR-ENDPOINT-ID'))"
```

For more details, see the [Globus Compute documentation](https://globus-compute.readthedocs.io/).

### Using Existing Configs

```bash
# List available configurations
configdiscovery list

# Show config details
configdiscovery show configs/polaris/psi4.yaml

# Test a configuration (verifies installation + runs calculation)
configdiscovery test configs/polaris/psi4.yaml

# Run with custom parameters
configdiscovery run configs/polaris/psi4.yaml -k method=b3lyp -k basis=6-31g
```

### Discovering New Configurations

```bash
# Discover how to run software on an HPC system
configdiscovery discover gromacs \
  --endpoint 0554c761-5a62-474d-b26e-df7455682bba \
  --system polaris \
  --docs https://manual.gromacs.org/
```

## Testing Workflow

For contributors testing configs:

```bash
# 1. Clone the repo
git clone https://github.com/ianfoster/ConfigDiscovery.git
cd ConfigDiscovery
pip install -e .

# 2. Set your endpoint ID
export GLOBUS_COMPUTE_ENDPOINT=your-endpoint-uuid

# 3. Check endpoint status
python scripts/test_configs.py --status

# 4. Test a specific config
configdiscovery test configs/polaris/psi4.yaml

# 5. Run with different parameters
configdiscovery run configs/polaris/psi4.yaml -k method=mp2 -k basis=cc-pvdz
```

### Batch Testing

Use the test script to test multiple configurations:

```bash
# Test all configs
python scripts/test_configs.py

# Test specific configs
python scripts/test_configs.py configs/polaris/psi4.yaml configs/polaris/ase.yaml

# Skip installation verification (faster)
python scripts/test_configs.py --skip-install

# Check endpoint status only
python scripts/test_configs.py --status
```

### Expected Test Output

```
Testing psi4 on polaris
Endpoint: 0554c761-5a62-474d-b26e-df7455682bba

Step 1: Verifying installation...
✓ Installation verified
  Output: Psi4 version: 1.10

Step 2: Running test calculation...
✓ Test calculation completed successfully
  energy_hartree: -74.9630334000636
  Output directory: /var/tmp/.../psi4_xxx

All tests passed!
```

## Configuration Format

Each YAML config contains:

```yaml
name: psi4
version: "1.10"
description: |
  Psi4 is an open-source quantum chemistry package...

hpc_system: polaris
endpoint_id: 0554c761-5a62-474d-b26e-df7455682bba

environment:
  conda_env: psi4_env
  conda_packages:
    - psi4
    - python=3.11

installation:
  prerequisites:
    - Conda (miniconda or anaconda)
  steps:
    - conda create -n psi4_env python=3.11 -y
    - conda install -c conda-forge psi4 -y
  verification: python -c "import psi4; print(psi4.__version__)"

execution:
  function: |
    def run_psi4_calculation(method="hf", basis="sto-3g"):
        ...
  function_name: run_psi4_calculation

discovery_log:
  discovered_date: "2026-02-10"
  notes: |
    Successfully installed via conda-forge...
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `configdiscovery list` | List all available configs |
| `configdiscovery show <config>` | Display config details |
| `configdiscovery test <config>` | Verify installation and run test |
| `configdiscovery run <config>` | Run with custom parameters |
| `configdiscovery discover <software>` | Discover new configuration |
| `configdiscovery refine <config>` | Improve existing config |

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Claude    │────▶│ ConfigDiscov │────▶│   Globus    │
│   (LLM)     │◀────│    CLI       │◀────│   Compute   │
└─────────────┘     └──────────────┘     └─────────────┘
                           │                    │
                           ▼                    ▼
                    ┌─────────────┐      ┌─────────────┐
                    │    YAML     │      │  HPC System │
                    │   Configs   │      │  (Polaris)  │
                    └─────────────┘      └─────────────┘
```

## Contributing

1. Fork the repository
2. Test existing configs: `configdiscovery test configs/polaris/*.yaml`
3. Discover new configs or improve existing ones
4. Submit a PR with your changes

## License

MIT
