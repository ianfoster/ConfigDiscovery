# ConfigDiscovery

LLM-driven HPC software configuration discovery via Globus Compute.

ConfigDiscovery uses Claude to automatically discover how to install and run scientific software on HPC systems. It probes the remote system through Globus Compute, figures out what's available (modules, conda, compilers), and generates validated YAML configurations.

## Table of Contents

- [Features](#features)
- [Available Configurations](#available-configurations)
- [Quick Start](#quick-start)
- [Skills](#skills)
- [Multi-Fidelity Pipeline](#multi-fidelity-pipeline)
- [CLI Commands](#cli-commands)
- [Configuration Format](#configuration-format)
- [Architecture](#architecture)
- [Contributing](#contributing)

## Features

- **Automatic Discovery**: Claude explores the HPC system and discovers working configurations
- **Validated Configs**: Each config is tested on the actual system before saving
- **Shareable YAML**: Configurations are stored as documented YAML files
- **Remote Execution**: Run calculations via Globus Compute without SSH access

## Available Configurations

Legend: ✅ = Tested (full calculation) | ✓ = Installed (import verified) | - = Not available

| Software | Domain | Polaris | Aurora |
|----------|--------|:-------:|:------:|
| Psi4 | Quantum Chemistry | [✅](configs/polaris/psi4.yaml) | [✅](configs/aurora/psi4.yaml) |
| PySCF | Quantum Chemistry | [✅](configs/polaris/pyscf.yaml) | [✅](configs/aurora/pyscf.yaml) |
| NWChem | Quantum Chemistry | [✅](configs/polaris/nwchem.yaml) | [✅](configs/aurora/nwchem.yaml) |
| ORCA | Quantum Chemistry | [✅](configs/polaris/orca.yaml) | - |
| CP2K | DFT (GPW method) | [✅](configs/polaris/cp2k.yaml) | [✅](configs/aurora/cp2k.yaml) |
| GPAW | DFT (PAW method) | [✅](configs/polaris/gpaw.yaml) | [✅](configs/aurora/gpaw.yaml) |
| Quantum ESPRESSO | Plane-wave DFT | [✅](configs/polaris/quantum-espresso.yaml) | [✅](configs/aurora/quantum-espresso.yaml) |
| Siesta | DFT (numerical orbitals) | [✅](configs/polaris/siesta.yaml) | [✅](configs/aurora/siesta.yaml) |
| Abinit | DFT (plane-wave) | [✅](configs/polaris/abinit.yaml) | [✅](configs/aurora/abinit.yaml) |
| DFTB+ | Tight-binding DFT | [✅](configs/polaris/dftbplus.yaml) | [✅](configs/aurora/dftbplus.yaml) |
| xtb | Semi-empirical QM | [✅](configs/polaris/xtb.yaml) | [✅](configs/aurora/xtb.yaml) |
| OpenMM | Biomolecular MD | [✅](configs/polaris/openmm.yaml) | [✅](configs/aurora/openmm.yaml) |
| GROMACS | Classical MD | [✅](configs/polaris/gromacs.yaml) | [✅](configs/aurora/gromacs.yaml) |
| LAMMPS | Classical MD | [✅](configs/polaris/lammps.yaml) | [✅](configs/aurora/lammps.yaml) |
| NAMD | Biomolecular MD | [✅](configs/polaris/namd.yaml) | - |
| AmberTools | Biomolecular Tools | [✅](configs/polaris/ambertools.yaml) | [✅](configs/aurora/ambertools.yaml) |
| ASE | Atomistic Simulations | [✅](configs/polaris/ase.yaml) | [✅](configs/aurora/ase.yaml) |
| MDAnalysis | Trajectory Analysis | [✅](configs/polaris/mdanalysis.yaml) | [✅](configs/aurora/mdanalysis.yaml) |
| Phonopy | Phonon Calculations | [✅](configs/polaris/phonopy.yaml) | [✅](configs/aurora/phonopy.yaml) |
| SchNetPack | ML Potentials | [✅](configs/polaris/schnetpack.yaml) | [✅](configs/aurora/schnetpack.yaml) |
| DeePMD-kit | ML Potentials | [✅](configs/polaris/deepmd-kit.yaml) | [✅](configs/aurora/deepmd-kit.yaml) |
| MACE | ML Potentials | [✅](configs/polaris/mace.yaml) | - |
| RDKit | Cheminformatics | [✅](configs/polaris/rdkit.yaml) | [✅](configs/aurora/rdkit.yaml) |
| Open Babel | Molecule Conversion | [✅](configs/polaris/openbabel.yaml) | [✅](configs/aurora/openbabel.yaml) |
| PyMatGen | Materials Analysis | [✅](configs/polaris/pymatgen.yaml) | [✅](configs/aurora/pymatgen.yaml) |
| OpenFOAM | CFD Simulations | [✅](configs/polaris/openfoam.yaml) | [✅](configs/aurora/openfoam.yaml) |

**Polaris**: 26 packages (all tested) | **Aurora**: 23 packages (all tested)

Notes:
- ORCA and NAMD require manual download and cannot be installed via conda
- MACE failed to install on Aurora due to disk quota limits

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

#### Aurora-Specific Setup

Aurora uses a PBS scheduler with specific configuration requirements:

```yaml
# Example Aurora endpoint config (~/.globus_compute/aurora_endpoint/config.yaml)
engine:
  type: GlobusComputeEngine
  provider:
    type: PBSProProvider
    account: YourAllocation
    queue: debug
    cpus_per_node: 208
    select_options: system=sunspot
    scheduler_options: "#PBS -l filesystems=home:flare"
    worker_init: |
      source /opt/aurora/25.190.0/oneapi/intel-conda-miniforge/etc/profile.d/conda.sh
      conda activate base
    launcher:
      type: MpiExecLauncher
      bind_cmd: --cpu-bind
      overrides: --depth=208 --ppn=1
    walltime: 01:00:00
    nodes_per_block: 1
    init_blocks: 0
    min_blocks: 0
    max_blocks: 1
```

Key Aurora settings:
- **queue**: Use `debug` for testing, `workq` for production
- **filesystems**: `home:flare` (required)
- **worker_init**: Source conda from `/opt/aurora/25.190.0/oneapi/intel-conda-miniforge/`

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

### Config Commands

| Command | Description |
|---------|-------------|
| `configdiscovery list` | List all available configs |
| `configdiscovery show <config>` | Display config details |
| `configdiscovery test <config>` | Verify installation and run test |
| `configdiscovery run <config>` | Run with custom parameters |
| `configdiscovery discover <software>` | Discover new configuration |
| `configdiscovery refine <config>` | Improve existing config |

### Skill Commands

| Command | Description |
|---------|-------------|
| `configdiscovery skill list` | List all available skills |
| `configdiscovery skill show <skill>` | Display skill details and implementations |
| `configdiscovery skill run <skill>` | Run skill (auto-selects implementation) |
| `configdiscovery skill search <query>` | Search skills by name/tag |

## Skills

Skills are abstract computational capabilities that can have multiple implementations across different HPC systems. This allows users to express intent ("compute molecular energy") without specifying how or where it runs.

### Available Skills

| Skill | Description | Implementations |
|-------|-------------|-----------------|
| `molecular_energy` | Compute quantum mechanical energy | Psi4, PySCF, NWChem, xtb, CP2K |
| `geometry_optimization` | Optimize molecular geometry | Psi4, ASE, xtb |
| `molecular_dynamics` | Run classical MD simulations | LAMMPS, GROMACS, OpenMM |
| `biomolecular_md` | Biomolecular MD with force fields | OpenMM, GROMACS, NAMD |
| `trajectory_analysis` | Analyze MD trajectories | MDAnalysis |
| `train_ml_potential` | Train machine learning potentials | SchNetPack, DeePMD-kit |
| `ml_potential_predict` | Predict with ML potentials | SchNetPack, DeePMD-kit |
| `phonon_calculation` | Compute phonon properties | Phonopy |
| `periodic_dft` | Periodic DFT calculations | Quantum ESPRESSO, CP2K, GPAW |
| `cfd_simulation` | Computational fluid dynamics | OpenFOAM |

### Using Skills

```bash
# List available skills
configdiscovery skill list

# Show skill details
configdiscovery skill show molecular_energy

# Run a skill (auto-selects best implementation)
configdiscovery skill run molecular_energy \
  --molecule water.xyz \
  --method HF \
  --basis sto-3g

# Run on a specific system
configdiscovery skill run molecular_energy \
  --molecule water.xyz \
  --system aurora

# Search for skills by capability
configdiscovery skill search "energy"
```

### Skill vs Config

- **Configs** are low-level: specific software + specific HPC system
- **Skills** are high-level: abstract capability with multiple implementations

```
User Request: "compute energy of this molecule"
        │
        ▼
┌─────────────────┐
│     Skill       │  molecular_energy
│  (abstract)     │
└────────┬────────┘
         │ selects best implementation
         ▼
┌─────────────────┐
│    Config       │  configs/aurora/pyscf.yaml
│  (concrete)     │
└────────┬────────┘
         │ executes on
         ▼
┌─────────────────┐
│   HPC System    │  Aurora via Globus Compute
└─────────────────┘
```

## Multi-Fidelity Pipeline

The repository includes a multi-fidelity molecular simulation pipeline that chains multiple computational steps:

```bash
# Run the pipeline on Polaris
python scripts/multi_fidelity_pipeline.py --molecule ethanol --n-conformers 20

# Run on Aurora
python scripts/multi_fidelity_pipeline_aurora.py --molecule ethanol --n-conformers 20
```

Pipeline steps:
1. **xtb** - Generate conformers and compute semi-empirical energies
2. **PySCF** - Compute accurate ab initio energy on lowest-energy conformer
3. **SchNetPack** - Train ML potential on conformer dataset
4. **xtb MD** - Run molecular dynamics using the trained model
5. **MDAnalysis** - Analyze trajectory (RMSD, Rg, fluctuations)

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
                    │   Configs   │      │ (Polaris/   │
                    └─────────────┘      │  Aurora)    │
                                         └─────────────┘
```

## Contributing

1. Fork the repository
2. Test existing configs: `configdiscovery test configs/polaris/*.yaml`
3. Discover new configs or improve existing ones
4. Submit a PR with your changes

## License

MIT
