# ConfigDiscovery Update - February 2026

## Overview

This document summarizes the major developments in ConfigDiscovery since the initial release, including expanded HPC system support, the new Skills abstraction layer, and lessons learned from deploying across multiple supercomputers.

## New HPC System: Aurora

ConfigDiscovery now supports two DOE Leadership Computing Facilities:

| System | Location | Architecture | Status |
|--------|----------|--------------|--------|
| **Polaris** | ALCF (Argonne) | NVIDIA A100 GPUs | 26 packages |
| **Aurora** | ALCF (Argonne) | Intel GPUs (PVC) | 23 packages |

### Aurora-Specific Challenges

Aurora is an Intel GPU-based exascale system with unique requirements:

1. **Conda Environment**: Aurora uses Intel's conda distribution at `/opt/aurora/25.190.0/oneapi/intel-conda-miniforge/`

2. **PBS Scheduling**: The system uses PBS Pro with specific options:
   ```yaml
   scheduler_options: "#PBS -l filesystems=home:flare"
   select_options: system=sunspot
   ```

3. **Worker Spin-up Time**: PBS jobs on Aurora can take 5-10 minutes to start, requiring client-side timeouts of 600+ seconds

### Packages Successfully Deployed on Aurora

All 23 packages passed full execution tests:

| Category | Packages |
|----------|----------|
| Quantum Chemistry | Psi4, PySCF, NWChem |
| DFT Codes | CP2K, GPAW, Quantum ESPRESSO, Siesta, Abinit, DFTB+ |
| Semi-empirical | xtb |
| Molecular Dynamics | OpenMM, GROMACS, LAMMPS |
| Analysis Tools | ASE, MDAnalysis, AmberTools |
| ML Potentials | SchNetPack, DeePMD-kit |
| Materials Science | Phonopy, PyMatGen |
| Cheminformatics | RDKit, Open Babel |
| CFD | OpenFOAM |

**Note**: MACE failed to install on Aurora due to disk quota limits during pip installation.

## Skills Abstraction Layer

A major new feature is the **Skills** system, which provides an abstraction over low-level configurations.

### Motivation

Users want to express computational intent ("compute molecular energy") without specifying implementation details (which code, which system, which parameters). Skills bridge this gap.

### Architecture

```
User Request: "compute energy of this molecule"
        |
        v
+------------------+
|      Skill       |  molecular_energy
|   (abstract)     |
+--------+---------+
         | selects best implementation
         v
+------------------+
|     Config       |  configs/aurora/pyscf.yaml
|   (concrete)     |
+--------+---------+
         | executes on
         v
+------------------+
|   HPC System     |  Aurora via Globus Compute
+------------------+
```

### Available Skills

| Skill | Description | Implementations |
|-------|-------------|-----------------|
| `molecular_energy` | Quantum mechanical energy | Psi4, PySCF, NWChem, xtb, GPAW |
| `geometry_optimization` | Optimize molecular geometry | xtb, ASE |
| `molecular_dynamics` | Classical/QM MD simulations | LAMMPS, GROMACS, OpenMM, xtb |
| `biomolecular_md` | Protein/biomolecule MD | OpenMM, GROMACS, NAMD |
| `trajectory_analysis` | Analyze MD trajectories | MDAnalysis |
| `train_ml_potential` | Train ML interatomic potentials | SchNetPack, DeePMD-kit |
| `ml_potential_predict` | Inference with ML potentials | SchNetPack, DeePMD-kit |
| `phonon_calculation` | Phonon properties | Phonopy |
| `periodic_dft` | Periodic DFT calculations | Quantum ESPRESSO, CP2K, GPAW |
| `cfd_simulation` | Computational fluid dynamics | OpenFOAM |

### Typed Inputs/Outputs

Skills define typed parameters with units and constraints:

```python
inputs={
    "molecule": ParameterSpec(
        type=DataType.XYZ,
        description="Molecular structure in XYZ format",
        required=True
    ),
    "method": ParameterSpec(
        type=DataType.STRING,
        description="Quantum chemistry method",
        default="HF",
        options=["HF", "DFT", "B3LYP", "CCSD", "MP2", "GFN2-xTB"]
    ),
}
outputs={
    "energy": ParameterSpec(
        type=DataType.FLOAT,
        description="Total energy",
        unit="hartree"
    ),
}
```

### Automatic System Selection

The `SkillExecutor` automatically selects the best available implementation:

```python
from configdiscovery.skills import run_skill

# Automatically picks best available implementation
result = run_skill("molecular_energy", molecule=xyz_string, method="HF")

# Or specify a system
result = run_skill("molecular_energy", molecule=xyz_string, system="aurora")
```

## Testing Methodology

### Two-Phase Testing

Each configuration undergoes two phases:

1. **Installation Verification**: Import the package and check version
2. **Execution Test**: Run an actual calculation and verify results

### Test Results Summary

| System | Total Packages | Fully Tested | Requires Manual Setup |
|--------|---------------|--------------|----------------------|
| Polaris | 26 | 24 | 2 (ORCA, NAMD) |
| Aurora | 23 | 23 | 0 |

**Manual Setup Required**:
- **ORCA**: Requires registration and manual download from ORCA forums
- **NAMD**: Requires registration and manual download from UIUC

## Lessons Learned

### 1. Globus Compute Timeouts

**Problem**: Initial tests failed with "Execution timed out" errors on Aurora.

**Root Cause**: Client-side timeouts were set to 60-120 seconds, but PBS worker jobs need 5-10 minutes to spin up when no workers are idle.

**Solution**: Use 600-second (10 minute) timeouts for all remote executions:
```python
result = compute.run_function(func, timeout=600, **kwargs)
```

**Lesson**: Don't confuse infrastructure delays (PBS queue time) with execution failures. The Globus Compute endpoint can be "running" while PBS workers are still being scheduled.

### 2. Conda Environment Activation

**Problem**: Packages installed in conda environments couldn't be imported by Globus Compute workers.

**Root Cause**: Workers run in their own environment and don't inherit shell configuration.

**Solution**: Explicitly activate conda in execution functions:
```python
conda_cmd = "source /path/to/conda.sh && conda activate myenv && "
subprocess.run(conda_cmd + "python script.py", shell=True)
```

### 3. MPI/GPU Conflicts

**Problem**: Some codes (GPAW, GROMACS) failed with MPI initialization errors on CPU-only nodes.

**Solution**: Disable GPU support in MPI:
```python
os.environ['MPICH_GPU_SUPPORT_ENABLED'] = '0'
```

### 4. Quote Escaping in Remote Execution

**Problem**: Nested quotes in shell commands caused syntax errors when serialized through Globus Compute.

**Solution**: Avoid complex quoting by:
- Writing scripts to temporary files
- Using environment variables instead of inline strings
- Structuring tests with separate command components

### 5. License-Restricted Software

**Problem**: Some packages (ORCA, NAMD) can't be auto-installed.

**Solution**:
- Mark these configs with `manual_download: required: true`
- Use warning indicator (⚠️) in documentation
- Provide clear installation instructions in config files

### 6. Disk Quota Limits

**Problem**: MACE failed to install on Aurora due to disk quota during pip install.

**Lesson**: Large ML packages with many dependencies can exceed user quotas. Monitor disk usage during installation.

### 7. API Version Changes

**Problem**: Libraries like Phonopy changed APIs between versions (e.g., `set_mesh()` → `run_mesh()`).

**Solution**: LLM-driven discovery naturally adapts to API changes by reading current documentation and testing iteratively.

## Multi-Fidelity Pipeline

The repository includes a demonstration pipeline that chains 5 computational steps:

1. **Conformer Generation (xtb)** - Generate molecular conformations
2. **Ab Initio Energy (PySCF)** - Accurate quantum chemistry on lowest conformer
3. **ML Training (SchNetPack)** - Train neural network potential
4. **Molecular Dynamics (xtb)** - Run MD simulation
5. **Trajectory Analysis (MDAnalysis)** - Compute RMSD, Rg, fluctuations

Scripts available for both systems:
- `scripts/multi_fidelity_pipeline.py` (Polaris)
- `scripts/multi_fidelity_pipeline_aurora.py` (Aurora)

## Configuration Counts

### By Domain

| Domain | Polaris | Aurora |
|--------|---------|--------|
| Quantum Chemistry | 5 | 4 |
| DFT Codes | 6 | 6 |
| Molecular Dynamics | 4 | 3 |
| ML Potentials | 3 | 2 |
| Analysis Tools | 4 | 4 |
| Materials Science | 2 | 2 |
| Engineering | 1 | 1 |
| **Total** | **26** | **23** |

### By Installation Method

| Method | Count |
|--------|-------|
| Conda (conda-forge) | 18 |
| Pip | 14 |
| System Modules | 6 |
| Manual Download | 2 |

## Future Work

1. **Additional Systems**: Extend to Frontier (ORNL), Perlmutter (NERSC)
2. **Container Generation**: Auto-generate Singularity containers from configs
3. **Workflow Integration**: Direct integration with Parsl, Prefect, Globus Flows
4. **Skill Chaining**: Declarative multi-step workflow definitions
5. **Cost Estimation**: Predict compute hours before execution

## Acknowledgments

This work used resources of the Argonne Leadership Computing Facility, a U.S. Department of Energy Office of Science user facility at Argonne National Laboratory. ConfigDiscovery was developed using Claude, Anthropic's AI assistant.
