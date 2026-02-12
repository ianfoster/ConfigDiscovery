"""Skill abstraction layer for ConfigDiscovery.

Skills are abstract computational capabilities that can have multiple
implementations across different HPC systems. This allows users to
express intent ("compute molecular energy") without specifying how
or where it runs.
"""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .compute import get_compute_client, HPC_COMPUTE_ENDPOINTS
from .schema import HPCConfig


class DataType(str, Enum):
    """Supported data types for skill inputs/outputs."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    XYZ = "xyz"  # Molecular coordinates
    TRAJECTORY = "trajectory"  # MD trajectory
    ARRAY = "array"  # Numpy array
    FILE = "file"  # File path
    JSON = "json"  # Arbitrary JSON


class ParameterSpec(BaseModel):
    """Specification for a skill parameter (input or output)."""

    type: DataType = Field(..., description="Data type")
    description: str | None = Field(default=None, description="Human-readable description")
    required: bool = Field(default=True, description="Whether this parameter is required")
    default: Any = Field(default=None, description="Default value if not required")
    unit: str | None = Field(default=None, description="Unit of measurement (e.g., 'hartree', 'eV', 'angstrom')")
    options: list[str] | None = Field(default=None, description="Allowed values for enum-like parameters")


class SkillImplementation(BaseModel):
    """A concrete implementation of a skill for a specific HPC system."""

    system: str = Field(..., description="HPC system name (e.g., 'polaris', 'aurora')")
    config_path: str = Field(..., description="Path to the config YAML file")
    endpoint_id: str | None = Field(default=None, description="Override endpoint ID")
    parameter_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Map skill parameter names to config function parameter names"
    )
    output_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Map config output keys to skill output names"
    )

    def get_config(self) -> HPCConfig:
        """Load the HPCConfig for this implementation."""
        return HPCConfig.from_yaml_file(self.config_path)


class Skill(BaseModel):
    """An abstract computational skill with typed inputs and outputs."""

    name: str = Field(..., description="Unique skill identifier")
    description: str = Field(..., description="What this skill does")
    category: str = Field(default="general", description="Skill category for organization")

    inputs: dict[str, ParameterSpec] = Field(
        default_factory=dict,
        description="Input parameters"
    )
    outputs: dict[str, ParameterSpec] = Field(
        default_factory=dict,
        description="Output parameters"
    )

    implementations: list[SkillImplementation] = Field(
        default_factory=list,
        description="Available implementations for different systems"
    )

    tags: list[str] = Field(default_factory=list, description="Searchable tags")

    def available_systems(self) -> list[str]:
        """Return list of systems where this skill can run."""
        return list(set(impl.system for impl in self.implementations))

    def get_implementation(self, system: str) -> SkillImplementation | None:
        """Get implementation for a specific system."""
        for impl in self.implementations:
            if impl.system == system:
                return impl
        return None

    def validate_inputs(self, **kwargs) -> dict[str, Any]:
        """Validate and normalize input parameters."""
        validated = {}

        for name, spec in self.inputs.items():
            if name in kwargs:
                validated[name] = kwargs[name]
            elif spec.required:
                if spec.default is not None:
                    validated[name] = spec.default
                else:
                    raise ValueError(f"Required input '{name}' not provided")
            elif spec.default is not None:
                validated[name] = spec.default

        # Check for options constraints
        for name, value in validated.items():
            spec = self.inputs[name]
            if spec.options and value not in spec.options:
                raise ValueError(f"Input '{name}' must be one of {spec.options}, got '{value}'")

        return validated


class SkillRegistry:
    """Registry of available skills."""

    def __init__(self, skills_dir: str | Path | None = None):
        self.skills: dict[str, Skill] = {}
        self._skills_dir = Path(skills_dir) if skills_dir else None

        # Register built-in skills
        self._register_builtin_skills()

    def _register_builtin_skills(self):
        """Register the built-in skill definitions."""

        # Molecular energy calculation - multiple implementations
        self.register(Skill(
            name="molecular_energy",
            description="Compute quantum mechanical energy of a molecular system",
            category="quantum_chemistry",
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
                    options=["HF", "DFT", "B3LYP", "PBE", "CCSD", "MP2", "GFN2-xTB"]
                ),
                "basis": ParameterSpec(
                    type=DataType.STRING,
                    description="Basis set",
                    default="6-31g"
                ),
            },
            outputs={
                "energy": ParameterSpec(
                    type=DataType.FLOAT,
                    description="Total energy",
                    unit="hartree"
                ),
                "status": ParameterSpec(
                    type=DataType.STRING,
                    description="Calculation status"
                ),
            },
            implementations=[
                # PySCF - good for HF, DFT, post-HF methods
                SkillImplementation(
                    system="polaris",
                    config_path="configs/polaris/pyscf.yaml",
                    parameter_mapping={"molecule": "xyz_string", "method": "method", "basis": "basis"},
                    output_mapping={"energy_hartree": "energy"}
                ),
                # Psi4 - good for accurate correlated methods
                SkillImplementation(
                    system="polaris",
                    config_path="configs/polaris/psi4.yaml",
                    parameter_mapping={"molecule": "molecule", "method": "method", "basis": "basis"},
                    output_mapping={"energy": "energy"}
                ),
                # xtb - fast semi-empirical (GFN2-xTB)
                SkillImplementation(
                    system="polaris",
                    config_path="configs/polaris/xtb.yaml",
                    parameter_mapping={"molecule": "xyz_input"},
                    output_mapping={"energy_hartree": "energy"}
                ),
                # GPAW - real-space DFT
                SkillImplementation(
                    system="polaris",
                    config_path="configs/polaris/gpaw.yaml",
                    parameter_mapping={"molecule": "system"},
                    output_mapping={"energy_eV": "energy"}
                ),
                # NWChem - general purpose
                SkillImplementation(
                    system="polaris",
                    config_path="configs/polaris/nwchem.yaml",
                    parameter_mapping={"molecule": "xyz_string", "method": "method", "basis": "basis"},
                    output_mapping={"total_energy": "energy"}
                ),
                # PySCF on Aurora - full QC methods
                SkillImplementation(
                    system="aurora",
                    config_path="configs/aurora/pyscf.yaml",
                    parameter_mapping={"molecule": "xyz_string", "method": "method", "basis": "basis"},
                    output_mapping={"energy_hartree": "energy"}
                ),
                # ASE on Aurora - fast EMT calculator (ignores method/basis, uses EMT)
                SkillImplementation(
                    system="aurora",
                    config_path="configs/aurora/ase.yaml",
                    parameter_mapping={"molecule": "formula", "method": "_ignore", "basis": "_ignore"},
                    output_mapping={"energy_eV": "energy"}
                ),
            ],
            tags=["energy", "quantum", "electronic_structure"]
        ))

        # Geometry optimization
        self.register(Skill(
            name="geometry_optimization",
            description="Optimize molecular geometry to find minimum energy structure",
            category="quantum_chemistry",
            inputs={
                "molecule": ParameterSpec(
                    type=DataType.XYZ,
                    description="Initial molecular structure",
                    required=True
                ),
                "method": ParameterSpec(
                    type=DataType.STRING,
                    description="Optimization method",
                    default="GFN2-xTB"
                ),
            },
            outputs={
                "optimized_geometry": ParameterSpec(
                    type=DataType.XYZ,
                    description="Optimized molecular structure"
                ),
                "energy": ParameterSpec(
                    type=DataType.FLOAT,
                    description="Energy at optimized geometry",
                    unit="hartree"
                ),
            },
            implementations=[
                SkillImplementation(
                    system="polaris",
                    config_path="configs/polaris/xtb.yaml",
                    parameter_mapping={"molecule": "xyz_input"},
                    output_mapping={"optimized_xyz": "optimized_geometry", "energy_hartree": "energy"}
                ),
                SkillImplementation(
                    system="aurora",
                    config_path="configs/aurora/xtb.yaml",
                    parameter_mapping={"molecule": "xyz_input"},
                    output_mapping={"optimized_xyz": "optimized_geometry", "energy_hartree": "energy"}
                ),
            ],
            tags=["optimization", "geometry", "structure"]
        ))

        # Molecular dynamics - multiple implementations
        self.register(Skill(
            name="molecular_dynamics",
            description="Run molecular dynamics simulation",
            category="simulation",
            inputs={
                "molecule": ParameterSpec(
                    type=DataType.XYZ,
                    description="Starting molecular structure",
                    required=True
                ),
                "temperature": ParameterSpec(
                    type=DataType.FLOAT,
                    description="Simulation temperature",
                    default=300.0,
                    unit="kelvin"
                ),
                "steps": ParameterSpec(
                    type=DataType.INTEGER,
                    description="Number of MD steps",
                    default=1000
                ),
                "timestep": ParameterSpec(
                    type=DataType.FLOAT,
                    description="Integration timestep",
                    default=0.5,
                    unit="femtosecond"
                ),
            },
            outputs={
                "trajectory": ParameterSpec(
                    type=DataType.TRAJECTORY,
                    description="MD trajectory frames"
                ),
                "energies": ParameterSpec(
                    type=DataType.ARRAY,
                    description="Energy at each frame"
                ),
            },
            implementations=[
                # xtb - fast QM/MD
                SkillImplementation(
                    system="polaris",
                    config_path="configs/polaris/xtb.yaml",
                    parameter_mapping={
                        "molecule": "xyz_input",
                        "temperature": "temperature",
                        "steps": "md_steps"
                    }
                ),
                # OpenMM - GPU-accelerated biomolecular MD
                SkillImplementation(
                    system="polaris",
                    config_path="configs/polaris/openmm.yaml",
                    parameter_mapping={
                        "temperature": "temperature",
                        "steps": "steps"
                    }
                ),
                # LAMMPS - classical MD
                SkillImplementation(
                    system="polaris",
                    config_path="configs/polaris/lammps.yaml",
                    parameter_mapping={
                        "steps": "timesteps",
                        "temperature": "temperature"
                    }
                ),
                # xtb MD on Aurora
                SkillImplementation(
                    system="aurora",
                    config_path="configs/aurora/xtb.yaml",
                    parameter_mapping={
                        "molecule": "xyz_input",
                        "steps": "_ignore",
                        "temperature": "_ignore"
                    }
                ),
                # OpenMM on Aurora
                SkillImplementation(
                    system="aurora",
                    config_path="configs/aurora/openmm.yaml",
                    parameter_mapping={
                        "temperature": "temperature",
                        "steps": "steps"
                    }
                ),
            ],
            tags=["dynamics", "simulation", "trajectory"]
        ))

        # Biomolecular MD (specialized for proteins/biomolecules)
        self.register(Skill(
            name="biomolecular_md",
            description="Run molecular dynamics on biomolecular systems (proteins, DNA, membranes)",
            category="simulation",
            inputs={
                "structure": ParameterSpec(
                    type=DataType.FILE,
                    description="PDB or structure file",
                    required=True
                ),
                "forcefield": ParameterSpec(
                    type=DataType.STRING,
                    description="Force field to use",
                    default="amber14",
                    options=["amber14", "charmm36", "opls"]
                ),
                "temperature": ParameterSpec(
                    type=DataType.FLOAT,
                    description="Simulation temperature",
                    default=300.0,
                    unit="kelvin"
                ),
                "steps": ParameterSpec(
                    type=DataType.INTEGER,
                    description="Number of MD steps",
                    default=10000
                ),
                "solvate": ParameterSpec(
                    type=DataType.BOOLEAN,
                    description="Add water box",
                    default=True
                ),
            },
            outputs={
                "trajectory": ParameterSpec(
                    type=DataType.FILE,
                    description="Trajectory file (DCD or XTC)"
                ),
                "final_structure": ParameterSpec(
                    type=DataType.FILE,
                    description="Final structure"
                ),
            },
            implementations=[
                SkillImplementation(
                    system="polaris",
                    config_path="configs/polaris/openmm.yaml",
                ),
                SkillImplementation(
                    system="polaris",
                    config_path="configs/polaris/namd.yaml",
                ),
                SkillImplementation(
                    system="aurora",
                    config_path="configs/aurora/openmm.yaml",
                ),
                SkillImplementation(
                    system="aurora",
                    config_path="configs/aurora/gromacs.yaml",
                ),
            ],
            tags=["biomolecular", "protein", "md", "simulation"]
        ))

        # Trajectory analysis
        self.register(Skill(
            name="trajectory_analysis",
            description="Analyze MD trajectory for structural properties",
            category="analysis",
            inputs={
                "trajectory": ParameterSpec(
                    type=DataType.TRAJECTORY,
                    description="MD trajectory to analyze",
                    required=True
                ),
            },
            outputs={
                "rmsd": ParameterSpec(
                    type=DataType.ARRAY,
                    description="RMSD from initial structure over time"
                ),
                "rmsf": ParameterSpec(
                    type=DataType.ARRAY,
                    description="Per-atom fluctuations"
                ),
                "radius_of_gyration": ParameterSpec(
                    type=DataType.ARRAY,
                    description="Radius of gyration over time"
                ),
            },
            implementations=[
                SkillImplementation(
                    system="polaris",
                    config_path="configs/polaris/mdanalysis.yaml",
                ),
                SkillImplementation(
                    system="aurora",
                    config_path="configs/aurora/mdanalysis.yaml",
                ),
            ],
            tags=["analysis", "trajectory", "structure"]
        ))

        # ML potential training - multiple implementations
        self.register(Skill(
            name="train_ml_potential",
            description="Train a machine learning interatomic potential",
            category="machine_learning",
            inputs={
                "structures": ParameterSpec(
                    type=DataType.ARRAY,
                    description="List of structures with energies for training",
                    required=True
                ),
                "epochs": ParameterSpec(
                    type=DataType.INTEGER,
                    description="Number of training epochs",
                    default=100
                ),
                "model_type": ParameterSpec(
                    type=DataType.STRING,
                    description="Type of ML model",
                    default="schnet",
                    options=["schnet", "painn", "so3net", "deepmd"]
                ),
            },
            outputs={
                "model_path": ParameterSpec(
                    type=DataType.FILE,
                    description="Path to trained model"
                ),
                "training_loss": ParameterSpec(
                    type=DataType.FLOAT,
                    description="Final training loss"
                ),
                "validation_loss": ParameterSpec(
                    type=DataType.FLOAT,
                    description="Final validation loss"
                ),
            },
            implementations=[
                # SchNetPack - equivariant neural networks
                SkillImplementation(
                    system="polaris",
                    config_path="configs/polaris/schnetpack.yaml",
                ),
                SkillImplementation(
                    system="aurora",
                    config_path="configs/aurora/schnetpack.yaml",
                ),
                # DeePMD-kit - Deep Potential
                SkillImplementation(
                    system="polaris",
                    config_path="configs/polaris/deepmd-kit.yaml",
                ),
            ],
            tags=["machine_learning", "potential", "training"]
        ))

        # Phonon calculation
        self.register(Skill(
            name="phonon_calculation",
            description="Calculate phonon properties of crystalline materials",
            category="materials_science",
            inputs={
                "structure": ParameterSpec(
                    type=DataType.FILE,
                    description="Crystal structure file (POSCAR, CIF)",
                    required=True
                ),
                "supercell": ParameterSpec(
                    type=DataType.ARRAY,
                    description="Supercell dimensions [nx, ny, nz]",
                    default=[2, 2, 2]
                ),
                "mesh": ParameterSpec(
                    type=DataType.ARRAY,
                    description="q-point mesh for DOS",
                    default=[20, 20, 20]
                ),
            },
            outputs={
                "frequencies": ParameterSpec(
                    type=DataType.ARRAY,
                    description="Phonon frequencies at gamma"
                ),
                "dos": ParameterSpec(
                    type=DataType.ARRAY,
                    description="Phonon density of states"
                ),
                "thermal_properties": ParameterSpec(
                    type=DataType.JSON,
                    description="Temperature-dependent thermal properties"
                ),
            },
            implementations=[
                SkillImplementation(
                    system="polaris",
                    config_path="configs/polaris/phonopy.yaml",
                ),
                SkillImplementation(
                    system="aurora",
                    config_path="configs/aurora/phonopy.yaml",
                ),
            ],
            tags=["phonons", "materials", "thermal", "lattice_dynamics"]
        ))

        # Periodic DFT calculation
        self.register(Skill(
            name="periodic_dft",
            description="Run DFT calculation on periodic systems (crystals, surfaces, slabs)",
            category="materials_science",
            inputs={
                "structure": ParameterSpec(
                    type=DataType.FILE,
                    description="Crystal structure file",
                    required=True
                ),
                "functional": ParameterSpec(
                    type=DataType.STRING,
                    description="Exchange-correlation functional",
                    default="PBE",
                    options=["PBE", "LDA", "PBEsol", "SCAN"]
                ),
                "kpoints": ParameterSpec(
                    type=DataType.ARRAY,
                    description="k-point mesh",
                    default=[4, 4, 4]
                ),
                "ecutwfc": ParameterSpec(
                    type=DataType.FLOAT,
                    description="Plane-wave cutoff energy",
                    default=60.0,
                    unit="Ry"
                ),
            },
            outputs={
                "energy": ParameterSpec(
                    type=DataType.FLOAT,
                    description="Total energy",
                    unit="Ry"
                ),
                "forces": ParameterSpec(
                    type=DataType.ARRAY,
                    description="Atomic forces"
                ),
                "band_gap": ParameterSpec(
                    type=DataType.FLOAT,
                    description="Band gap (if insulator)",
                    unit="eV"
                ),
            },
            implementations=[
                # Quantum ESPRESSO - plane-wave DFT
                SkillImplementation(
                    system="polaris",
                    config_path="configs/polaris/quantum-espresso.yaml",
                    output_mapping={"total_energy_Ry": "energy"}
                ),
                # CP2K - mixed Gaussian/plane-wave
                SkillImplementation(
                    system="polaris",
                    config_path="configs/polaris/cp2k.yaml",
                ),
                # GPAW - real-space/LCAO
                SkillImplementation(
                    system="polaris",
                    config_path="configs/polaris/gpaw.yaml",
                ),
                # Quantum ESPRESSO on Aurora
                SkillImplementation(
                    system="aurora",
                    config_path="configs/aurora/quantum-espresso.yaml",
                    output_mapping={"total_energy_Ry": "energy"}
                ),
                # CP2K on Aurora
                SkillImplementation(
                    system="aurora",
                    config_path="configs/aurora/cp2k.yaml",
                    output_mapping={"energy_hartree": "energy"}
                ),
            ],
            tags=["dft", "periodic", "crystal", "materials"]
        ))

        # CFD simulation
        self.register(Skill(
            name="cfd_simulation",
            description="Run computational fluid dynamics simulation",
            category="engineering",
            inputs={
                "case_dir": ParameterSpec(
                    type=DataType.FILE,
                    description="OpenFOAM case directory",
                    required=True
                ),
                "solver": ParameterSpec(
                    type=DataType.STRING,
                    description="OpenFOAM solver to use",
                    default="simpleFoam",
                    options=["simpleFoam", "icoFoam", "pisoFoam", "interFoam"]
                ),
                "parallel": ParameterSpec(
                    type=DataType.BOOLEAN,
                    description="Run in parallel",
                    default=False
                ),
                "num_procs": ParameterSpec(
                    type=DataType.INTEGER,
                    description="Number of processors for parallel run",
                    default=4
                ),
            },
            outputs={
                "results_dir": ParameterSpec(
                    type=DataType.FILE,
                    description="Directory with simulation results"
                ),
                "residuals": ParameterSpec(
                    type=DataType.ARRAY,
                    description="Convergence residuals"
                ),
            },
            implementations=[
                SkillImplementation(
                    system="polaris",
                    config_path="configs/polaris/openfoam.yaml",
                ),
                SkillImplementation(
                    system="aurora",
                    config_path="configs/aurora/openfoam.yaml",
                ),
            ],
            tags=["cfd", "fluid", "engineering", "openfoam"]
        ))

        # ML potential inference (use trained model)
        self.register(Skill(
            name="ml_potential_predict",
            description="Predict energies/forces using a trained ML potential",
            category="machine_learning",
            inputs={
                "model_path": ParameterSpec(
                    type=DataType.FILE,
                    description="Path to trained model",
                    required=True
                ),
                "structures": ParameterSpec(
                    type=DataType.ARRAY,
                    description="Structures to predict",
                    required=True
                ),
            },
            outputs={
                "energies": ParameterSpec(
                    type=DataType.ARRAY,
                    description="Predicted energies"
                ),
                "forces": ParameterSpec(
                    type=DataType.ARRAY,
                    description="Predicted forces"
                ),
            },
            implementations=[
                SkillImplementation(
                    system="polaris",
                    config_path="configs/polaris/schnetpack.yaml",
                ),
                SkillImplementation(
                    system="polaris",
                    config_path="configs/polaris/deepmd-kit.yaml",
                ),
                SkillImplementation(
                    system="aurora",
                    config_path="configs/aurora/schnetpack.yaml",
                ),
            ],
            tags=["machine_learning", "potential", "inference"]
        ))

    def register(self, skill: Skill):
        """Register a skill."""
        self.skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        """Get a skill by name."""
        return self.skills.get(name)

    def list_skills(self, category: str | None = None, tag: str | None = None) -> list[Skill]:
        """List skills, optionally filtered by category or tag."""
        skills = list(self.skills.values())

        if category:
            skills = [s for s in skills if s.category == category]

        if tag:
            skills = [s for s in skills if tag in s.tags]

        return skills

    def list_categories(self) -> list[str]:
        """List all skill categories."""
        return list(set(s.category for s in self.skills.values()))

    def search(self, query: str) -> list[Skill]:
        """Search skills by name, description, or tags."""
        query = query.lower()
        results = []

        for skill in self.skills.values():
            if (query in skill.name.lower() or
                query in skill.description.lower() or
                any(query in tag for tag in skill.tags)):
                results.append(skill)

        return results


class SkillExecutor:
    """Execute skills with automatic implementation selection."""

    def __init__(self, registry: SkillRegistry | None = None):
        self.registry = registry or SkillRegistry()
        self._preferred_system: str | None = None
        self._available_endpoints: dict[str, str] = {}

    def set_preferred_system(self, system: str):
        """Set preferred HPC system for execution."""
        self._preferred_system = system

    def add_endpoint(self, system: str, endpoint_id: str):
        """Register an available endpoint."""
        self._available_endpoints[system] = endpoint_id

    def detect_available_systems(self) -> list[str]:
        """Detect which systems have available endpoints."""
        # Start with explicitly added endpoints
        available = list(self._available_endpoints.keys())

        # Add well-known endpoints
        for system, endpoint_id in HPC_COMPUTE_ENDPOINTS.items():
            if system not in available:
                available.append(system)
                self._available_endpoints[system] = endpoint_id

        return available

    def select_implementation(self, skill: Skill) -> tuple[SkillImplementation, str]:
        """Select the best available implementation for a skill.

        Returns:
            Tuple of (implementation, endpoint_id)
        """
        available = self.detect_available_systems()

        # Prefer explicitly set system
        if self._preferred_system:
            impl = skill.get_implementation(self._preferred_system)
            if impl and self._preferred_system in available:
                endpoint = impl.endpoint_id or self._available_endpoints[self._preferred_system]
                return impl, endpoint

        # Otherwise pick first available
        for impl in skill.implementations:
            if impl.system in available:
                endpoint = impl.endpoint_id or self._available_endpoints[impl.system]
                return impl, endpoint

        raise RuntimeError(
            f"No available implementation for skill '{skill.name}'. "
            f"Skill supports: {skill.available_systems()}, "
            f"Available: {available}"
        )

    def run(
        self,
        skill_name: str,
        system: str | None = None,
        timeout: int = 600,
        **kwargs
    ) -> dict[str, Any]:
        """Execute a skill.

        Args:
            skill_name: Name of the skill to run
            system: Specific system to run on (optional)
            timeout: Execution timeout in seconds
            **kwargs: Skill input parameters

        Returns:
            Dictionary with skill outputs
        """
        skill = self.registry.get(skill_name)
        if not skill:
            raise ValueError(f"Unknown skill: {skill_name}")

        # Validate inputs
        validated_inputs = skill.validate_inputs(**kwargs)

        # Select implementation
        if system:
            impl = skill.get_implementation(system)
            if not impl:
                raise ValueError(f"Skill '{skill_name}' has no implementation for '{system}'")
            endpoint_id = impl.endpoint_id or self._available_endpoints.get(system) or HPC_COMPUTE_ENDPOINTS.get(system)
            if not endpoint_id:
                raise ValueError(f"No endpoint configured for system '{system}'")
        else:
            impl, endpoint_id = self.select_implementation(skill)

        # Map skill parameters to config function parameters
        config = impl.get_config()
        func_kwargs = {}
        for skill_param, value in validated_inputs.items():
            config_param = impl.parameter_mapping.get(skill_param, skill_param)
            # Skip parameters marked to ignore (e.g., when implementation doesn't support them)
            if config_param and not config_param.startswith("_"):
                func_kwargs[config_param] = value

        # Execute
        client = get_compute_client(endpoint_id)
        result = client.register_and_run_function(
            config.execution.function,
            config.execution.function_name,
            timeout=timeout,
            **func_kwargs
        )

        if not result.success:
            return {"status": "failed", "error": result.error}

        # Map outputs
        raw_output = result.return_value
        if isinstance(raw_output, dict):
            mapped_output = {}
            for config_key, skill_key in impl.output_mapping.items():
                if config_key in raw_output:
                    mapped_output[skill_key] = raw_output[config_key]
            # Include unmapped outputs
            for key, value in raw_output.items():
                if key not in impl.output_mapping:
                    mapped_output[key] = value
            return mapped_output

        return {"result": raw_output}


# Convenience functions
_default_registry: SkillRegistry | None = None
_default_executor: SkillExecutor | None = None


def get_registry() -> SkillRegistry:
    """Get the default skill registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = SkillRegistry()
    return _default_registry


def get_executor() -> SkillExecutor:
    """Get the default skill executor."""
    global _default_executor
    if _default_executor is None:
        _default_executor = SkillExecutor(get_registry())
    return _default_executor


def run_skill(skill_name: str, system: str | None = None, **kwargs) -> dict[str, Any]:
    """Convenience function to run a skill.

    Example:
        result = run_skill("molecular_energy", molecule="H 0 0 0\\nH 0 0 0.74", method="HF")
    """
    return get_executor().run(skill_name, system=system, **kwargs)
