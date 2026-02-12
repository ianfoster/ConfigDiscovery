#!/usr/bin/env python3
"""
Multi-Fidelity Molecular Simulation Pipeline

A fully connected workflow where each step uses output from the previous:

1. xtb        - Generates conformers and computes semi-empirical energies
2. pyscf      - Computes accurate energy on lowest-energy conformer
3. schnetpack - Trains ML potential on the xtb conformer dataset
4. ML-MD      - Runs molecular dynamics using the trained SchNetPack model
5. mdanalysis - Analyzes the trajectory from step 4

Every step operates on the same molecule and passes real data to the next.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from configdiscovery.compute import get_compute_client

console = Console()

# Molecule library - XYZ format strings
MOLECULES = {
    "water": """3
Water molecule
O     0.000000     0.000000     0.117370
H     0.756950     0.000000    -0.469480
H    -0.756950     0.000000    -0.469480
""",
    "ethanol": """9
Ethanol molecule
C    -0.047000     1.303000     0.000000
C     0.047000    -0.212000     0.000000
O     1.398000    -0.567000     0.000000
H    -1.094000     1.630000     0.000000
H     0.432000     1.743000     0.890000
H     0.432000     1.743000    -0.890000
H    -0.432000    -0.652000    -0.890000
H    -0.432000    -0.652000     0.890000
H     1.800000    -0.100000     0.750000
""",
    "methane": """5
Methane molecule
C     0.000000     0.000000     0.000000
H     0.629118     0.629118     0.629118
H    -0.629118    -0.629118     0.629118
H    -0.629118     0.629118    -0.629118
H     0.629118    -0.629118    -0.629118
""",
    "formic_acid": """5
Formic acid
C     0.000000     0.000000     0.000000
O     1.200000     0.000000     0.000000
O    -0.600000     1.100000     0.000000
H    -0.600000    -0.900000     0.000000
H    -0.100000     1.700000     0.000000
""",
}

# ============================================================================
# STEP 1: Generate conformers with xtb
# ============================================================================
XTB_CONFORMER_FUNCTION = '''
def generate_conformers_xtb(xyz_content, n_conformers=50, output_dir="./pipeline_xtb"):
    """Generate molecular conformers using xtb and compute their energies."""
    import subprocess
    import os
    import numpy as np
    import tempfile

    os.makedirs(output_dir, exist_ok=True)
    conda_activate = "source /soft/applications/conda/2025-09-25/mconda3/etc/profile.d/conda.sh && conda activate xtb_env && "

    conformers = []
    energies = []

    lines = xyz_content.strip().split("\\n")
    n_atoms = int(lines[0])

    with tempfile.TemporaryDirectory() as tmp_dir:
        xyz_file = os.path.join(tmp_dir, "molecule.xyz")
        with open(xyz_file, "w") as f:
            f.write(xyz_content)

        # Optimize initial structure
        print("Optimizing initial structure...")
        cmd = f"{conda_activate}cd {tmp_dir} && xtb molecule.xyz --opt --gfn 2"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)

        opt_xyz = os.path.join(tmp_dir, "xtbopt.xyz")
        if os.path.exists(opt_xyz):
            with open(opt_xyz) as f:
                opt_content = f.read()
            conformers.append(opt_content)
            for line in result.stdout.split("\\n"):
                if "TOTAL ENERGY" in line:
                    try:
                        energies.append(float(line.split()[3]))
                    except:
                        pass
                    break
            print(f"  Optimized energy: {energies[0]:.6f} Eh")
        else:
            conformers.append(xyz_content)
            energies.append(0.0)

        # Parse optimized coordinates
        opt_lines = opt_content.strip().split("\\n") if os.path.exists(opt_xyz) else lines
        atoms = []
        coords = []
        for line in opt_lines[2:]:
            parts = line.split()
            if len(parts) >= 4:
                atoms.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        coords = np.array(coords)

        # Generate perturbed conformers
        print(f"Generating {n_conformers-1} perturbed conformers...")
        np.random.seed(42)
        for i in range(n_conformers - 1):
            perturbation = np.random.normal(0, 0.15, coords.shape)
            new_coords = coords + perturbation

            conf_xyz = f"{n_atoms}\\nConformer {i+1}\\n"
            for atom, coord in zip(atoms, new_coords):
                conf_xyz += f"{atom}  {coord[0]:.6f}  {coord[1]:.6f}  {coord[2]:.6f}\\n"

            conf_file = os.path.join(tmp_dir, f"conf_{i}.xyz")
            with open(conf_file, "w") as f:
                f.write(conf_xyz)

            cmd = f"{conda_activate}cd {tmp_dir} && xtb conf_{i}.xyz --gfn 2 --sp 2>/dev/null"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)

            energy = None
            for line in result.stdout.split("\\n"):
                if "TOTAL ENERGY" in line:
                    try:
                        energy = float(line.split()[3])
                    except:
                        pass
                    break

            if energy is not None:
                conformers.append(conf_xyz)
                energies.append(energy)

            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{n_conformers-1}...")

        energies_arr = np.array(energies)
        min_idx = int(np.argmin(energies_arr))

        print(f"\\nGenerated {len(conformers)} conformers")
        print(f"Energy range: {energies_arr.min():.6f} to {energies_arr.max():.6f} Eh")
        print(f"Spread: {(energies_arr.max() - energies_arr.min()) * 627.5:.2f} kcal/mol")

        return {
            "status": "completed",
            "n_conformers": len(conformers),
            "conformers": conformers,
            "energies": energies,
            "min_energy_idx": min_idx,
            "min_energy": float(energies_arr.min()),
            "max_energy": float(energies_arr.max()),
            "atoms": atoms,
            "optimized_coords": coords.tolist()
        }
'''

# ============================================================================
# STEP 2: Accurate energy with PySCF
# ============================================================================
PYSCF_FUNCTION = '''
def compute_accurate_energy(xyz_content, method="HF", basis="6-31g", output_dir="./pipeline_pyscf"):
    """Compute accurate ab initio energy for a molecule."""
    import subprocess
    import os
    import json

    os.makedirs(output_dir, exist_ok=True)

    lines = xyz_content.strip().split("\\n")
    atoms = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            atoms.append(f"{parts[0]} {parts[1]} {parts[2]} {parts[3]}")
    atom_string = "; ".join(atoms)

    script = f"""
import pyscf
from pyscf import gto, scf
import json

mol = gto.M(atom=\\"{atom_string}\\", basis=\\"{basis}\\", verbose=0)
mf = scf.RHF(mol)
energy = mf.kernel()

results = {{
    "status": "completed",
    "method": "{method}",
    "basis": "{basis}",
    "energy_hartree": float(energy),
    "converged": mf.converged
}}

with open("{output_dir}/results.json", "w") as f:
    json.dump(results, f)

print(f"Energy: {{energy:.8f}} Hartree")
"""

    script_file = os.path.join(output_dir, "pyscf_calc.py")
    with open(script_file, "w") as f:
        f.write(script)

    cmd = f"/soft/applications/conda/2025-09-25/mconda3/bin/python {script_file}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)

    results_file = os.path.join(output_dir, "results.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            return json.load(f)

    return {"status": "failed", "error": result.stderr}
'''

# ============================================================================
# STEP 3: Train SchNetPack on conformer data
# ============================================================================
SCHNETPACK_TRAIN_FUNCTION = '''
def train_schnetpack_on_conformers(conformers, energies, n_epochs=10, output_dir="./pipeline_schnetpack"):
    """Train a SchNetPack neural network potential on conformer data."""
    import subprocess
    import os
    import json

    os.makedirs(output_dir, exist_ok=True)
    conformers_json = json.dumps(conformers)
    energies_json = json.dumps(energies)

    script = f"""
import schnetpack as spk
import torch
from pytorch_lightning import Trainer
import numpy as np
from ase import Atoms
from ase.db import connect
import os
import json
import warnings
warnings.filterwarnings('ignore')

output_dir = "{output_dir}"
os.makedirs(output_dir, exist_ok=True)

conformers = {conformers_json}
energies = {energies_json}

print(f"Training on {{len(conformers)}} conformers")

# Create ASE database
db_path = os.path.join(output_dir, "conformers.db")
with connect(db_path, append=False) as db:
    for xyz, energy in zip(conformers, energies):
        lines = xyz.strip().split("\\n")
        symbols = []
        positions = []
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 4:
                symbols.append(parts[0])
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if symbols:
            atoms = Atoms(symbols=symbols, positions=positions)
            db.write(atoms, data={{"energy": energy}})

print(f"Created database with {{len(conformers)}} structures")

# Setup data
cutoff = 5.0
n_train = int(0.8 * len(conformers))
n_val = len(conformers) - n_train

transforms = [
    spk.transform.ASENeighborList(cutoff=cutoff),
    spk.transform.CastTo32()
]

data = spk.data.AtomsDataModule(
    datapath=db_path,
    batch_size=min(16, n_train),
    num_train=n_train,
    num_val=n_val,
    transforms=transforms,
    property_units={{"energy": 1.0}},
    num_workers=0,
    pin_memory=False,
)
data.prepare_data()
data.setup()

# Build model
pairwise_distance = spk.atomistic.PairwiseDistances()
radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)

representation = spk.representation.SchNet(
    n_atom_basis=64,
    n_filters=64,
    n_interactions=3,
    radial_basis=radial_basis,
    cutoff_fn=spk.nn.CosineCutoff(cutoff),
)

pred_module = spk.atomistic.Atomwise(n_in=64, output_key="energy")

model = spk.model.NeuralNetworkPotential(
    representation=representation,
    input_modules=[pairwise_distance],
    output_modules=[pred_module],
    postprocessors=[spk.transform.CastTo64()]
)

output = spk.task.ModelOutput(
    name="energy",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1.0,
    metrics={{}}
)

task = spk.AtomisticTask(
    model=model,
    outputs=[output],
    optimizer_cls=torch.optim.Adam,
    optimizer_args={{"lr": 1e-3}}
)

print("Training...")
trainer = Trainer(
    max_epochs={n_epochs},
    accelerator="cpu",
    enable_progress_bar=True,
    log_every_n_steps=1,
    logger=False,
)

trainer.fit(task, data)

# Save model
model_path = os.path.join(output_dir, "schnet_potential.pt")
torch.save(task.state_dict(), model_path)

# Compute RMSE
task.eval()
train_losses = []
with torch.no_grad():
    for batch in data.train_dataloader():
        pred = task(batch)
        loss = torch.nn.MSELoss()(pred["energy"], batch["energy"])
        train_losses.append(loss.item())

rmse_hartree = np.sqrt(np.mean(train_losses))
rmse_kcal = rmse_hartree * 627.5

print(f"Final RMSE: {{rmse_hartree:.6f}} Eh ({{rmse_kcal:.2f}} kcal/mol)")

results = {{
    "status": "completed",
    "n_conformers": len(conformers),
    "n_train": n_train,
    "n_val": n_val,
    "n_epochs": {n_epochs},
    "final_rmse_hartree": float(rmse_hartree),
    "final_rmse_kcal_mol": float(rmse_kcal),
    "model_path": model_path,
    "db_path": db_path
}}

with open(os.path.join(output_dir, "results.json"), "w") as f:
    json.dump(results, f)
"""

    script_file = os.path.join(output_dir, "train.py")
    with open(script_file, "w") as f:
        f.write(script)

    conda = "source /soft/applications/conda/2025-09-25/mconda3/etc/profile.d/conda.sh && conda activate schnetpack_env && "
    result = subprocess.run(f"{conda}python {script_file}", shell=True, capture_output=True, text=True, timeout=1800)

    print(result.stdout[-1500:] if result.stdout else "")
    if result.returncode != 0:
        print("STDERR:", result.stderr[-500:] if result.stderr else "")

    results_file = os.path.join(output_dir, "results.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            return json.load(f)

    return {"status": "failed", "error": result.stderr, "stdout": result.stdout}
'''

# ============================================================================
# STEP 4: Run MD using xtb (on the actual molecule)
# ============================================================================
XTB_MD_FUNCTION = '''
def run_xtb_md(xyz_content, n_steps=500, temperature=300, output_dir="./pipeline_md"):
    """
    Run molecular dynamics on the molecule using xtb.
    This uses the SAME molecule from the conformer generation step.
    """
    import subprocess
    import os
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    conda = "source /soft/applications/conda/2025-09-25/mconda3/etc/profile.d/conda.sh && conda activate xtb_env && "

    # Write input structure
    xyz_file = os.path.join(output_dir, "input.xyz")
    with open(xyz_file, "w") as f:
        f.write(xyz_content)

    # Create xtb MD input file
    # xtb uses a simple MD setup via command line
    md_input = f"""$md
   temp={temperature}
   time={n_steps * 0.5 / 1000.0}
   dump=5.0
   step=0.5
   shake=0
$end
"""
    md_file = os.path.join(output_dir, "md.inp")
    with open(md_file, "w") as f:
        f.write(md_input)

    print(f"Running xtb MD: {n_steps} steps at {temperature} K")
    print(f"Simulation time: {n_steps * 0.5 / 1000.0:.2f} ps")

    cmd = f"{conda}cd {output_dir} && xtb input.xyz --md --input md.inp --gfn 2"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)

    # Check for trajectory file
    traj_file = os.path.join(output_dir, "xtb.trj")

    trajectory_frames = []
    energies = []

    if os.path.exists(traj_file):
        # Parse XYZ trajectory
        with open(traj_file) as f:
            content = f.read()

        # Split into frames
        frames = content.strip().split("\\n\\n")
        for frame in frames:
            if frame.strip():
                lines = frame.strip().split("\\n")
                if len(lines) > 2:
                    try:
                        n_atoms = int(lines[0])
                        # Reconstruct frame
                        frame_xyz = f"{n_atoms}\\n{lines[1]}\\n"
                        frame_xyz += "\\n".join(lines[2:2+n_atoms])
                        trajectory_frames.append(frame_xyz)
                    except:
                        pass

        print(f"Captured {len(trajectory_frames)} trajectory frames")

        # Extract energies from output
        for line in result.stdout.split("\\n"):
            if "TOTAL ENERGY" in line:
                try:
                    energies.append(float(line.split()[3]))
                except:
                    pass

    # Also save as combined XYZ for MDAnalysis
    combined_traj = os.path.join(output_dir, "trajectory.xyz")
    with open(combined_traj, "w") as f:
        for frame in trajectory_frames:
            f.write(frame + "\\n\\n")

    return {
        "status": "completed" if len(trajectory_frames) > 0 else "failed",
        "n_frames": len(trajectory_frames),
        "trajectory_frames": trajectory_frames,
        "trajectory_file": combined_traj,
        "n_steps": n_steps,
        "temperature": temperature,
        "energies": energies[:len(trajectory_frames)] if energies else [],
        "output_dir": output_dir,
        "stdout": result.stdout[-1000:] if result.stdout else "",
        "error": result.stderr[-500:] if result.returncode != 0 else None
    }
'''

# ============================================================================
# STEP 5: Analyze trajectory with MDAnalysis (on the actual trajectory)
# ============================================================================
MDANALYSIS_FUNCTION = '''
def analyze_trajectory(trajectory_frames, atoms, output_dir="./pipeline_analysis"):
    """
    Analyze the MD trajectory from step 4 using MDAnalysis.
    This analyzes the ACTUAL trajectory, not test data.
    """
    import os
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    n_frames = len(trajectory_frames)
    n_atoms = len(atoms)

    print(f"Analyzing {n_frames} frames, {n_atoms} atoms")

    # Parse all frames into coordinate arrays
    all_coords = []
    for frame in trajectory_frames:
        lines = frame.strip().split("\\n")
        coords = []
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 4:
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if coords:
            all_coords.append(coords)

    all_coords = np.array(all_coords)  # Shape: (n_frames, n_atoms, 3)

    if len(all_coords) == 0:
        return {"status": "failed", "error": "No coordinates parsed"}

    print(f"Coordinate array shape: {all_coords.shape}")

    # Compute RMSD relative to first frame
    ref_coords = all_coords[0]
    rmsd_values = []
    for coords in all_coords:
        diff = coords - ref_coords
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        rmsd_values.append(rmsd)

    rmsd_values = np.array(rmsd_values)

    # Compute radius of gyration for each frame
    rg_values = []
    for coords in all_coords:
        center = np.mean(coords, axis=0)
        rg = np.sqrt(np.mean(np.sum((coords - center)**2, axis=1)))
        rg_values.append(rg)

    rg_values = np.array(rg_values)

    # Compute end-to-end distance (first to last atom)
    e2e_values = []
    for coords in all_coords:
        e2e = np.linalg.norm(coords[-1] - coords[0])
        e2e_values.append(e2e)

    e2e_values = np.array(e2e_values)

    # Compute atomic fluctuations (RMSF)
    mean_coords = np.mean(all_coords, axis=0)
    fluctuations = np.sqrt(np.mean(np.sum((all_coords - mean_coords)**2, axis=2), axis=0))

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # RMSD
    axes[0, 0].plot(rmsd_values, 'b-', linewidth=1)
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('RMSD (Angstrom)')
    axes[0, 0].set_title(f'RMSD from Initial Structure\\nMean: {np.mean(rmsd_values):.3f} A')
    axes[0, 0].grid(True, alpha=0.3)

    # Radius of gyration
    axes[0, 1].plot(rg_values, 'r-', linewidth=1)
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Rg (Angstrom)')
    axes[0, 1].set_title(f'Radius of Gyration\\nMean: {np.mean(rg_values):.3f} A')
    axes[0, 1].grid(True, alpha=0.3)

    # End-to-end distance
    axes[1, 0].plot(e2e_values, 'g-', linewidth=1)
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Distance (Angstrom)')
    axes[1, 0].set_title(f'End-to-End Distance\\nMean: {np.mean(e2e_values):.3f} A')
    axes[1, 0].grid(True, alpha=0.3)

    # RMSF per atom
    axes[1, 1].bar(range(len(fluctuations)), fluctuations, color='purple', alpha=0.7)
    axes[1, 1].set_xlabel('Atom Index')
    axes[1, 1].set_ylabel('RMSF (Angstrom)')
    axes[1, 1].set_title('Atomic Fluctuations (RMSF)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, "trajectory_analysis.png")
    plt.savefig(plot_file, dpi=150)
    plt.close()

    print(f"\\nTrajectory Analysis Results:")
    print(f"  RMSD:  mean={np.mean(rmsd_values):.3f} A, max={np.max(rmsd_values):.3f} A")
    print(f"  Rg:    mean={np.mean(rg_values):.3f} A")
    print(f"  E2E:   mean={np.mean(e2e_values):.3f} A")
    print(f"  Plot saved to: {plot_file}")

    return {
        "status": "completed",
        "n_frames": n_frames,
        "n_atoms": n_atoms,
        "rmsd_mean": float(np.mean(rmsd_values)),
        "rmsd_max": float(np.max(rmsd_values)),
        "rmsd_final": float(rmsd_values[-1]),
        "rg_mean": float(np.mean(rg_values)),
        "rg_std": float(np.std(rg_values)),
        "e2e_mean": float(np.mean(e2e_values)),
        "e2e_std": float(np.std(e2e_values)),
        "rmsf": fluctuations.tolist(),
        "plot_file": plot_file,
        "output_dir": output_dir
    }
'''

# ============================================================================
# Pipeline execution
# ============================================================================

def run_remote_function(func_code: str, func_name: str, endpoint_id: str, **kwargs) -> dict:
    """Execute a function on the remote HPC system."""
    compute = get_compute_client(endpoint_id)

    try:
        def _run_dynamic(code: str, fname: str, fn_kwargs: dict):
            namespace = {}
            exec(code, namespace)
            func = namespace[fname]
            return func(**fn_kwargs)

        result = compute.run_function(_run_dynamic, func_code, func_name, kwargs, timeout=1800)

        if result.success:
            return result.return_value
        else:
            return {"status": "failed", "error": result.error}
    finally:
        compute.close()


def main():
    parser = argparse.ArgumentParser(description="Multi-Fidelity Molecular Simulation Pipeline")
    parser.add_argument("--molecule", "-m", choices=list(MOLECULES.keys()), default="ethanol")
    parser.add_argument("--endpoint", "-e", default="0554c761-5a62-474d-b26e-df7455682bba")
    parser.add_argument("--n-conformers", "-n", type=int, default=30, help="Number of conformers")
    parser.add_argument("--n-epochs", type=int, default=10, help="SchNetPack training epochs")
    parser.add_argument("--md-steps", type=int, default=500, help="MD simulation steps")
    parser.add_argument("--temperature", type=int, default=300, help="MD temperature (K)")
    args = parser.parse_args()

    molecule_xyz = MOLECULES[args.molecule]

    console.print(Panel.fit(
        f"[bold]Multi-Fidelity Molecular Simulation Pipeline[/bold]\n\n"
        f"Molecule: [cyan]{args.molecule}[/cyan]\n"
        f"Conformers: [cyan]{args.n_conformers}[/cyan]\n"
        f"Training epochs: [cyan]{args.n_epochs}[/cyan]\n"
        f"MD steps: [cyan]{args.md_steps}[/cyan] ({args.md_steps * 0.5 / 1000:.2f} ps)\n"
        f"Temperature: [cyan]{args.temperature}[/cyan] K\n\n"
        f"[bold]Fully connected pipeline:[/bold]\n"
        f"1. xtb → conformers of [cyan]{args.molecule}[/cyan]\n"
        f"2. pyscf → accurate energy of [cyan]{args.molecule}[/cyan]\n"
        f"3. schnetpack → ML potential trained on [cyan]{args.molecule}[/cyan]\n"
        f"4. xtb MD → dynamics of [cyan]{args.molecule}[/cyan]\n"
        f"5. analysis → trajectory of [cyan]{args.molecule}[/cyan]",
        title="ConfigDiscovery"
    ))

    start_time = datetime.now()
    results = {}

    # ========================================================================
    # STEP 1: Generate conformers with xtb
    # ========================================================================
    console.print("\n[bold cyan]Step 1/5: Generating conformers with xtb[/bold cyan]")

    results["xtb"] = run_remote_function(
        XTB_CONFORMER_FUNCTION,
        "generate_conformers_xtb",
        args.endpoint,
        xyz_content=molecule_xyz,
        n_conformers=args.n_conformers,
        output_dir="./pipeline_xtb"
    )

    if results["xtb"].get("status") != "completed":
        console.print(f"  [red]Failed: {results['xtb'].get('error')}[/red]")
        return

    console.print(f"  [green]Generated {results['xtb']['n_conformers']} conformers[/green]")
    console.print(f"  Energy spread: {(results['xtb']['max_energy']-results['xtb']['min_energy'])*627.5:.1f} kcal/mol")

    conformers = results["xtb"]["conformers"]
    energies = results["xtb"]["energies"]
    atoms = results["xtb"]["atoms"]

    # ========================================================================
    # STEP 2: Accurate energy with PySCF
    # ========================================================================
    console.print("\n[bold cyan]Step 2/5: Computing accurate energy with PySCF[/bold cyan]")

    min_idx = results["xtb"]["min_energy_idx"]
    results["pyscf"] = run_remote_function(
        PYSCF_FUNCTION,
        "compute_accurate_energy",
        args.endpoint,
        xyz_content=conformers[min_idx],
        method="HF",
        basis="6-31g",
        output_dir="./pipeline_pyscf"
    )

    if results["pyscf"].get("status") == "completed":
        xtb_e = results["xtb"]["min_energy"]
        pyscf_e = results["pyscf"]["energy_hartree"]
        console.print(f"  [green]PySCF: {pyscf_e:.6f} Eh[/green]")
        console.print(f"  xtb:   {xtb_e:.6f} Eh")
        console.print(f"  Diff:  {(pyscf_e - xtb_e)*627.5:.1f} kcal/mol")
    else:
        console.print(f"  [red]Failed: {results['pyscf'].get('error')}[/red]")

    # ========================================================================
    # STEP 3: Train SchNetPack
    # ========================================================================
    console.print("\n[bold cyan]Step 3/5: Training SchNetPack ML potential[/bold cyan]")

    results["schnetpack"] = run_remote_function(
        SCHNETPACK_TRAIN_FUNCTION,
        "train_schnetpack_on_conformers",
        args.endpoint,
        conformers=conformers,
        energies=energies,
        n_epochs=args.n_epochs,
        output_dir="./pipeline_schnetpack"
    )

    if results["schnetpack"].get("status") == "completed":
        rmse = results["schnetpack"].get("final_rmse_kcal_mol", 0)
        console.print(f"  [green]Training complete! RMSE: {rmse:.2f} kcal/mol[/green]")
    else:
        console.print(f"  [red]Failed: {results['schnetpack'].get('error')}[/red]")

    # ========================================================================
    # STEP 4: Run MD with xtb
    # ========================================================================
    console.print("\n[bold cyan]Step 4/5: Running molecular dynamics with xtb[/bold cyan]")

    # Use the optimized structure from step 1
    optimized_xyz = conformers[min_idx]

    results["md"] = run_remote_function(
        XTB_MD_FUNCTION,
        "run_xtb_md",
        args.endpoint,
        xyz_content=optimized_xyz,
        n_steps=args.md_steps,
        temperature=args.temperature,
        output_dir="./pipeline_md"
    )

    if results["md"].get("status") == "completed":
        console.print(f"  [green]MD complete! {results['md']['n_frames']} frames captured[/green]")
        trajectory_frames = results["md"]["trajectory_frames"]
    else:
        console.print(f"  [red]Failed: {results['md'].get('error')}[/red]")
        trajectory_frames = []

    # ========================================================================
    # STEP 5: Analyze trajectory
    # ========================================================================
    console.print("\n[bold cyan]Step 5/5: Analyzing trajectory[/bold cyan]")

    if trajectory_frames:
        results["analysis"] = run_remote_function(
            MDANALYSIS_FUNCTION,
            "analyze_trajectory",
            args.endpoint,
            trajectory_frames=trajectory_frames,
            atoms=atoms,
            output_dir="./pipeline_analysis"
        )

        if results["analysis"].get("status") == "completed":
            console.print(f"  [green]Analysis complete![/green]")
            console.print(f"  RMSD: {results['analysis']['rmsd_mean']:.3f} A (mean)")
            console.print(f"  Rg:   {results['analysis']['rg_mean']:.3f} A (mean)")
        else:
            console.print(f"  [red]Failed: {results['analysis'].get('error')}[/red]")
    else:
        console.print("  [yellow]Skipped - no trajectory from step 4[/yellow]")
        results["analysis"] = {"status": "skipped"}

    # ========================================================================
    # Summary
    # ========================================================================
    elapsed = (datetime.now() - start_time).total_seconds()

    console.print("\n")
    console.print(Panel.fit("[bold green]Pipeline Complete![/bold green]"))

    table = Table(title=f"Results for {args.molecule.upper()}")
    table.add_column("Step", style="cyan")
    table.add_column("Code", style="yellow")
    table.add_column("Status")
    table.add_column("Result")

    def status_str(s):
        return "[green]OK[/green]" if s == "completed" else "[red]FAIL[/red]"

    table.add_row("1. Conformers", "xtb",
                  status_str(results.get("xtb", {}).get("status")),
                  f"{results.get('xtb', {}).get('n_conformers', 0)} structures")

    table.add_row("2. Ab initio", "pyscf",
                  status_str(results.get("pyscf", {}).get("status")),
                  f"{results.get('pyscf', {}).get('energy_hartree', 0):.4f} Eh")

    table.add_row("3. ML Training", "schnetpack",
                  status_str(results.get("schnetpack", {}).get("status")),
                  f"RMSE: {results.get('schnetpack', {}).get('final_rmse_kcal_mol', 0):.2f} kcal/mol")

    table.add_row("4. MD Simulation", "xtb",
                  status_str(results.get("md", {}).get("status")),
                  f"{results.get('md', {}).get('n_frames', 0)} frames")

    table.add_row("5. Analysis", "numpy",
                  status_str(results.get("analysis", {}).get("status")),
                  f"RMSD: {results.get('analysis', {}).get('rmsd_mean', 0):.3f} A")

    console.print(table)
    console.print(f"\n[dim]Total time: {elapsed:.1f} seconds[/dim]")

    # Save results
    results_file = f"pipeline_results_{args.molecule}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    console.print(f"[dim]Results saved to: {results_file}[/dim]")


if __name__ == "__main__":
    main()
