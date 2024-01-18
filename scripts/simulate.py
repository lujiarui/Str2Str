"""
MD simulations based on pdbfixer and OpenMM.
"""

import io
import os
import sys
import time 
import logging
from typing import Collection, Optional, Sequence
import tempfile
import shutil 
from glob import glob
import argparse

import numpy as np
# import pandas as pd
from pdbfixer import PDBFixer
import openmm
from simtk import unit
from openmm import app as openmm_app

from src.common.pdb_utils import split_pdbfile, merge_pdbfiles


logging.basicConfig(level=logging.INFO)

ENERGY = unit.kilocalories_per_mole
LENGTH = unit.angstroms
TEMPERATURE = unit.kelvin
TIME = unit.picoseconds
PRESSURE = unit.atmospheres
BOLTZMANN_CONST = 0.001985875 # kcal/mol K
# per-system parameters in desres paper
PS_SIMULATION_TIME_US = {
    'CLN025': 106,
    '2JOF': 208,
    '1FME': 325,
    '2F4K': 125,
    'GTT': 1137,
    'NTL9': 2936,
    '2WAV': 429,
    'PRB': 104,
    'UVF': 327,
    'NuG2': 1155,
    'A3D': 707,
    'lambda': 643,
}
PS_TEMPERATURE_KELVIN = {
    'CLN025': 340,
    '2JOF': 290,
    '1FME': 325,
    '2F4K': 360,
    'GTT': 360,
    'NTL9': 355,
    '2WAV': 298,
    'PRB': 340,
    'UVF': 360,
    'NuG2': 350,
    'A3D': 370,
    'lambda': 350,
}


def openmm_to_pdb_string(topology: openmm_app.Topology, positions: unit.Quantity):
    """Returns a pdb string provided OpenMM topology and positions."""
    with io.StringIO() as f:
        openmm_app.PDBFile.writeFile(topology, positions, f)
        return f.getvalue()

def clean_pdb_file(
    pdb_file, 
    save_to=None, 
    output_dir=None, 
    add_Hs=False, 
    verbose=False
):
    """Apply pdbfixer to the contents of a PDB file; return a PDB string result.
    
    *** This function will only process the first model in the PDB file ***
    
    Example inspired by https://htmlpreview.github.io/?https://github.com/openmm/pdbfixer/blob/master/Manual.html.
    
    1) Replaces nonstandard residues.
    2) Removes heterogens (non protein residues) including water.
    3) Adds missing residues and missing atoms within existing residues.
    4) Adds hydrogens assuming pH=7.0.
    5) KeepIds is currently true, so the fixer must keep the existing chain and
        residue identifiers. This will fail for some files in wider PDB that have
        invalid IDs.
    By default, the input pdbfile contains single-chain structure CA array.
    
    Args:
        pdbfile: Input PDB file handle.
        save_to: if not None, write the fixed PDB file to this file handle.
    Returns:
        Fixed PDB string.
    """
    fixer = PDBFixer(filename=pdb_file)
    
    # standardize the residue name
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingResidues()   # not do this
    
    # add side chains 
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=0)
    
     # add hydrogens
    if add_Hs:
        fixer.addMissingHydrogens(7.0)  # necessary for minimization
    fixer.removeHeterogens(keepWater=False)   # remove heterogens including water
    
    # save to pdb string
    out_handle = io.StringIO()
    openmm_app.PDBFile.writeFile(
        fixer.topology, 
        fixer.positions, 
        out_handle,
        keepIds=True,
    )
    pdb_string = out_handle.getvalue()    # pdb string
    
    # Configure output directory.
    if save_to is not None:
        assert not os.path.exists(save_to), f"File {save_to} already exists."
        save_to = os.path.abspath(save_to)
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
    elif output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_to = os.path.join(output_dir, os.path.basename(pdb_file))
    else:
        save_to = None
    
    if save_to is not None:
        if verbose: print(f"Saving fixed PDB file ({pdb_file}) => {save_to}")
        with open(save_to, 'w') as f:
            f.write(pdb_string)
    
    return pdb_string

def prepare_simulation(
    pdb_str: str,
    use_gpu: bool = True,
    
    temperature: float = 298,
    friction: float = 1.0,
    timestep: float = 0.0025,
    
    implicit_solvent: bool = False,
    
    restrain_coords: bool = False,
    stiffness: float = 10.0,
    restraint_set: str = "non_hydrogen",
    exclude_residues: Optional[Collection[int]] = None,
):
    """Minimize energy via openmm.
        The snippets are inspired from http://docs.openmm.org/latest/userguide/application/02_running_sims.html.
    
    #  Default Langevin dynamics in OpenMM:
        #   the simulation temperature (298 K),
        #   the friction coefficient (1 ps-1),
        #   and the step size (4 fs).
    
    Args:
        stiffness: kcal/mol A**2, the restraint stiffness. 
            The default value is the AlphaFold default.
        friction: ps^-1, the friction coefficient for Langevin dynamics. 
            Unit of reciprocal time.
        temperature: kelvin, the temperature for simulation.
        timestep: ps, the timestep for Langevin dynamics. 
        
        restrain_coords: bool, whether to restrain the coordinates. 
            Set to True if you want to relax. (default: False)
    """
    # assign physical units
    exclude_residues = exclude_residues or []
    stiffness = stiffness * ENERGY / (LENGTH**2)
    temperature = temperature * TEMPERATURE
    friction = friction / TIME
    timestep = timestep * TIME
    
    # read in pdb
    pdb_file = io.StringIO(pdb_str)
    pdb = openmm_app.PDBFile(pdb_file)
    
    # create system and force field
    if implicit_solvent:
        forcefield_name = 'amber14/protein.ff14SB.xml'
        solvent_name = 'implicit/gbn2.xml'
        forcefield = openmm_app.ForceField(forcefield_name, solvent_name) 
        topology, positions = pdb.topology, pdb.positions
        
        system = forcefield.createSystem(topology, nonbondedMethod=openmm_app.NoCutoff,
                                        nonbondedCutoff=1*unit.nanometer, constraints=openmm_app.HBonds,
                                        soluteDielectric=1.0, solventDielectric=78.5)
        
    else:
        forcefield_name = 'amber14/protein.ff14SB.xml'
        solvent_name = 'amber14/tip3p.xml'
        forcefield = openmm_app.ForceField(forcefield_name, solvent_name)
        # add hydrogen
        modeller = openmm.app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(forcefield, pH=7.0)
        topology, positions = modeller.getTopology(), modeller.getPositions()
        # add solvent (see http://docs.openmm.org/latest/userguide/application/03_model_building_editing.html?highlight=padding)
        box_padding = 1.0 * unit.nanometers
        ionicStrength = 0 * unit.molar
        positiveIon = 'Na+' 
        negativeIon = 'Cl-'
        modeller = openmm_app.Modeller(topology, positions)
        modeller.addSolvent(forcefield, 
                            model=solvent_name.split('.xml')[0].split('/')[-1],
                            # boxSize=openmm.Vec3(5.0, 5.0, 5.0) * unit.nanometers,
                            padding=box_padding,
                            ionicStrength=ionicStrength,
                            positiveIon=positiveIon,
                            negativeIon=negativeIon,
        )
        topology, positions = modeller.getTopology(), modeller.getPositions()
        
        system = forcefield.createSystem(topology, nonbondedMethod=openmm_app.PME, constraints=None,
                                        rigidWater=None)
    # add restraints if necessary
    if restrain_coords and stiffness > 0 * ENERGY / (LENGTH**2):
        _add_restraints(system, pdb, stiffness, restraint_set, exclude_residues)
    
    # see http://docs.openmm.org/latest/userguide/theory/04_integrators.html#integrators-theory for choice of integrators
    integrator = openmm.LangevinMiddleIntegrator(temperature, friction, timestep)
    platform = openmm.Platform.getPlatformByName("CUDA" if use_gpu else "CPU")
    simulation = openmm_app.Simulation(topology, system, integrator, platform) 
    simulation.context.setPositions(positions)  # assign positions, different between implicit and explicit solvent
    
    return simulation
    

def run_minimization(
    simulation,
    tolerance: float = 2.39,
    max_iter: int = 0,
):
    """Run minimization for the input simulation obj.
    
    Args:
        tolerance: kcal/mol, the energy tolerance of L-BFGS.
            The default value is: the OpenMM default 10, while in AF2, 2.39 is used.
            (See http://docs.openmm.org/latest/userguide/application/02_running_sims.html#energy-minimization)
        max_iter: int, the maximum number of iterations for minimization. (default: 0, or no limit)
    """
    ret = {}
    tolerance = tolerance * ENERGY
    
    # initial states
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    ret["einit"] = state.getPotentialEnergy().value_in_unit(ENERGY)
    ret["posinit"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)
    start = time.time()
    _minimized = False
    # minimization steps
    try:
        simulation.minimizeEnergy(
            tolerance=tolerance, maxIterations=max_iter
        )
        _minimized = True
    except Exception as e:
        pass
        logging.info(e)
        logging.warning("Minimization failed on building model")
        logging.warning(
            f"potential energy was {simulation.context.getState(getEnergy=True).getPotentialEnergy()}"
        )
        logging.warning(f"initial positions were: {ret['posinit']}")
    
    if not _minimized: # TODO: other options
        raise ValueError(f"Minimization failed after {max_iter} attempts.")
    # final states
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    ret["efinal"] = state.getPotentialEnergy().value_in_unit(ENERGY)
    ret["pos"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)
    ret["minimize_time"] = time.time() - start
    return ret


def setup_nvt_simulation(
    simulation,
    temperature: float = 298,
):
    """
    Setup NVT equilibration on the simulation object, default settings:
    """
    temperature = temperature * TEMPERATURE
    # Set initial velocities
    simulation.context.setVelocitiesToTemperature(temperature)    
    return simulation


def setup_npt_simulation(
    simulation,
    use_gpu: bool = True,
    pressure: float = 1.0,
    temperature: float = 298,
    friction: float = 1.0,
    timestep: bool = 0.0025,
):
    temperature = temperature * TEMPERATURE
    pressure = pressure * PRESSURE
    friction = friction / TIME
    timestep = timestep * TIME
    
    system = simulation.system
    topology = simulation.topology
    positions = simulation.context.getState(getPositions=True).getPositions()
    velocities = simulation.context.getState(getVelocities=True).getVelocities()

    integrator = openmm.LangevinMiddleIntegrator(temperature, friction, timestep)
    platform = openmm.Platform.getPlatformByName("CUDA" if use_gpu else "CPU")

    # Add a barostat to maintain constant pressure
    barostat = openmm.MonteCarloBarostat(pressure, temperature)
    system.addForce(barostat)

    npt_simulation = openmm_app.Simulation(topology, system, integrator, platform)

    npt_simulation.context.setPositions(positions)
    npt_simulation.context.setVelocities(velocities)
    
    return npt_simulation


def setup_reporter(
    simulation,
    output_traj="output.pdb",
    output_data="output.dat",
    report_frequency: int = 40000,
    stdout_frequency: Optional[int] = None,
):
    """
    report_frequency: int, the frequency of reporting the simulation status.
        default: 40,000, or 100 ps for timestep of 2.5 fs.
        
    """
    if output_traj is not None:
        traj_format = os.path.splitext(output_traj)[-1]
        if traj_format == '.pdb':
            simulation.reporters.append(openmm_app.PDBReporter(output_traj, report_frequency))
        elif traj_format == '.dcd':
            simulation.reporters.append(openmm_app.DCDReporter(output_traj, report_frequency))
        else:
            raise ValueError(f"Unknown trajectory format {traj_format}")
    
    if output_data is not None:
        assert os.path.splitext(output_data)[-1] == '.dat'
        simulation.reporters.append(
            openmm_app.StateDataReporter(
                output_data,
                report_frequency,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                density=True,
                speed=True,
                separator='\t'
            )
        )
    # always stdout reporter
    if stdout_frequency is None:
        stdout_frequency = report_frequency
    simulation.reporters.append(
        openmm_app.StateDataReporter(
            sys.stdout, stdout_frequency, step=True, time=True, potentialEnergy=True, temperature=True, speed=True,
        )
    )
    return 


def clean_reporters(simulation):
    """
    Remove all reporters from the simulation object
    """
    for reporter in simulation.reporters:
        simulation.reporters.remove(reporter)
    return


def subroutine(
    pdb_path: str,
    timestep: float = 0.0025,   # follow des
    temperature: Optional[float] = None,
    npt_simulation_time: Optional[float] = None, # 100 ns
    nvt_equilibration_time: float = 1000.0, # 1 ns
    npt_equilibration_time: float = 1000.0, # 1 ns
    output_dir: str = "openmm_outputs/raw",
    clean: bool = True,
    n_saved_models: int = 100,
):
    # read string
    assert os.path.isfile(pdb_path), f"File {pdb_path} does not exist."
    
    if clean:
        logging.info(f'Cleaning pdb file...')
        pdb_str = clean_pdb_file(pdb_path)
        # add sidechains and hydrogens if needed
    else:   # read from file
        with open(pdb_path, 'r') as f:
            pdb_str = f.read()
    
    str_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    base = os.path.basename(pdb_path).split('.pdb')[0]
    out_base = f"{base}_npt{npt_simulation_time}_ts{timestep}_{str_time}"
    output_dir = os.path.join(output_dir, out_base)
    os.makedirs(output_dir, exist_ok=False)
    
    # load simulation parameters
    base = base.split("_")[0]
    temperature = temperature if temperature is not None else PS_TEMPERATURE_KELVIN[base]
    npt_simulation_time = npt_simulation_time if npt_simulation_time is not None else PS_SIMULATION_TIME_US[base] * 1e6
    
    # crete simulation object
    simulation = prepare_simulation(pdb_str, timestep=timestep, temperature=temperature)
    
    # pre-stage 1: minimize
    logging.info(f'Minimizing until convergence...')
    _ = run_minimization(simulation)
    # pre-stage 2: NVT equilibration
    simulation = setup_nvt_simulation(simulation)
    _steps = int(np.floor(nvt_equilibration_time / timestep))
    logging.info(f'Equilibrating NVT with {_steps} steps...')
    setup_reporter(simulation,
                   output_traj=None,
                   output_data=os.path.join(output_dir, f"nvt_equi.dat"),
                   report_frequency=int(_steps // 100),
    )
    simulation.step(steps=_steps)
    clean_reporters(simulation)
    # pre-stage 3: NPT equilibration
    simulation = setup_npt_simulation(simulation)
    _steps = int(np.floor(npt_equilibration_time / timestep))
    logging.info(f'Equilibrating NPT with {_steps} steps...')
    setup_reporter(simulation,
                   output_traj=None,
                   output_data=os.path.join(output_dir, f"npt_equi.dat"),
                   report_frequency=int(_steps // 100),
    )
    simulation.step(steps=_steps)
    clean_reporters(simulation)
    # subroutine: NPT MD simulation
    _steps = int(np.floor(npt_simulation_time / timestep))
    logging.info(f'Simulating NPT with {_steps} steps...')
    setup_reporter(simulation,
                   output_traj=os.path.join(output_dir, f"npt.pdb"),
                   output_data=os.path.join(output_dir, f"npt.dat"),
                   report_frequency=int(_steps // n_saved_models),
    )
    simulation.step(steps=_steps)
    clean_reporters(simulation)
    return output_dir
    

def enhance_sampling_pdb(
    pdb_path: str,
    timestep: float = 0.0025,   # follow des
    nvt_equilibration_time: float = 1000.0, # 1 ns
    npt_equilibration_time: float = 1000.0, # 1 ns
    npt_simulation_time: float = 1000.0, # 1 ns
    output_dir: str = "openmm_outputs/trajectory",
    n_max_input_models: int = 100,
    n_saved_models: int = 100,
):
    base = os.path.basename(pdb_path).replace('.pdb', '')
    with tempfile.TemporaryDirectory(suffix=None, prefix=None, dir=None) as tmpdir:
        fixer_output_dir = os.path.join(tmpdir, 'fixed')
        tmp_output_dir = os.path.join(tmpdir, 'outputs') 
        os.makedirs(fixer_output_dir)
        pdb_ids = []
        # split files
        _ = split_pdbfile(pdb_path, output_dir=tmpdir, verbose=True)    # {tmpdir}/{base}_{i}.pdb i from 0
        
        pdb_files = glob(os.path.join(tmpdir, '*.pdb'))
        pdb_files = np.random.choice(pdb_files, n_max_input_models, replace=False) if len(pdb_files) > n_max_input_models else pdb_files
        
        # clean pdb files prior to simulation
        for ppath in pdb_files:
            _ = clean_pdb_file(ppath, output_dir=fixer_output_dir, verbose=True)
            
        cleaned_pdb_paths = glob(os.path.join(fixer_output_dir, '*.pdb'))
        logging.info(f'Total {len(cleaned_pdb_paths)} pdb files to be processed.')
        for i, ppath in enumerate(cleaned_pdb_paths):
            # try:
            subroutine(ppath, timestep=timestep, nvt_equilibration_time=nvt_equilibration_time,
                     npt_equilibration_time=npt_equilibration_time, npt_simulation_time=npt_simulation_time,
                    output_dir=tmp_output_dir, clean=False, n_saved_models=n_saved_models)
            # except Exception as e:
                # print(f"Warning, the simulation failed for {i}'s conformation", e)
                # pass
        glob_pattern = os.path.join(tmp_output_dir, f"{base}_*/npt.pdb")
        save_to = os.path.join(output_dir, f"es_npt{npt_simulation_time:0.0f}_ts{timestep}", f"{base}.pdb")
        post_process_trajectory(glob_pattern, save_to)
    return save_to


def post_process_trajectory(glob_pattern, output_path):
    """Post-process the trajectory files to remove the duplicated frames."""
    # collect results
    out_pdb_paths = glob(glob_pattern)
    print(f"Found totally {len(out_pdb_paths)} pdb files. Save to {output_path} now.")
    merge_pdbfiles(out_pdb_paths, output_path, verbose=True)
    return output_path


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--pdb_dir", type=str, default="data/Science2011")
    parser.add_argument("--output_dir", type=str, default="openmm_outputs")
    parser.add_argument("--es", action="store_true")
    args = parser.parse_args()
    return args

                
if __name__ == "__main__":
    args = get_args()
    test_pdbs = glob(os.path.join(args.pdb_dir, '*.pdb'))
    
    for test_pdb in test_pdbs:
        base = os.path.basename(test_pdb)
        if args.es:
            if args.debug:
                enhance_sampling_pdb(test_pdb, n_max_input_models=5, output_dir=os.path.join(args.output_dir, "trajectory"))
            else:
                enhance_sampling_pdb(test_pdb, output_dir=os.path.join(args.output_dir, "trajectory"))
        else:
            if args.debug:
                kwargs = dict(n_saved_models=100, npt_simulation_time=1000, 
                            output_dir=os.path.join(args.output_dir, "debug"),
                            npt_equilibration_time=100, nvt_equilibration_time=100)   # 1ns for 100 models
            else:
                kwargs = dict(n_saved_models=10000, npt_simulation_time=100000, 
                              output_dir=os.path.join(args.output_dir, "raw"))      # 100ns for 10000 models
            
            saved_dir = subroutine(test_pdb, **kwargs)
            glob_pattern = os.path.join(saved_dir, f"npt.pdb")
            post_process_trajectory(glob_pattern, 
                os.path.join(
                    args.output_dir, 
                    "trajectory", 
                    f"{'_'.join(os.path.basename(saved_dir).split('_')[1:-1])}", 
                    f"{base}"
                )
            )