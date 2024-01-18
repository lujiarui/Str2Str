"""Utility functions for operating PDB files.
"""
import os
import re
from typing import Optional
from collections import OrderedDict

import numpy as np
from tqdm import tqdm
import biotite.structure as struct
from biotite.structure.io.pdb import PDBFile

from src.common import protein
    
    
def write_pdb_string(pdb_string: str, save_to: str):
    """Write pdb string to file"""
    with open(save_to, 'w') as f:
        f.write(pdb_string)
        
def read_pdb_to_string(pdb_file):
    """Read PDB file as pdb string. Convenient API"""
    with open(pdb_file, 'r') as fi:
        pdb_string = ''
        for line in fi:
            if line.startswith('END') or line.startswith('TER') \
                    or line.startswith('MODEL') or line.startswith('ATOM'):
                pdb_string += line
        return pdb_string

def merge_pdbfiles(input, output_file, verbose=True):
    """ordered merging process of pdbs"""
    if isinstance(input, str):
        pdb_files = [os.path.join(input, f) for f in os.listdir(input) if f.endswith('.pdb')]
    elif isinstance(input, list):
        pdb_files = input
        
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    model_number = 0
    pdb_lines = []
    if verbose: 
        _iter = tqdm(pdb_files, desc='Merging PDBs')
    else:
        _iter = pdb_files
    
    for pdb_file in _iter:
        with open(pdb_file, 'r') as pdb:
            lines = pdb.readlines()
        single_model = True
        
        for line in lines: 
            if line.startswith('MODEL') or line.startswith('ENDMDL'):
                single_model = False
                break
        
        if single_model: # single model
            model_number += 1
            pdb_lines.append(f"MODEL     {model_number}")
            for line in lines: 
                if line.startswith('TER') or line.startswith('ATOM'): 
                    pdb_lines.append(line.strip())
            pdb_lines.append("ENDMDL")
        else:        # multiple models
            for line in lines:
                if line.startswith('MODEL'):
                    model_number += 1
                    if model_number > 1:
                        pdb_lines.append("ENDMDL")
                    pdb_lines.append(f"MODEL     {model_number}")
                elif line.startswith('END'):
                    continue
                elif line.startswith('TER') or line.startswith('ATOM'): 
                    pdb_lines.append(line.strip())
    pdb_lines.append('ENDMDL')
    pdb_lines.append('END')
    pdb_lines = [_line.ljust(80) for _line in pdb_lines]
    pdb_str = '\n'.join(pdb_lines) + '\n'
    with open(output_file, 'w') as fo:
        fo.write(pdb_str)
    
    if verbose:
        print(f"Merged {len(pdb_files)} PDB files into {output_file} with {model_number} models.")


def split_pdbfile(pdb_file, output_dir=None, suffix='index', verbose=True):
    """Split a PDB file into multiple PDB files in output_dir.
    Preassume that each model is wrapped by 'MODEL' and 'ENDMDL'.
    """
    assert os.path.exists(pdb_file), f"File {pdb_file} does not exist."
    assert suffix == 'index', 'Only support [suffix=index] for now.'
    
    if output_dir is not None:  # also dump to output_dir
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(pdb_file))[0]
    
    i = 0
    pdb_strs = []
    pdb_string = ''
    with open(pdb_file, 'r') as fi:
        # pdb_string = ''
        for line in fi:
            if line.startswith('MODEL'):
                pdb_string = ''
            elif line.startswith('ATOM') or line.startswith('TER'):
                pdb_string += line
            elif line.startswith('ENDMDL') or line.startswith('END'):
                if pdb_string == '': continue
                pdb_string += 'END\n'
                if output_dir is not None:
                    _save_to = os.path.join(output_dir, f'{base}_{i}.pdb') if suffix == 'index' else None
                    with open(_save_to, 'w') as fo:
                        fo.write(pdb_string)
                pdb_strs.append(pdb_string)
                pdb_string = ''
                i += 1
            else:
                if verbose:
                    print(f"Warning: line '{line}' is not recognized. Skip.")
    if verbose:
        print(f">>> Split pdb {pdb_file} into {i}/{len(pdb_strs)} structures.")
    return pdb_strs


def stratify_sample_pdbfile(input_path, output_path, n_max_sample=1000, end_at=0, verbose=True):
    """ """
    assert os.path.exists(input_path), f"File {input_path} does not exist."
    assert not os.path.exists(output_path), f"Output path {output_path} already exists."
    
    i = 0
    pdb_strs = []
    with open(input_path, 'r') as fi:
        # pdb_string = ''
        pdb_lines_per_model = []
        for line in fi:
            if line.startswith('MODEL'):
                pdb_lines_per_model = [] 
            elif line.startswith('ATOM') or line.startswith('TER'):
                pdb_lines_per_model.append(line.strip())
            elif line.startswith('ENDMDL') or line.startswith('END'):
                if pdb_lines_per_model == []: continue  # skip empty model
                # wrap up the model
                pdb_lines_per_model.append('ENDMDL')
                # Pad all lines to 80 characters.
                pdb_lines_per_model = [_line.ljust(80) for _line in pdb_lines_per_model]
                pdb_str_per_model = '\n'.join(pdb_lines_per_model) + '\n'  # Add terminating newline.
                pdb_strs.append(pdb_str_per_model)
                # reset
                pdb_lines_per_model = []
                i += 1
            else:
                if verbose:
                    print(f"Warning: line '{line}' is not recognized. Skip.")
            if end_at > 0 and i > end_at:
                break
    end =  end_at if end_at > 0 else len(pdb_strs)
    
    # sample evenly
    if end > n_max_sample:    
        interleave_step = int(end // n_max_sample) # floor
        sampled_pdb_strs = pdb_strs[:end][::interleave_step][:n_max_sample]
    else:
        sampled_pdb_strs = pdb_strs[:end]
    
    output_str = ''
    for i, pdb_str  in enumerate(sampled_pdb_strs): # renumber models
        output_str += f"MODEL     {i+1}".ljust(80) + '\n'
        output_str += pdb_str
    output_str = output_str + ('END'.ljust(80) + '\n')
    
    write_pdb_string(output_str, save_to=output_path)
    if verbose: 
        print(f">>> Split pdb {input_path} into {len(sampled_pdb_strs)}/{n_max_sample} structures.")
    return
        

def protein_with_default_params(
    atom_positions: np.ndarray,
    atom_mask: np.ndarray,
    aatype: Optional[np.ndarray] = None,
    b_factors: Optional[np.ndarray] = None,
    chain_index: Optional[np.ndarray] = None,
    residue_index: Optional[np.ndarray] = None,
):
    assert atom_positions.ndim == 3
    assert atom_positions.shape[-1] == 3
    assert atom_positions.shape[-2] == 37
    n = atom_positions.shape[0]
    sqz = lambda x: np.squeeze(x) if x.shape[0] == 1 and len(x.shape) > 1 else x
    
    residue_index = np.arange(n) + 1 if residue_index is None else sqz(residue_index)
    chain_index = np.zeros(n) if chain_index is None else sqz(chain_index)
    b_factors = np.zeros([n, 37]) if b_factors is None else sqz(b_factors)
    aatype = np.zeros(n, dtype=int) if aatype is None else sqz(aatype)
        
    return protein.Protein(
        atom_positions=atom_positions,
        atom_mask=atom_mask,
        aatype=aatype,
        residue_index=residue_index, 
        chain_index=chain_index,
        b_factors=b_factors
    )

def atom37_to_pdb(
    save_to: str,
    atom_positions: np.ndarray,
    aatype: Optional[np.ndarray] = None,
    b_factors: Optional[np.ndarray] = None,
    chain_index: Optional[np.ndarray] = None,
    residue_index: Optional[np.ndarray] = None,
    overwrite: bool = False,
    no_indexing: bool = True,
):
    # configure save path
    if overwrite:
        max_existing_idx = 0
    else:
        file_dir = os.path.dirname(save_to)
        file_name = os.path.basename(save_to).strip('.pdb')
        existing_files = [x for x in os.listdir(file_dir) if file_name in x]
        max_existing_idx = max([
            int(re.findall(r'_(\d+).pdb', x)[0]) for x in existing_files if re.findall(r'_(\d+).pdb', x)
            if re.findall(r'_(\d+).pdb', x)] + [0])
    if not no_indexing:
        save_to = save_to.replace('.pdb', '') + f'_{max_existing_idx+1}.pdb'
    else:
        save_to = save_to
        
    with open(save_to, 'w') as f:
        if atom_positions.ndim == 4:
            for mi, pos37 in enumerate(atom_positions):
                atom_mask = np.sum(np.abs(pos37), axis=-1) > 1e-7
                prot = protein_with_default_params(
                    pos37, atom_mask, aatype=aatype, b_factors=b_factors, 
                    chain_index=chain_index, residue_index=residue_index
                )
                pdb_str = protein.to_pdb(prot, model=mi+1, add_end=False)
                f.write(pdb_str)
        elif atom_positions.ndim == 3:
            atom_mask = np.sum(np.abs(atom_positions), axis=-1) > 1e-7
            prot = protein_with_default_params(
                atom_positions, atom_mask, aatype=aatype, b_factors=b_factors, 
                chain_index=chain_index, residue_index=residue_index
            )
            pdb_str = protein.to_pdb(prot, model=1, add_end=False)
            f.write(pdb_str)
        else:
            raise ValueError(f'Invalid positions shape {atom_positions.shape}')
        f.write('END')
        
    return save_to


def extract_backbone_coords_from_pdb(pdb_path: str, target_atoms: Optional[list] = ["CA"]):
    structure = PDBFile.read(pdb_path)
    structure_list = structure.get_structure()
    
    coords_list = []
    for b_idx in range(structure.get_model_count()):
        chain = structure_list[b_idx]

        backbone_atoms = chain[struct.filter_backbone(chain)]   # This includes the “N”, “CA” and “C” atoms of amino acids.
        ret_coords = OrderedDict()
        # init dict
        for k in target_atoms:
            ret_coords[k] = []
            
        for c in backbone_atoms:
            if c.atom_name in ret_coords:
                ret_coords[c.atom_name].append(c.coord)
                
        ret_coords = [np.vstack(v) for k,v in ret_coords.items()]
        if len(target_atoms) == 1:
            ret_coords = ret_coords[0]  # L, 3
        else:
            ret_coords = np.stack(ret_coords, axis=1)   # L, na, 3
        
        coords_list.append(ret_coords)
    
    coords_list = np.stack(coords_list, axis=0) # B, L, na, 3 or B, L, 3 (ca only)
    return coords_list


def extract_backbone_coords_from_pdb_dir(pdb_dir: str):
    return np.concatenate([
            extract_backbone_coords_from_pdb(os.path.join(pdb_dir, f))
                for f in os.listdir(pdb_dir) if f.endswith('.pdb')
        ], axis=0) 

def extract_backbone_coords_from_npy(npy_path: str):
    return np.load(npy_path)


def extract_backbone_coords(input_path: str, 
                            max_n_model: Optional[int] = None,
):
    """Extract backbone coordinates from PDB file.
    
    Args:
        input_path (str): The path to the PDB file.
        ca_only (bool): Whether to extract only CA coordinates.
        max_n_model (int): The maximum number of models to extract.
    """
    assert os.path.exists(input_path), f"File {input_path} does not exist."
    if input_path.endswith('.pdb'):
        coords = extract_backbone_coords_from_pdb(input_path)
    elif input_path.endswith('.npy'):
        coords = extract_backbone_coords_from_npy(input_path)
    elif os.path.isdir(input_path):
        coords = extract_backbone_coords_from_pdb_dir(input_path)
    else:
        raise ValueError(f"Unrecognized input path {input_path}.")

    if max_n_model is not None and len(coords) > max_n_model > 0:
        coords = coords[:max_n_model]
    return coords



if __name__ == '__main__':
    import argparse
    def get_argparser():
        parser = argparse.ArgumentParser(description='Main script for pdb processing.')
        parser.add_argument("input", type=str, help="The generic path to sampled pdb directory / pdb file.")
        parser.add_argument("-m", "--mode", type=str, help="The mode of processing.",
                            default="split")
        parser.add_argument("-o", "--output", type=str, help="The output directory for processed pdb files.",
                            default=None)

        args = parser.parse_args()
        return args 
    args = get_argparser()
    
    # ad hoc functions
    def split_pdbs(args):
        os.makedirs(args.output, exist_ok=True)
        _ = split_pdbfile(pdb_file=args.input,
                        output_dir=args.output)
            
    def merge_pdbs(args):
        output = args.output or f"{args.input}_all.pdb"
        merge_pdbfiles(input=args.input,
                        output_file=output)
        
    if args.mode == "split":
        split_pdbs(args)
    elif args.mode == "merge":
        merge_pdbs(args)
    elif args.mode == "stratify":
        stratify_sample_pdbfile(input_path=args.input, output_path=args.output)
    else:
        raise ValueError(f"Unrecognized mode {args.mode}.")