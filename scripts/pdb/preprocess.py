"""Pre-processing downloaded mmcif files as pickled file for faster loading during training.

It supports filtering by:

1. Resolution (recorded in PDB).
2. PISCES cluster file (a clustered subset of PDB). 
    (Other cluster files can be used as well but the code may be adapted accordingly)

and additional features:
1. DSSP secondary structure.
2. Radius of gyration.

*This script is inspired by https://github.com/jasonkyuyim/se3_diffusion/blob/master/data/process_pdb_dataset.py.*
"""

import argparse
import string
import os
import time
import pickle
import collections
import dataclasses
import multiprocessing as mp
from functools import partial
from glob import glob
from typing import List, Dict, Any, Optional
from tempfile import NamedTemporaryFile

from tqdm import tqdm
import mdtraj as md
import numpy as np
import pandas as pd
from Bio.PDB.Chain import Chain

import errors
import mmcif_parsing

import src.common.residue_constants as rc
from src.common import protein


# Global map from chain characters to integers. e.g, A -> 0, B -> 1, etc.
ALPHANUMERIC = string.ascii_letters + string.digits + ' '
CHAIN_TO_INT = {
    chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)
}
INT_TO_CHAIN = {
    i: chain_char for i, chain_char in enumerate(ALPHANUMERIC)
}
CHAIN_FEATS = [
    'atom_positions', 'aatype', 'atom_mask', 'residue_index', 'chain_index', 'b_factors',
]   # original chain features


def sanity_check(prot):
    chain_feats = dataclasses.asdict(prot)
    length = chain_feats['aatype'].shape[0]
    for k,v in chain_feats.items():
        if v.shape[0] != length:
            print(f'Error: {k} has length {v.shape[0]}')
            return False
        if k not in CHAIN_FEATS:
            print(f'Error: {k} not in CHAIN_FEATS')
            return False
    for k in CHAIN_FEATS:
        if k not in chain_feats:
            print(f'Error: {k} not in chain_feats')
            return False
    return True

def chain_str_to_int(chain_str: str):
    chain_int = 0
    if len(chain_str) == 1:
        return CHAIN_TO_INT[chain_str]
    for i, chain_char in enumerate(chain_str):
        chain_int += CHAIN_TO_INT[chain_char] + (i * len(ALPHANUMERIC))
    return chain_int

def write_to_pickle(save_path: str, pkl_data: Any, create_dir: bool = False, use_torch=False):
    if create_dir:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if use_torch:
        torch.save(pkl_data, save_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(save_path, 'wb') as handle:
            pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_mmcif_paths(
    mmcif_dir: str, 
    max_file_size: int, 
    min_file_size: int, 
    debug: bool,
    target_pdb_ids: List[str] = None,
):
    """Set up all the mmcif files to read."""
    mmcif_dir = os.path.expanduser(mmcif_dir)
    print('>>> Globbing mmCIF paths w/ [file-size] filter')
    all_mmcif_paths = glob(os.path.join(mmcif_dir, '**/*.cif'), recursive=True)
    if debug:
        # Don't process all files for debugging
        all_mmcif_paths = all_mmcif_paths[:20]
    
    if target_pdb_ids is not None:
        # Filter by target pdb ids
        all_mmcif_paths = [
            p for p in all_mmcif_paths
            if os.path.basename(p)[:4].upper() in target_pdb_ids
        ]
    
    filtered_mmcif_paths = [
        p for p in all_mmcif_paths
        if min_file_size <= os.path.getsize(p) <= max_file_size
    ]    
    print(f'>>> Select {len(filtered_mmcif_paths)} files out of {len(all_mmcif_paths)} by file size (debug={debug})')
    return filtered_mmcif_paths

def parse_pisces_subset(path_to_pisces: str):
    """Parse PISCES list file into a set of pdb ids."""
    df = pd.read_csv(path_to_pisces, sep='\s+')
    pdbchain = df['PDBchain'].tolist()
    pdb_ids = [x[:4] for x in pdbchain]
    pdb_and_chain_ids = [f"{x[:4]}_{x[4:]}" for x in pdbchain]
    return pdb_ids, pdb_and_chain_ids

def concat_chain_features(chain_feats: List[Dict[str, np.ndarray]]):
    """Performs a nested concatenation of feature dicts.

    Args:
        chain_feats: list of dicts with the same structure.
            Each dict must have the same keys and numpy arrays as the values.

    Returns:
        A single dict with all the features concatenated.
    """
    combined_dict = collections.defaultdict(list)
    for chain_dict in chain_feats:
        for feat_name, feat_val in chain_dict.items():
            combined_dict[feat_name].append(feat_val)
    # Concatenate each feature
    for feat_name, feat_vals in combined_dict.items():
        combined_dict[feat_name] = np.concatenate(feat_vals, axis=0)
    return combined_dict


def instantiate_protein(chain: Chain, chain_id: str) -> protein.Protein:
    """Convert a PDB chain object into a AlphaFold Protein instance.
    
    Forked from alphafold.common.protein.from_pdb_string
    
    WARNING: All non-standard residue types will be converted into UNK. All
        non-standard atoms will be ignored.
    
    Took out lines 94-97 which don't allow insertions in the PDB.
    Sabdab uses insertions for the chothia numbering so we need to allow them.
    
    Took out lines 110-112 since that would mess up CDR numbering.
    
    Args:
        chain: Instance of Biopython's chain class.
    
    Returns:
        Protein object with protein features.
    """
    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    b_factors = []
    chain_ids = []
    for res in chain:
        res_shortname = rc.restype_3to1.get(res.resname, 'X')
        restype_idx = rc.restype_order.get(
            res_shortname, rc.restype_num)
        pos = np.zeros((rc.atom_type_num, 3))
        mask = np.zeros((rc.atom_type_num,))
        res_b_factors = np.zeros((rc.atom_type_num,))
        for atom in res:
            if atom.name not in rc.atom_types:
                continue
            pos[rc.atom_order[atom.name]] = atom.coord
            mask[rc.atom_order[atom.name]] = 1.
            res_b_factors[rc.atom_order[atom.name]
                          ] = atom.bfactor
        aatype.append(restype_idx)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(res.id[1])
        b_factors.append(res_b_factors)
        chain_ids.append(chain_id)

    return protein.Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=np.array(chain_ids),
        b_factors=np.array(b_factors),
    )

def compute_dssp_feats(prot: Dict[str, np.ndarray]):
    assert sanity_check(prot), 'Error: sanity check failed'
    # try:
    # Workaround for MDtraj not supporting mmcif in their latest release.
    # MDtraj source does support mmcif https://github.com/mdtraj/mdtraj/issues/652
    # We temporarily save the mmcif as a pdb and delete it after running mdtraj.
    pdb_string = protein.to_pdb(prot)
    
    with NamedTemporaryFile(mode='a', suffix=".pdb") as tmp:
        tmp.write(pdb_string)
        tmp.seek(0)
        traj = md.load(tmp.name)
    # SS calculation
    ss = md.compute_dssp(traj, simplified=True)
    # Radius of gyration calculation
    rg = md.compute_rg(traj)
    # except Exception as e:
    #     raise errors.DataProcessingError(f'Mdtraj failed with error {e}')
    
    pdb_ss  = ss[0] # (L, )
    data_dict = dict(
        coil_percent = np.sum(pdb_ss == 'C') / len(pdb_ss),
        helix_percent = np.sum(pdb_ss == 'H') / len(pdb_ss),
        strand_percent = np.sum(pdb_ss == 'E') / len(pdb_ss),
        radius_gyration = rg[0],
    )
   
    return pdb_ss, data_dict
    
def strip_feats_by_modeled_idx(
    chain_dict: Dict[str, np.ndarray],
    min_idx: int,
    max_idx: int,                   
):
    assert min_idx <= max_idx, f'Error: min_idx {min_idx} > max_idx {max_idx}'
    chain_dict = {
        k: v[min_idx: max_idx + 1] for k, v in chain_dict.items()
    }
    return chain_dict

def process_mmcif_file(
    mmcif_path: str, 
    *,
    max_resolution: int, 
    output_dir: str,
    target_chains: List[str] = None,
    mode: str = 'complex',
    strip_array: bool = True,
    compute_ss: bool = True,
    verbose=False,
    target_pdb_and_chain_ids=None
):
    """Processes a single MMCIF file into processed pickles.
    Saves processed protein to pickle and returns metadata.

    Args:
        mmcif_path: Path to mmcif file to read.
        max_resolution: Max resolution to allow.
        output_dir: Directory to write pickles to.
        target_chains: List of chain ids to process. If None, will process all chains.
        mode: 'complex' or 'chain'. If 'complex', will process all chains into a single pickle.
            If 'chain', will process and save each chain separately.
        compute_ss: Whether to compute secondary structure. (slow)

    Returns:
        metadata.

    Raises:
        DataProcessingError (defined in error.py) will be caught and logged.
        All other errors are unexpected and are thrown as-is.
    """
    metadata = {}
    mmcif_name = os.path.basename(mmcif_path).replace('.cif', '')
    metadata['pdb_name'] = mmcif_name
    mmcif_subdir = os.path.join(output_dir, mmcif_name[1:3].lower()) # tree directory
    
    if not os.path.isdir(mmcif_subdir):
        os.mkdir(mmcif_subdir)
    
    if target_pdb_and_chain_ids is not None:
        assert mode == 'chain', "Error: w/ identified <target_chains> only works in 'chain' mode"
    
    # parse mmcif
    try:
        with open(mmcif_path, 'r') as f:
            parsed_mmcif = mmcif_parsing.parse(
                file_id=mmcif_name, mmcif_string=f.read())
    except:
        raise errors.FileExistsError(
            f'Error file do not exist {mmcif_path}'
        )    
    metadata['raw_path'] = mmcif_path
    if parsed_mmcif.errors:
        raise errors.MmcifParsingError(
            f'Encountered errors {parsed_mmcif.errors}'
        )
    parsed_mmcif = parsed_mmcif.mmcif_object
    raw_mmcif = parsed_mmcif.raw_string
    # parse oligomeric state
    if '_pdbx_struct_assembly.oligomeric_count' in raw_mmcif:
        raw_olig_count = raw_mmcif['_pdbx_struct_assembly.oligomeric_count']
        oligomeric_count = ','.join(raw_olig_count).lower()
    else:
        oligomeric_count = None
    if '_pdbx_struct_assembly.oligomeric_details' in raw_mmcif:
        raw_olig_detail = raw_mmcif['_pdbx_struct_assembly.oligomeric_details']
        oligomeric_detail = ','.join(raw_olig_detail).lower()
    else:
        oligomeric_detail = None
    metadata['oligomeric_count'] = oligomeric_count
    metadata['oligomeric_detail'] = oligomeric_detail

    ##############################
    # global filters
    ##############################
    # Parse mmcif header
    mmcif_header = parsed_mmcif.header
    mmcif_resolution = mmcif_header['resolution']
    metadata['resolution'] = mmcif_resolution
    metadata['structure_method'] = mmcif_header['structure_method']
    if mmcif_resolution >= max_resolution:
        raise errors.ResolutionError(
            f'Too high resolution {mmcif_resolution} > {max_resolution}'
        )
    if mmcif_resolution == 0.0:
        raise errors.ResolutionError(
            f'Invalid resolution {mmcif_resolution}'
        )
    # Extract all chains
    struct_chains = {
        chain.id.upper(): chain
        for chain in parsed_mmcif.structure.get_chains()}
    metadata['num_chains'] = len(struct_chains)
    
    # Extract features per chain
    chain_metadatas = []
    chain_dicts = {}
    all_raw_seqs = set()
    for chain_id, chain in struct_chains.items():   ### ITERATE OVER CHAINS ###
        chain_metadata = {}
        pdb_chain_name = f"{mmcif_name}_{chain_id}"
        
        if target_pdb_and_chain_ids is not None:
            if pdb_chain_name.upper() not in target_pdb_and_chain_ids:
                continue    
        processed_chain_path = os.path.abspath(os.path.join(mmcif_subdir, f'{pdb_chain_name}.pkl'))
        
        # Get protein object
        chain_id = chain_str_to_int(chain_id)
        chain_prot = instantiate_protein(chain, chain_id)
        
        # Convert to dict
        chain_dict = chain_prot.to_dict()

        # Process geometry features
        chain_aatype = chain_dict['aatype']
        modeled_idx = np.where(chain_aatype != 20)[0]
        
        raw_tup_seq = tuple(chain_aatype)
        all_raw_seqs.add(raw_tup_seq)
    
        if mode == 'chain':
            chain_metadata['processed_path'] = processed_chain_path
            chain_metadata['pdb_chain_name'] = pdb_chain_name
            
            # filter and prevent reduce ops as well
            if np.sum(chain_aatype != 20) == 0:
                print(f'Warning: No modeled residues in {pdb_chain_name}')
                continue
            
            min_modeled_idx = np.min(modeled_idx)
            max_modeled_idx = np.max(modeled_idx)
            
            chain_metadata['raw_seq_len'] = len(chain_aatype)
            chain_metadata['modeled_seq_len'] = max_modeled_idx - min_modeled_idx + 1
            
            if strip_array:
                chain_dict = strip_feats_by_modeled_idx(
                    chain_dict, min_modeled_idx, max_modeled_idx)   # -> (modeled_seq_len, *)
                
            if compute_ss:
                chain_prot= protein.Protein(**chain_dict)
                chain_ss, ss_info = compute_dssp_feats(chain_prot)
                chain_dict['ss'] = chain_ss
                chain_metadata.update(ss_info)
                assert len(chain_ss) == len(chain_dict['aatype']), f"Error: ss len {len(chain_ss)} != aatype len {chain_dict['aatype']}"
        
        chain_dicts[processed_chain_path] = chain_dict
        chain_metadatas.append(chain_metadata)
        
    # Process complex features
    metadata['quaternary_category'] = 'homomer' if len(all_raw_seqs) == 1 else 'heteromer'
    if len(chain_metadatas) == 0:
        raise errors.DataProcessingError(f'No chains founded in {mmcif_path}')
    
    if mode == 'complex':
        del chain_metadatas
        processed_complex_path = os.path.abspath(os.path.join(mmcif_subdir, f'{mmcif_name}.pkl'))
        metadata['processed_complex_path'] = processed_complex_path
        
        complex_feats = concat_chain_features(list(chain_dicts.values()))
        complex_aatype = complex_feats['aatype']
        modeled_idx = np.where(complex_aatype != 20)[0]
        min_modeled_idx = np.min(modeled_idx)
        max_modeled_idx = np.max(modeled_idx)
        
        metadata['raw_seq_len'] = len(complex_aatype)
        metadata['modeled_seq_len'] = max_modeled_idx - min_modeled_idx + 1
        
        if strip_array:
            complex_feats = strip_feats_by_modeled_idx(
                complex_feats, min_modeled_idx, max_modeled_idx)

        if compute_ss:
            complex_prot = protein.Protein(**complex_feats)
            complex_ss, ss_info = compute_dssp_feats(complex_prot)
            complex_feats['ss'] = complex_ss
            
            assert len(chain_ss) == len(chain_dict['aatype']), f"Error: ss len {len(chain_ss)} != aatype len {chain_dict['aatype']}"
            
            metadata.update(ss_info)
        
        write_to_pickle(processed_complex_path, complex_feats)
        return [metadata]
    elif mode == 'chain':
        for processed_chain_path, chain_dict in chain_dicts.items():
            write_to_pickle(processed_chain_path, chain_dict)
        for chain_metadata in chain_metadatas:
            chain_metadata.update(metadata)
        return chain_metadatas
    else:
        raise ValueError(f'Invalid mode {mode}')
    
def process_fn(
    mmcif_path: str,
    max_resolution: Optional[float] = None,
    output_dir: Optional[str] = None,
    per_chain: Optional[bool] = True,
    strip_array: Optional[bool] = True,
    compute_ss: Optional[bool] = False,
    verbose: Optional[bool] = False,
    target_pdb_and_chain_ids: Optional[List] = None, 
):
    """Wrapper for process_mmcif_file to allow for multiprocessing."""
    mode = 'chain' if per_chain else 'complex'
    try:
        start_time = time.time()
        metadata = process_mmcif_file(
            mmcif_path,
            max_resolution=max_resolution,
            mode=mode,
            compute_ss=compute_ss,
            strip_array=strip_array,
            output_dir=output_dir,
            target_pdb_and_chain_ids=target_pdb_and_chain_ids
        )
        elapsed_time = time.time() - start_time
        if verbose:
            print(f'Finished {mmcif_path} in {elapsed_time:2.2f}s')
        return metadata
    except errors.DataProcessingError as e:
        if verbose:
            print(f'Failed {mmcif_path}: {e}')


def main(args):
    if args.pisces is not None:
        # Filter by PISCES cluster
        target_pdb_ids, target_pdb_and_chain_ids = parse_pisces_subset(args.pisces)
        print(f'Filtering by PISCES cluster with {len(target_pdb_ids)} pdb ids')
    else:
        target_pdb_ids, target_pdb_and_chain_ids = None, None
        
    # Get all mmcif files to read.
    all_mmcif_paths = get_mmcif_paths(args.mmcif_dir, 
                                    args.max_file_size, 
                                    args.min_file_size, 
                                    args.debug,
                                    target_pdb_ids=target_pdb_ids,
    )
    num_mmcif_paths = len(all_mmcif_paths)
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.debug:
        metadata_file_name = 'metadata_debug.csv'
        args.verbose = True
    else:
        metadata_file_name = 'metadata.csv'   
    metadata_path = os.path.join(output_dir, metadata_file_name)
    print(f'Files will be written to {output_dir}')

    # Process each mmcif file
    _process_fn = partial(
            process_fn,
            max_resolution=args.max_resolution,
            output_dir=output_dir,
            per_chain=args.per_chain,
            strip_array=args.strip_array,
            compute_ss=args.compute_ss,
            verbose=args.verbose,
            target_pdb_and_chain_ids=target_pdb_and_chain_ids,
    )
    all_metadata = []
    if args.num_processes == 1 or args.debug:
        for mmcif_path in all_mmcif_paths:
            metadata = _process_fn(mmcif_path)
            metadata = metadata if metadata is not None else []
            all_metadata.extend(metadata)
    else:
        # Uses max number of available cores.
        with mp.Pool() as pool:
            _all_metadata = pool.map(_process_fn, all_mmcif_paths)
        for list_data in _all_metadata:
            if list_data is not None:
                all_metadata.extend([x for x in list_data if x is not None])
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_path, index=False)
    processed_md = len(all_metadata)
    print(f'Finished processing {processed_md}/{num_mmcif_paths} files')


def get_args():
    # Define the parser
    parser = argparse.ArgumentParser(description='mmCIF processing script.')
    
    parser.add_argument('--mmcif_dir', help='Path to directory with mmcif files.', type=str)
    parser.add_argument('--output_dir', help='Path to write results to.', type=str, 
                        default='./data/processed_pdb')
    parser.add_argument('--max_file_size', help='Max file size.', type=int, 
                        default=300000000)  # Only process files up to 300MB large.
    parser.add_argument('--min_file_size', help='Min file size.', type=int, 
                        default=100)  # Files must be at least 0.1KB.
    parser.add_argument('--max_resolution', help='Max resolution of files.', type=float, 
                        default=9.0)    # AF2
    parser.add_argument('--num_processes', help='Number of processes. (Set to be 1 if serially)', type=int,
                        default=32)
    parser.add_argument('--per_chain', help='Whether to process single chain instead of complex.', 
                        action='store_true')
    parser.add_argument('--strip_array', help='Whether to strip array.', 
                        action='store_true')
    parser.add_argument('--compute_ss', help='Whether to process single chain instead of complex.', 
                        action='store_true')
    parser.add_argument('--pisces', help='Path to the cluster file to prefilter the result.', type=str,
                        default=None)    
    parser.add_argument('--debug', help='Turn on for debugging.',  
                        action='store_true')
    parser.add_argument('--verbose', help='Whether to log everything.',
                        action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    # Don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = get_args()
    main(args)