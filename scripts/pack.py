"""Script to pack the side chains of a backbone pdb file using FASPR.

Installation:

git clone https://github.com/tommyhuangthu/FASPR
cd FASPR 
g++ -O3 --fast-math -o FASPR src/*.cpp

"""

import sys
import os
import argparse
import subprocess
import tempfile
from glob import glob
from time import time
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict

from src.utils.multiprocs import mp_map
from src.common.pdb_utils import split_pdbfile, merge_pdbfiles


def get_args():
    parser = argparse.ArgumentParser(description='Main script for pdb processing.')
    parser.add_argument("-i", "--input", type=str, help="The generic path to sampled pdb directory / pdb file.")
    parser.add_argument("-o", "--output", type=str, help="The output directory for processed pdb files.", default=None)
    parser.add_argument("--faspr_bin", type=str, default="FASPR")
    parser.add_argument("--n_cpu", type=int, default=1, help="Number of cpus to use.")
    args = parser.parse_args()
    return args


def timing(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.5f}s to execute.")
        return result
    return wrapper


def exec_subproc(cmd, timeout: int = 100) -> str:
    """Execute the external docking-related command.
    """
    if not isinstance(cmd, str):
        cmd = ' '.join([str(entry) for entry in cmd])
    
    try:
        rtn = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, timeout=timeout)
        out, err, return_code = str(rtn.stdout, 'utf-8'), str(rtn.stderr, 'utf-8'), rtn.returncode
        if return_code != 0:
            logging.error('[ERROR] Execuete failed with command "' + cmd + '", stderr: ' + err)
            raise ValueError('Return code is not 0, check input files')
        return out
    except subprocess.TimeoutExpired:
        logging.error('[ERROR] Execuete failed with command ' + cmd)
        raise ValueError(f'Timeout({timeout})')
    

def process_fn(input_output, faspr_bin):
    """single model processing"""
    input_path, output_path = input_output
    input_path = os.path.abspath(input_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        out = exec_subproc([faspr_bin, '-i', input_path, '-o', output_path])
        return output_path
    except Exception as e:
        print(f'warning: error thrown processing {input_path} as {e}')
    return None

@timing
def faspr_pack_multimodel(pdb_paths, output_dir, faspr_bin, n_cpu=8):
    """processing pdb file with multiple models"""
    use_mp = n_cpu > 1
    _fn = partial(process_fn, faspr_bin=faspr_bin)
    
    with tempfile.TemporaryDirectory(suffix=None, prefix=None, dir=None) as tmpdir:
        pairs_to_process = []   # total structures to process
        for pdb_path in pdb_paths:
            input_path = os.path.abspath(pdb_path)  # input pdb
            pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]

            split_dir = os.path.join(tmpdir, 'split', pdb_id)
            tmp_output_dir = os.path.join(tmpdir, 'output', pdb_id)
            os.makedirs(split_dir, exist_ok=True)
            os.makedirs(tmp_output_dir, exist_ok=True)
            
            _ = split_pdbfile(input_path, output_dir=split_dir, verbose=False) # .../base_i.pdb from 0
            pairs_to_process += [(os.path.join(split_dir, f), os.path.join(tmp_output_dir, f)) 
                                    for f in os.listdir(split_dir) if f.endswith('.pdb')]
        
        if len(pairs_to_process) < 10:  # too few models, use single process
            use_mp = False
            
        if use_mp:
            results = mp_map(_fn, pairs_to_process, n_cpu=min(n_cpu, len(pairs_to_process)))
        else:
            results = [_fn(pair) for pair in pairs_to_process]
            
        results = [r for r in results if r is not None]
        print(f">>> Successfully packed {len(results)} files.")
        
        valid_pdb_ids = list(set([os.path.basename(os.path.dirname(r)) for r in results]))
        for pdb_id in valid_pdb_ids:
            paths_to_merge = [r for r in results if os.path.basename(os.path.dirname(r)) == pdb_id]
            output_path = os.path.join(output_dir, f"{pdb_id}.pdb")
            merge_pdbfiles(paths_to_merge, output_path, verbose=False)        


if __name__ == '__main__':
    args = get_args()
    assert os.path.exists(args.input), f"Input path {args.input} does not exist."
    
    if args.output is None:
        tmp_dir = args.input[:-1] if args.input[-1] == "/" else args.input
        args.output = tmp_dir + "_faspr_packed"
    os.makedirs(args.output, exist_ok=True)
    
    if os.path.isdir(args.input):
        pdb_paths = glob(os.path.join(args.input, "*.pdb"))
    elif args.input.endswith(".pdb"):
        pdb_paths = [args.input]
    else:
        raise ValueError(f"Input path {args.input} is neither a directory nor a pdb file.") 
    
    faspr_pack_multimodel(pdb_paths, output_dir=args.output, faspr_bin=args.faspr_bin, n_cpu=args.n_cpu)
    