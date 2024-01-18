"""Fold protein sequences into structures using ESMFold.
To use this script, you may need to install it using pip:
 
pip install "fair-esm[esmfold]"
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
# module load cuda/11.3
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
"""

import os
import argparse
from glob import glob

import pandas as pd
import torch
from Bio import PDB
import esm

# The path to the directory where the model weights are cached.
torch.hub.set_dir(os.path.join(os.getenv('SCRATCH'), '.cache/torch/'))

def get_argparser():
    parser = argparse.ArgumentParser(description='Main script for pdb processing.')
    parser.add_argument("-i", "--input", type=str, help="The path to sampled pdb directory / fasta file.")
    parser.add_argument("-o", "--output", type=str, help="The output directory for processed pdb files.", default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    return args


def read_sequence_from_pdbs(path):
    def extract_amino_acid_sequence(structure):
        sequence = ""
        for model in structure:
            for chain in model:
                for residue in chain:
                    if PDB.Polypeptide.is_aa(residue):
                        sequence += PDB.Polypeptide.three_to_one(residue.get_resname())
        return sequence

    pdb_parser = PDB.PDBParser(QUIET=True)
    if not os.path.isdir(path):
        assert os.path.isfile(path) and path.endswith('.pdb'), f"Path {path} is neither a directory nor a file."
        pdb_paths = [path]
    else:
        pdb_paths = glob(os.path.join(path, "*.pdb"))

    name_to_seqs = {}
    for pdb_path in pdb_paths:
        pdb_name = os.path.basename(pdb_path).replace(".pdb", "")
        structure = pdb_parser.get_structure(pdb_name, pdb_path)
        seq = extract_amino_acid_sequence(structure)
        name_to_seqs[pdb_name] = seq
    return name_to_seqs


def read_sequence_from_fasta(path):
    with open(path, "r") as f:
        lines = f.readlines()
    name_to_seqs = {}
    pdb_name = None
    seq = ""
    for line in lines:
        if line.startswith(">"):
            if pdb_name is not None:
                name_to_seqs[pdb_name] = seq
            pdb_name = line[1:].strip()
            seq = ""
        else:
            seq += line.strip()
    
    if pdb_name is not None:    # last seq
        name_to_seqs[pdb_name] = seq
        
    return name_to_seqs


def inference(args):
    # Load ESMFold_v1 model.
    model = esm.pretrained.esmfold_v1()
    if args.device is None:
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(">>> By default, set device to be ", args.device)
    model = model.eval().to(args.device)

    if os.path.isdir(args.input):
        # a directory of pdb files
        name_to_seqs = read_sequence_from_pdbs(args.input)
    elif args.input.endswith(".pdb"):
        name_to_seqs = read_sequence_from_pdbs(args.input)
    elif args.input.endswith(".fasta") or args.input.endswith('fa'):
        name_to_seqs = read_sequence_from_fasta(args.input)
        
    print(f">>> [inference] Loaded {len(name_to_seqs)} sequences as input")
    
    # Configure output directory.
    if args.output is None:
        tmp_dir = args.input[:-1] if args.input[-1] == "/" else args.input
        args.output = tmp_dir + "_esmfolded"
    os.makedirs(args.output, exist_ok=True)
    print(">>> [inference] Folded structures will be saved to ", args.output)
    
    # Multimer prediction can be done with chains separated by ':'
    for pdb_name, seq in name_to_seqs.items():
        with torch.no_grad():
            output = model.infer_pdb(seq)
        save_to = os.path.join(args.output, f"{pdb_name}.pdb")
        with open(save_to, "w") as f:
            f.write(output)


if __name__ == '__main__':
    args = get_argparser()
    inference(args)