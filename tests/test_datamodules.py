from pathlib import Path

import pytest
import torch

from src.data.protein_datamodule import ProteinDataModule
from src.data.components.dataset import (PretrainPDBDataset, ProteinFeatureTransform, MetadataFilter)

@pytest.mark.parametrize("batch_size", [1, 4])
def test_mnist_datamodule(batch_size: int) -> None:
    """Tests `ProteinDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    dataset = PretrainPDBDataset("data/Science2011_esmfolded",
                MetadataFilter(max_len=500),
                ProteinFeatureTransform(
                    strip_missing_residues=False, 
                    truncate_length=None,
                    recenter_and_scale=True
                ),
                suffix='.pdb',
    )
    dm = ProteinDataModule(dataset, train_val_split=(0.5, 0.5), batch_size=batch_size)
    dm.prepare_data()
        
    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup('fit')
    assert dm.data_train and dm.data_val 
    assert dm.train_dataloader() and dm.val_dataloader() 

    num_datapoints = len(dm.data_train) + len(dm.data_val)
    assert num_datapoints == 12

    batch = next(iter(dm.train_dataloader()))
    x, y = batch['atom_positions'], batch['aatype']
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float64
    assert y.dtype == torch.int64
