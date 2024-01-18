import os
from typing import Any, Dict, Tuple, Optional
from random import random
from copy import deepcopy

import numpy as np
import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric

from src.models.score.frame import FrameDiffuser
from src.models.loss import ScoreMatchingLoss
from src.common.rigid_utils import Rigid
from src.common.all_atom import compute_backbone
from src.common.pdb_utils import atom37_to_pdb, merge_pdbfiles


class DiffusionLitModule(LightningModule):
    """Example of a `LightningModule` for denoising diffusion training.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        diffuser: FrameDiffuser,
        loss: Dict[str, Any],
        compile: bool,
        inference: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # network and diffusion module
        self.net = net
        self.diffuser = diffuser
        
        # loss function
        self.loss = ScoreMatchingLoss(config=self.hparams.loss)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        # self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`. 
        (Not actually used)

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], training: Optional[bool] = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        # preprocess by augmenting additional feats
        rigids_0 = Rigid.from_tensor_4x4(batch['rigidgroups_gt_frames'][..., 0, :, :])
        batch_size = rigids_0.shape[0]
        t = (1.0 - self.diffuser.min_t) * torch.rand(batch_size, device=rigids_0.device) + self.diffuser.min_t
        perturb_feats = self.diffuser.forward_marginal(
            rigids_0=rigids_0,
            t=t,
            diffuse_mask=None,
            as_tensor_7=True,
        )
        patch_feats = {
            't': t,
            'rigids_0': rigids_0,
        }
        batch.update({**perturb_feats, **patch_feats})
        
        # probably add self-conditioning (recycle once)
        if self.net.embedder.self_conditioning and random() > 0.5:
            with torch.no_grad():
                batch['sc_ca_t'] = self.net(batch, as_tensor_7=True)['rigids'][..., 4:]

        # feedforward
        out = self.net(batch)
        
        # postprocess by add score computation
        pred_scores = self.diffuser.score(
            rigids_0=out['rigids'],
            rigids_t=Rigid.from_tensor_7(batch['rigids_t']),
            t=t,
            mask=batch['residue_mask'],
        )
        out.update(pred_scores)
        
        # calculate losses
        loss, loss_bd = self.loss(out, batch, _return_breakdown=True)
        return loss, loss_bd

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, loss_bd = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        for k,v in loss_bd.items():
            if k == 'loss': continue
            self.log(f"train/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, loss_bd = self.model_step(batch, training=False)

        # update and log metrics
        self.val_loss(loss) # update
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        _vall = self.val_loss.compute()  # get current val acc
        self.val_loss_best(_vall)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        raise NotImplementedError("Test step not implemented.")

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int,
    ) -> str:
        """Perform a prediction step on a batch of data from the dataloader.
        
        This prediction step will sample `n_replica` copies from the forward-backward process,
            repeated for each delta-T in the range of [delta_min, delta_max] with step size
            `delta_step`. If `backward_only` is set to True, then only backward process will be
            performed, and `n_replica` will be multiplied by the number of delta-Ts.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        # extract hyperparams for inference
        n_replica = self.hparams.inference.n_replica
        replica_per_batch = self.hparams.inference.replica_per_batch
        delta_range = np.arange(
            self.hparams.inference.delta_min,
            self.hparams.inference.delta_max + 1e-5,
            self.hparams.inference.delta_step
        )
        delta_range = np.around(delta_range, decimals=2)  # up to 2 decimal places
        num_timesteps = self.hparams.inference.num_timesteps
        noise_scale = self.hparams.inference.noise_scale
        probability_flow = self.hparams.inference.probability_flow
        self_conditioning = self.hparams.inference.self_conditioning and self.net.embedder.self_conditioning
        min_t = self.hparams.inference.min_t
        output_dir = self.hparams.inference.output_dir
        backward_only = self.hparams.inference.backward_only
        # if backward_only, then only perform backward process (vanilla sampling of diffusion)
        if backward_only:
            n_replica *= len(delta_range)
            delta_range = [-1.0]
        
        assert batch['aatype'].shape[0] == 1, "Batch size must be 1 for correct inference."
        
        # get extra features of the current protein
        accession_code = batch['accession_code'][0]
        extra = {
            'aatype': batch['aatype'][0].detach().cpu().numpy(),
            'chain_index': batch['chain_index'][0].detach().cpu().numpy(),
            'residue_index': batch['residue_index'][0].detach().cpu().numpy(),
        }
        
        # define sampling subroutine
        def forward_backward(rigids_0: Rigid, t_delta: float):
            # if t_delta <= 0 (invalid), then perform backward only (from t=T)
            T = t_delta if t_delta > 0 else 1.0
            
            batch_size, device = rigids_0.shape[0], rigids_0.device
            _num_timesteps = int(float(num_timesteps) * T)
            dt = 1.0 / _num_timesteps
            ts = np.linspace(min_t, T, _num_timesteps)[::-1] # reverse in time
            
            _feats = deepcopy({
                k: v.repeat(batch_size, *(1,)*(v.ndim-1))
                for k,v in batch.items() if k in ('aatype', 'residue_mask', 'fixed_mask', 'residue_idx', 'torsion_angles_sin_cos')
            }) 
            if t_delta > 0:
                rigids_t = self.diffuser.forward_marginal(
                    rigids_0=rigids_0,
                    t=t_delta * torch.ones(batch_size, device=device),
                    diffuse_mask=_feats['residue_mask'],
                    as_tensor_7=True,
                )['rigids_t']
            else:
                rigids_t = self.diffuser.sample_prior(
                    shape=rigids_0.shape,
                    device=device,
                    as_tensor_7=True,
                )['rigids_t']
            
            _feats['rigids_t'] = rigids_t
            
            traj_atom37 = []
            with torch.no_grad():
                fixed_mask = _feats['fixed_mask'] * _feats['residue_mask']
                diffuse_mask = (1 - _feats['fixed_mask']) * _feats['residue_mask']
                
                if self_conditioning:
                    _feats['sc_ca_t'] = torch.zeros_like(rigids_t[..., 4:])
                    _feats['t'] = ts[0] * torch.ones(batch_size, device=device)
                    _feats['sc_ca_t'] = self.net(_feats, as_tensor_7=True)['rigids'][..., 4:]  # update self-conditioning feats
                
                for t in ts:
                    _feats['t'] = t * torch.ones(batch_size, device=device)                    
                    out = self.net(_feats, as_tensor_7=False)
                    
                    # compute predicted rigids
                    if t == min_t:
                        rigids_pred = out['rigids']
                    else:
                        # update self-conditioning feats
                        if self_conditioning:
                            _feats['sc_ca_t'] = out['rigids'].to_tensor_7()[..., 4:]
                        # get score based on predicted rigids
                        pred_scores = self.diffuser.score(
                            rigids_0=out['rigids'],
                            rigids_t=Rigid.from_tensor_7(_feats['rigids_t']),
                            t=_feats['t'],
                            mask=_feats['residue_mask'],
                        )
                        rigids_pred = self.diffuser.reverse(
                            rigids_t=Rigid.from_tensor_7(_feats['rigids_t']),
                            rot_score=pred_scores['rot_score'],
                            trans_score=pred_scores['trans_score'],
                            t=_feats['t'],
                            dt=dt,
                            diffuse_mask=diffuse_mask,
                            center_trans=True,
                            noise_scale=noise_scale,
                            probability_flow=probability_flow,
                        )   # Rigid object
                        # update rigids_t as tensor_7
                        _feats['rigids_t'] = rigids_pred.to_tensor_7()   
                    
                # compute atom37 positions
                atom37 = compute_backbone(rigids_pred, out['psi'], aatype=_feats['aatype'])[0]
                atom37 = atom37.detach().cpu().numpy() # (Bi, L, 37 ,3)
                return atom37
        
        saved_paths = []
        
        # iterate over delta-Ts
        for t_delta in delta_range:
            gt_rigids_4x4 = batch['rigidgroups_gt_frames'][..., 0, :, :].clone()
            n_bs = n_replica // replica_per_batch   # number of batches
            last_bs = n_replica % replica_per_batch # last batch size
            atom_positions = []
            for _ in range(n_bs):
                rigids_0 = Rigid.from_tensor_4x4(gt_rigids_4x4.repeat(replica_per_batch, *(1,)*(gt_rigids_4x4.ndim-1)))
                traj_atom37 = forward_backward(rigids_0, t_delta)
                atom_positions.append(traj_atom37)
            if last_bs > 0:
                rigids_0 = Rigid.from_tensor_4x4(gt_rigids_4x4.repeat(last_bs, *(1,)*(gt_rigids_4x4.ndim-1)))
                traj_atom37 = forward_backward(rigids_0, t_delta)
                atom_positions.append(traj_atom37)
            atom_positions = np.concatenate(atom_positions, axis=0)   # (B, L, 37, 3)
            
            # Save atom positions to pdb.
            t_delta_dir = os.path.join(output_dir, f"{t_delta}")
            os.makedirs(t_delta_dir, exist_ok=True)
            save_to = os.path.join(t_delta_dir, f"{accession_code}.pdb")
            saved_to = atom37_to_pdb(
                atom_positions=atom_positions,
                save_to=save_to, 
                **extra,
            )
            saved_paths.append(saved_to)

        all_delta_dir = os.path.join(output_dir, "all_delta")
        os.makedirs(all_delta_dir, exist_ok=True)
        merge_pdbfiles(saved_paths, os.path.join(all_delta_dir, f"{accession_code}.pdb"))

        return all_delta_dir

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = DiffusionLitModule(None, None, None, None, None)
