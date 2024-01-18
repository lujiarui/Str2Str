from typing import Any, Dict, List, Tuple
import os
from time import strftime

import numpy as np
import pandas as pd
import torch
import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
    checkpoint_utils,
    plot_utils,
)
from src.common.pdb_utils import extract_backbone_coords
from src.metrics import metrics 

log = RankedLogger(__name__, rank_zero_only=True)


def evaluate_prediction(pred_dir: str, target_dir: str = None, tag: str = None):
    """Evaluate prediction results based on pdb files.
    """
    if target_dir is None or not os.path.isdir(target_dir):
        log.warning(f"target_dir {target_dir} does not exist. Skip evaluation.")
        return {}
        
    assert os.path.isdir(pred_dir), f"pred_dir {pred_dir} is not a directory."
    
    targets = [
        d.replace(".pdb", "") for d in os.listdir(target_dir)
    ]
    # pred_bases = os.listdir(pred_dir)
    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(pred_dir)))
    tag = tag if tag is not None else "dev"
    timestamp = strftime("%m%d-%H-%M")
    
    fns = {
        'val_clash': metrics.validity, 
        'val_bond': metrics.bonding_validity,
        'js_pwd': metrics.js_pwd, 
        'js_rg': metrics.js_rg, 
        'js_tica': metrics.js_tica,
    }
    eval_res = {k: {} for k in fns}
    
    for target in targets:
        pred_file = os.path.join(pred_dir, f"{target}.pdb")
        # assert os.path.isfile(pred_file), f"pred_file {pred_file} does not exist."
        if not os.path.isfile(pred_file):
            continue
        
        target_file = os.path.join(target_dir, f"{target}.pdb")
        ca_coords = {
            'target': extract_backbone_coords(target_file),
            'pred': extract_backbone_coords(pred_file),
        }
        for f_name, func in fns.items():
            res = func(ca_coords, ref_key='target') if f_name.startswith('js_') else func(ca_coords)
            if f_name == 'js_tica':
                eval_res[f_name][target] = res[0]['pred']
                save_to = os.path.join(output_dir, f"tica_{target}_{tag}_{timestamp}.png")
                plot_utils.scatterplot_2d(res[1], save_to=save_to, ref_key='target')
            else:
                eval_res[f_name][target] = res['pred']
    
    csv_save_to = os.path.join(output_dir, f"metrics_{tag}_{timestamp}.csv")
    df = pd.DataFrame.from_dict(eval_res) # row = target, col = metric name
    df.loc['mean'] = np.around(df.mean(), decimals=4)
    mean_metrics = df.loc['mean']
    df.to_csv(csv_save_to, index=True, sep='\t')
    
    return mean_metrics
        

@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Sample on a test set and report evaluation metrics.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    # assert cfg.ckpt_path
    pred_dir = cfg.get("pred_dir")
    if pred_dir and os.path.isdir(pred_dir):
        log.info(f"Found pre-computed prediction directory {pred_dir}.")
        metric_dict = evaluate_prediction(pred_dir, target_dir=cfg.target_dir)
        return metric_dict, None

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # Load checkpoint manually.
    model, ckpt_path = checkpoint_utils.load_model_checkpoint(model, cfg.ckpt_path)

    # log.info("Starting testing!")
    # trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    
    # Get dataloader for prediction.
    datamodule.setup(stage="predict")
    dataloaders = datamodule.test_dataloader()
    
    log.info("Starting predictions.")
    pred_dir = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=ckpt_path)[-1]

    # metric_dict = trainer.callback_metrics
    log.info("Starting evaluations.")
    metric_dict = evaluate_prediction(pred_dir, target_dir=cfg.target_dir)
    
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
