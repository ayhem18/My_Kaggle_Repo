"""
This script contains the functionalities to
1. train the CBM model
2. monitor the experiments
3. save the different checkpoints
"""

import torch
import os

import pytorch_modular.CBM.model.utilities as ut

from typing import Dict, Any, Optional, Union
from tqdm import tqdm
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pytorch_modular.CBM.model.CBM import CBModel
from pytorch_modular.CBM.model.CBM_epoch import train_CBM_per_epoch, val_CBM_per_epoch
from pytorch_modular.exp_tracking import create_summary_writer
from pytorch_modular.directories_and_files import process_save_path
from pytorch_modular.image_classification.classification_metrics import accuracy
from pytorch_modular.pytorch_utilities import save_model


def _validate_training_configuration(train_configuration: Dict) -> Dict[str, Any]:
    # first step: extract the necessary parameters for the training: optimizer and scheduler
    optimizer = train_configuration.get(ut.OPTIMIZER, None)
    scheduler = train_configuration.get(ut.SCHEDULER, None)

    # set the default multi-class classification loss
    train_configuration[ut.LOSS_FUNCTION] = train_configuration.get(ut.LOSS_FUNCTION, nn.CrossEntropyLoss())

    train_configuration[ut.OUTPUT_LAYER] = train_configuration.get(ut.OUTPUT_LAYER, None)

    necessary_training_params = [(ut.OPTIMIZER, optimizer),
                                 (ut.SCHEDULER, scheduler)]

    # make sure these parameters are indeed passed to the train_model function
    for name, tp in enumerate(necessary_training_params):
        if tp is None:
            raise TypeError(f"The argument {name} is expected to be passed as non-None to the configuration\n"
                            f"Found: {type(tp)}")

    # set the default parameters
    train_configuration[ut.METRICS] = train_configuration.get(ut.METRICS, {'accuracy': accuracy})
    train_configuration[ut.MIN_TRAIN_LOSS] = train_configuration.get(ut.MIN_TRAIN_LOSS, None)
    train_configuration[ut.MIN_VAL_LOSS] = train_configuration.get(ut.MIN_VAL_LOSS, None)
    train_configuration[ut.MAX_EPOCHS] = train_configuration.get(ut.MAX_EPOCHS, 50)
    train_configuration[ut.MIN_EVALUATION_EPOCH] = train_configuration.get(ut.MIN_EVALUATION_EPOCH,
                                                                           train_configuration[ut.MAX_EPOCHS] // 10)

    train_configuration[ut.DEVICE] = train_configuration.get(ut.DEVICE,
                                                             ('cuda' if torch.cuda.is_available() else 'cpu'))

    train_configuration[ut.PROGRESS] = train_configuration.get(ut.PROGRESS, True)
    train_configuration[ut.REPORT_EPOCH] = train_configuration.get(ut.REPORT_EPOCH, None)
    # the default value will be set to 5% of the max number of epochs
    train_configuration[ut.NO_IMPROVE_STOP] = train_configuration.get(ut.NO_IMPROVE_STOP,
                                                                      train_configuration[ut.MAX_EPOCHS] * 0.15)

    train_configuration[ut.DEBUG] = train_configuration.get(ut.DEBUG, False)
    train_configuration[ut.COMPUTE_LOSS] = train_configuration.get(ut.COMPUTE_LOSS, None)

    # the last step in the validation is to make sure the metric objects
    # are on the same device (if they are device-dependent objects)

    for metric_name, metric_function in train_configuration[ut.METRICS].items():
        if hasattr(metric_function, 'to'):
            train_configuration[ut.METRICS][metric_name].to(train_configuration[ut.DEVICE])

    return train_configuration


def _set_summary_writer(writer: SummaryWriter,
                        epoch: int,
                        train_losses: Union[Dict[str: float], float] = None,
                        val_losses: Union[Dict[str: float], float] = None,
                        train_metrics: Union[Dict[str: float], float] = None,
                        val_metrics: Union[Dict[str: float], float] = None
                        ) -> None:
    if writer is not None:
        # make sure to convert all input arguments to the dictionary data type
        train_losses = {"loss": train_losses} if not (isinstance(train_losses, Dict)) else train_losses
        val_losses = {"loss": val_losses} if not (isinstance(val_losses, Dict)) else val_losses

        # before proceeding with adding the different values to the writer, make sure train_losses and val_losses
        # have the same field names
        tl_names, vl_names = set(train_losses.keys()), set(val_losses.keys())
        if tl_names != vl_names:
            raise ValueError(f"PLease make sure the training losses and the val losses have the same field names.")

        for n, _ in train_losses.items():
            writer.add_scalars(main_tag=n,
                               tag_scalar_dict={f'train_{n}': train_losses[n], f'val_{n}': val_losses[n]},
                               global_step=epoch)

        # the same test should be conducted for the metrics
        tl_names, vl_names = set(train_metrics.keys()), set(val_losses.keys())
        if tl_names != vl_names:
            raise ValueError(f"PLease make sure the training losses and the val losses have the same field names.")

        for name, _ in train_metrics.items():
            writer.add_scalars(main_tag=name,
                               tag_scalar_dict={f"train_{name}": train_metrics[name],
                                                f"val_{name}": val_metrics[name]},
                               global_step=epoch)

        # make sure to close the writer
        writer.close()


def train_CBM(model: CBModel,
              train_dataloader: DataLoader[torch.Tensor, torch.Tensor, torch.Tensor],
              val_dataloader: DataLoader[torch.Tensor, torch.Tensor, torch.Tensor],
              train_configuration: Dict[str, Any],
              log_dir: Optional[Union[Path, str]] = None,
              save_path: Optional[Union[Path, str]] = None,
              ) -> CBModel:
    # set the default parameters
    train_configuration = _validate_training_configuration(train_configuration)

    save_path = save_path if save_path is not None else log_dir

    # before proceeding with the training, let's set the summary writer
    writer, save_path = (None, None) if log_dir is None else (create_summary_writer(log_dir, return_path=True))
    checkpoints_path = (process_save_path(os.path.join(save_path, 'checkpoints'))) if save_path is not None else None

    min_train_loss, best_model = float('inf'), None

    for epoch in tqdm(range(train_configuration[ut.MAX_EPOCHS])):
        epoch_train_losses, epoch_train_metrics = train_CBM_per_epoch(model=model,
                                                                      train_dataloader=train_dataloader,
                                                                      loss_function=train_configuration[
                                                                          ut.LOSS_FUNCTION],
                                                                      optimizer=train_configuration[ut.OPTIMIZER],
                                                                      output_layer=train_configuration[ut.OUTPUT_LAYER],
                                                                      scheduler=train_configuration[ut.SCHEDULER],
                                                                      device=train_configuration[ut.DEVICE])

        epoch_val_losses, epoch_val_metrics = val_CBM_per_epoch(model=model,
                                                                val_dataloader=val_dataloader,
                                                                loss_function=train_configuration[ut.LOSS_FUNCTION],
                                                                output_layer=train_configuration[ut.OUTPUT_LAYER],
                                                                device=train_configuration[ut.DEVICE])

        _set_summary_writer(writer=writer,
                            epoch=epoch,
                            train_losses=epoch_train_losses,
                            val_losses=epoch_val_losses,
                            train_metrics=epoch_train_metrics)

        # save the best model
        if epoch_train_losses['loss'] < min_train_loss:
            best_model = model.copy()
        # update the min train loss
        min_train_loss = min(min_train_loss, epoch_train_losses['loss'])

        # save checkpoints
        if epoch % 10 == 9 and checkpoints_path is not None:
            n_checkpoint = len(os.listdir(checkpoints_path))
            save_model(model=model, path=os.path.join(checkpoints_path, f'checkpoint_{n_checkpoint}.pt'))

    if save_path is not None:
        save_model(best_model, path=save_path)

    return best_model
