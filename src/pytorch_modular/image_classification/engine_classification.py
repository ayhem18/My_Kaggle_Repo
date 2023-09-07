"""
This script contains general functionalities to train image classification models as well
as offers a number of frequent use cases such as:
    * saving models
    * creating summary writers to visualize the model's performance
    *
"""

import torch
import itertools
from tqdm import tqdm

import numpy as np
import src.pytorch_modular.image_classification.utilities as ut

from typing import Union, Dict, Optional, List
from pathlib import Path

from torch import nn
from torch.utils.data import DataLoader

from src.pytorch_modular.pytorch_utilities import get_default_device, save_model
from src.pytorch_modular.exp_tracking import save_info
from src.pytorch_modular.image_classification.classification_metrics import accuracy
from src.pytorch_modular.image_classification.epoch_engine import train_per_epoch, val_per_epoch
from src.pytorch_modular.exp_tracking import create_summary_writer


def binary_output(x: torch.Tensor) -> torch.IntTensor:
    sigma = nn.Sigmoid()
    output = sigma(x).to(torch.int32)
    return output


# let's define a function to validate the passed training configuration
def _validate_training_configuration(train_configuration: Dict) -> Dict[str]:
    # first step: extract the necessary parameters for the training: optimizer and scheduler
    optimizer = train_configuration.get(ut.OPTIMIZER, None)
    scheduler = train_configuration.get(ut.SCHEDULER, None)

    # set the default multi-class classification loss
    loss_function = train_configuration.get(ut.LOSS_FUNCTION, nn.CrossEntropyLoss)
    # the default output layer: argmax
    output_layer = train_configuration.get(ut.OUTPUT_LAYER, lambda x: x.argmax(dim=-1))

    necessary_training_params = [(ut.OPTIMIZER, optimizer),
                                 (ut.SCHEDULER, scheduler)]

    # make sure these parameters are indeed passed to the train_model function
    for name, tp in enumerate(necessary_training_params):
        if tp is None:
            raise TypeError(f"The argument {name} is expected to be passed as non-None to the configuration\n"
                            f"Found: {type(tp)}")

    # set the default parameters

    train_configuration[ut.METRICS] = train_configuration.get(ut.METRICS, [accuracy])
    train_configuration[ut.MIN_TRAIN_LOSS] = train_configuration.get(ut.MIN_TRAIN_LOSS, None)
    train_configuration[ut.MIN_VAL_LOSS] = train_configuration.get(ut.MIN_VAL_LOSS, None)
    train_configuration[ut.MAX_EPOCHS] = train_configuration.get(ut.MAX_EPOCHS, 50)
    train_configuration[ut.MIN_EVALUATION_EPOCH] = train_configuration.get(ut.MIN_EVALUATION_EPOCH,
                                                                           train_configuration[ut.MAX_EPOCHS] // 10)

    train_configuration[ut.DEVICE] = train_configuration.get(ut.DEVICE, get_default_device())
    train_configuration[ut.PROGRESS] = train_configuration.get(ut.PROGRESS, True)
    train_configuration[ut.REPORT_BATCH] = train_configuration.get(ut.REPORT_BATCH, None)

    return train_configuration


def _report_performance(train_configuration: Dict,
                        train_loss: float,
                        val_loss: float,
                        train_metrics: Dict[str, float],
                        val_metrics: Dict[str, float]) -> None:
    if train_configuration[ut.REPORT_BATCH]:
        print("#" * 25)
        print(f"training loss: {train_loss}")

        for metric_name, metric_value in train_metrics.items():
            print(f"train_{metric_name}: {metric_value}")

        print(f"validation loss : {val_loss}")
        for metric_name, metric_value in val_metrics.items():
            print(f"train_{metric_name}: {metric_value}")
        print("#" * 25)


def _track_performance(performance_dict: Dict[str, List[float]],
                       train_loss: float,
                       val_loss: float,
                       train_metric: Dict[str, float],
                       val_metrics: Dict[str, float]) -> None:
    # add the losses first
    performance_dict[ut.TRAIN_LOSS] += train_loss
    performance_dict[ut.VAL_LOSS] += val_loss

    # update train metrics
    for metric_name, metric_value in train_metric.items():
        performance_dict[metric_name].append(metric_value)

    # update val metrics
    for metric_name, metric_value in val_metrics.items():
        performance_dict[metric_name].append(metric_value)


def _set_summary_writer(log_dir: Union[Path, str],
                        epoch_train_loss,
                        epoch_val_loss,
                        epoch_train_metrics,
                        epoch_val_metrics,
                        epoch) -> None:
    if log_dir is not None:
        # initialize a SummaryWriter object
        writer = create_summary_writer(parent_dir=log_dir)
        # track loss results
        writer.add_scalars(main_tag='Loss',
                           tag_scalar_dict={"train_loss": epoch_train_loss, 'val_loss': epoch_val_loss},
                           global_step=epoch)

        for name, m in epoch_train_metrics.items():
            writer.add_scalars(main_tag=name,
                               tag_scalar_dict={f"train_{name}": m, f"val_{name}": epoch_val_metrics[name]},
                               global_step=epoch)

        writer.close()


def train_model(model: nn.Module,
                train_dataloader: DataLoader[torch.Tensor],
                test_dataloader: DataLoader[torch.Tensor],
                train_configuration: Dict,
                log_dir: Optional[Union[Path, str]] = None,
                save_path: Optional[Union[Path, str]] = None,
                ):
    # set the default parameters
    train_configuration = _validate_training_configuration(train_configuration)

    save_path = save_path if save_path is not None else log_dir

    performance_dict = {ut.TRAIN_LOSS: [],
                        ut.VAL_LOSS: []}

    metrics = train_configuration[ut.METRICS]

    # save 2 copies: val and train for each metric
    for name, _ in metrics.items():
        performance_dict[f'train_{name}'] = []
        performance_dict[f'val_{name}'] = []

    best_model, best_loss = None, None

    for epoch in tqdm(range(train_configuration[ut.MAX_EPOCHS])):

        epoch_train_metrics = train_per_epoch(model=model,
                                              train_dataloader=train_dataloader,
                                              loss_function=train_configuration[ut.LOSS_FUNCTION],
                                              optimizer=train_configuration[ut.OPTIMIZER],
                                              output_layer=train_configuration[ut.OUTPUT_LAYER],
                                              scheduler=train_configuration[ut.SCHEDULER],
                                              device=train_configuration[ut.DEVICE],
                                              report_batch=train_configuration[ut.REPORT_BATCH])

        epoch_val_metrics = val_per_epoch(model=model,
                                          dataloader=test_dataloader,
                                          loss_fn=train_configuration[ut.TRAIN_LOSS],
                                          output_layer=train_configuration[ut.OUTPUT_LAYER],
                                          device=train_configuration[ut.DEVICE])

        epoch_train_loss = epoch[ut.TRAIN_LOSS]
        del (epoch_train_metrics[ut.TRAIN_LOSS])

        epoch_val_loss = epoch_val_metrics[ut.VAL_LOSS]
        del (epoch_val_metrics[ut.VAL_LOSS])

        # track the best performing model on the validation portion
        # only consider the losses after a minimal number of epochs
        if (epoch >= train_configuration[ut.MIN_EVALUATION_EPOCH] and
                (best_loss is None or best_loss >= epoch_val_loss)):
            best_loss = epoch_val_loss
            best_model = model

        _report_performance(train_configuration,
                            epoch_train_loss,
                            epoch_val_loss,
                            epoch_train_metrics,
                            epoch_val_metrics)

        # save the model's performance for this epoch
        _track_performance(performance_dict=performance_dict,
                           train_loss=epoch_train_loss,
                           val_loss=epoch_val_loss,
                           train_metric=epoch_train_metrics,
                           val_metrics=epoch_val_metrics)

        _set_summary_writer(log_dir,
                            epoch_train_loss=epoch_train_loss,
                            epoch_val_loss=epoch_val_loss,
                            epoch_train_metrics=epoch_train_metrics,
                            epoch_val_metrics=epoch_val_metrics,
                            epoch=epoch
                            )

        # check if the losses reached the minimum thresholds
        if ((train_configuration[ut.MIN_TRAIN_LOSS] is not None and
             train_configuration[ut.MIN_TRAIN_LOSS] >= epoch_train_loss) or

                (train_configuration[ut.MIN_VAL_LOSS] is not None
                 and train_configuration[ut.MIN_VAL_LOSS] >= epoch_val_loss)):
            # the model that goes lower than these thresholds is automatically the best model
            break

    # in addition to the model save all the details:
    # build the details:
    details = {ut.OPTIMIZER: train_configuration[ut.OPTIMIZER],
               ut.SCHEDULER: train_configuration[ut.SCHEDULER],
               ut.MAX_EPOCHS: train_configuration[ut.MAX_EPOCHS],
               ut.MIN_TRAIN_LOSS: train_configuration[ut.MIN_TRAIN_LOSS],
               ut.MIN_VAL_LOSS: train_configuration[ut.MIN_VAL_LOSS]}

    save_info(save_path=log_dir, details=details)
    save_model(best_model, path=save_path)
    return performance_dict


VALID_RETURN_TYPES = ['np', 'pt', 'list']


def inference(model: nn.Module,
              inference_dataloader: DataLoader[torch.tensor],
              output_layer: Union[nn.Module, callable],
              device: str = None,
              return_tensor: str = 'np'
              ) -> Union[np.ndarray, torch.tensor, list[int]]:
    device = get_default_device() if device is None else device
    # make sure the return_tensor argument is a set to a valid value
    if return_tensor not in VALID_RETURN_TYPES:
        raise ValueError(f'the `return_tensor` argument is expected to be among {VALID_RETURN_TYPES}\n'
                         f'found: {return_tensor}')

    # set to the inference mode
    model.eval()
    model.to(device)

    with torch.inference_mode():
        result = [output_layer(model(X.to(device))) for X in inference_dataloader]

    # now we have a list of pytorch tensors
    if return_tensor == 'pt':
        res = torch.stack(result)
        res = torch.squeeze(res, dim=-1)

    else:
        # convert res to a list of lists
        res = [torch.squeeze(r, dim=-1).cpu().tolist() for r in result]
        # flatten the list
        res = list(itertools.chain(*res))
        res = np.asarray(res) if return_tensor == 'np' else res
    return res
