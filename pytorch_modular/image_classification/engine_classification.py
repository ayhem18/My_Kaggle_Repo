"""
This script contains general functionalities to train image classification models as well
as offers a number of frequent use cases such as:
    * saving models
    * creating summary writers to visualize the model's performance
    *
"""
import os
import warnings

import torch
import itertools
from tqdm import tqdm

import numpy as np
import pytorch_modular.image_classification.utilities as ut

from typing import Union, Dict, Optional, List, Any
from pathlib import Path
from PIL import Image

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as tr

from pytorch_modular.pytorch_utilities import get_default_device, save_model
from pytorch_modular.exp_tracking import save_info
from pytorch_modular.image_classification.classification_metrics import accuracy, ACCURACY
from pytorch_modular.image_classification.epoch_engine import train_per_epoch, val_per_epoch
from pytorch_modular.exp_tracking import create_summary_writer
from pytorch_modular.directories_and_files import process_save_path


def binary_output(x: torch.Tensor) -> torch.IntTensor:
    sigma = nn.Sigmoid()
    output = sigma(x).to(torch.int32)
    return output


##################################################################################################################
# UTILITY TRAINING FUNCTIONS:

# let's define a function to validate the passed training configuration
def _validate_training_configuration(train_configuration: Dict) -> Dict[str, Any]:
    # first step: extract the necessary parameters for the training: optimizer and scheduler
    optimizer = train_configuration.get(ut.OPTIMIZER, None)
    scheduler = train_configuration.get(ut.SCHEDULER, None)

    # set the default multi-class classification loss
    train_configuration[ut.LOSS_FUNCTION] = train_configuration.get(ut.LOSS_FUNCTION, nn.CrossEntropyLoss())

    # the default output layer: argmax: since only the default loss expects logits: the predictions need hard labels
    def default_output(x: torch.Tensor) -> torch.Tensor:
        return x.argmax(dim=-1)

    train_configuration[ut.OUTPUT_LAYER] = train_configuration.get(ut.OUTPUT_LAYER, default_output)

    necessary_training_params = [(ut.OPTIMIZER, optimizer),
                                 (ut.SCHEDULER, scheduler)]

    # make sure these parameters are indeed passed to the train_model function
    for name, tp in enumerate(necessary_training_params):
        if tp is None:
            raise TypeError(f"The argument {name} is expected to be passed as non-None to the configuration\n"
                            f"Found: {type(tp)}")

    # set the default parameters
    train_configuration[ut.METRICS] = train_configuration.get(ut.METRICS, {ACCURACY: accuracy})
    train_configuration[ut.MIN_TRAIN_LOSS] = train_configuration.get(ut.MIN_TRAIN_LOSS, None)
    train_configuration[ut.MIN_VAL_LOSS] = train_configuration.get(ut.MIN_VAL_LOSS, None)
    train_configuration[ut.MAX_EPOCHS] = train_configuration.get(ut.MAX_EPOCHS, 50)
    train_configuration[ut.MIN_EVALUATION_EPOCH] = train_configuration.get(ut.MIN_EVALUATION_EPOCH,
                                                                           train_configuration[ut.MAX_EPOCHS] // 10)

    train_configuration[ut.DEVICE] = train_configuration.get(ut.DEVICE, get_default_device())
    train_configuration[ut.PROGRESS] = train_configuration.get(ut.PROGRESS, True)
    train_configuration[ut.REPORT_EPOCH] = train_configuration.get(ut.REPORT_EPOCH, None)
    # the default value will be set to 5% of the max number of epochs
    train_configuration[ut.NO_IMPROVE_STOP] = train_configuration.get(ut.NO_IMPROVE_STOP,
                                                                      train_configuration[ut.MAX_EPOCHS] * 0.15)

    train_configuration[ut.DEBUG] = train_configuration.get(ut.DEBUG, False)
    train_configuration[ut.COMPUTE_LOSS] = train_configuration.get(ut.COMPUTE_LOSS, None)

    # the last step in the validation is to make sure the metric objects (if they are pytorchMetrics objects) are on the same device
    for metric_name, metric_function in train_configuration[ut.METRICS].items():
        if hasattr(metric_function, 'to'):
            train_configuration[ut.METRICS][metric_name].to(train_configuration[ut.DEVICE])

    return train_configuration


def _report_performance(train_loss: float,
                        val_loss: float,
                        train_metrics: Dict[str, float],
                        val_metrics: Dict[str, float]) -> None:
    print("#" * 25)
    print(f"training loss: {train_loss}")

    for metric_name, metric_value in train_metrics.items():
        print(f"train_{metric_name}: {metric_value}")

    print(f"validation loss : {val_loss}")
    for metric_name, metric_value in val_metrics.items():
        print(f"val_{metric_name}: {metric_value}")
    print("#" * 25)


def _track_performance(performance_dict: Dict[str, List[float]],
                       train_loss: float,
                       val_loss: float,
                       train_metric: Dict[str, float],
                       val_metrics: Dict[str, float]) -> None:
    # add the losses first
    performance_dict[ut.TRAIN_LOSS].append(train_loss)
    performance_dict[ut.VAL_LOSS].append(val_loss)

    # update the best training and validation losses
    performance_dict[f'best_{ut.TRAIN_LOSS}'] = min([train_loss,
                                                     performance_dict.get(f'best_{ut.TRAIN_LOSS}', float('inf'))])

    performance_dict[f'best_{ut.VAL_LOSS}'] = min([val_loss,
                                                   performance_dict.get(f'best_{ut.VAL_LOSS}', float('inf'))])

    # update train metrics
    for metric_name, metric_value in train_metric.items():
        performance_dict[f'train_{metric_name}'].append(metric_value)
        # update the best value of the given metric on the train split
        performance_dict[f'best_train_{metric_name}'] = min([metric_value,
                                                             performance_dict.get(f'best_train_{metric_name}',
                                                                                  float('inf'))])

    # update val metrics
    for metric_name, metric_value in val_metrics.items():
        performance_dict[f'val_{metric_name}'].append(metric_value)

        performance_dict[f'best_val_{metric_name}'] = min([metric_value,
                                                           performance_dict.get(f'best_val_{metric_name}',
                                                                                float('inf'))])


def _set_summary_writer(writer: SummaryWriter,
                        epoch_train_loss,
                        epoch_val_loss,
                        epoch_train_metrics,
                        epoch_val_metrics,
                        epoch) -> None:
    if writer is not None:
        # track loss results
        writer.add_scalars(main_tag='Loss',
                        tag_scalar_dict={ut.TRAIN_LOSS: epoch_train_loss, ut.VAL_LOSS: epoch_val_loss},
                        global_step=epoch)

        for name, m in epoch_train_metrics.items():
            writer.add_scalars(main_tag=name,
                            tag_scalar_dict={f"train_{name}": m, f"val_{name}": epoch_val_metrics[name]},
                            global_step=epoch)

        writer.close()


# THE MAIN TRAINING FUNCTION:
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

    # best_model, best_loss = None, None
    min_training_loss, no_improve_counter, best_model = float('inf'), 0, None

    # in addition to the model save all the details:
    # build the details:
    details = {ut.OPTIMIZER: train_configuration[ut.OPTIMIZER],
               ut.SCHEDULER: train_configuration[ut.SCHEDULER],
               ut.MAX_EPOCHS: train_configuration[ut.MAX_EPOCHS],
               ut.MIN_TRAIN_LOSS: train_configuration[ut.MIN_TRAIN_LOSS],
               ut.MIN_VAL_LOSS: train_configuration[ut.MIN_VAL_LOSS]}

    # before proceeding with the training, let's set the summary writer
    writer, save_path  = (None, None) if log_dir is None else (create_summary_writer(log_dir, return_path=True))

    checkpoints_path = (process_save_path(os.path.join(save_path, 'checkpoints'))) if save_path is not None else (None)

    debug = train_configuration[ut.DEBUG]

    for epoch in tqdm(range(train_configuration[ut.MAX_EPOCHS])):

        epoch_train_metrics = train_per_epoch(model=model,
                                              train_dataloader=train_dataloader,
                                              compute_loss=train_configuration[ut.COMPUTE_LOSS],
                                              compute_loss_kwargs=train_configuration[ut.COMPUTE_LOSS_KWARGS],
                                              loss_function=train_configuration[ut.LOSS_FUNCTION],
                                              optimizer=train_configuration[ut.OPTIMIZER],
                                              output_layer=train_configuration[ut.OUTPUT_LAYER],
                                              scheduler=train_configuration[ut.SCHEDULER],
                                              device=train_configuration[ut.DEVICE],
                                              debug=train_configuration[ut.DEBUG], 
                                              metrics=train_configuration[ut.METRICS])
        
        if debug:
            print(f"Train epoch: {epoch} done")            

        epoch_val_metrics = val_per_epoch(model=model, 
                                          dataloader=test_dataloader,
                                            loss_function=train_configuration[ut.LOSS_FUNCTION],
                                            output_layer=train_configuration[ut.OUTPUT_LAYER],
                                            compute_loss=train_configuration[ut.COMPUTE_LOSS],
                                            computer_loss_kwargs=train_configuration[ut.COMPUTE_LOSS_KWARGS],
                                            device=train_configuration[ut.DEVICE],
                                            debug=train_configuration[ut.DEBUG], 
                                            metrics=train_configuration[ut.METRICS])
        if debug: 
            print(f"val epoch: {epoch} done!!")

        epoch_train_loss = epoch_train_metrics[ut.TRAIN_LOSS]
        del (epoch_train_metrics[ut.TRAIN_LOSS])

        epoch_val_loss = epoch_val_metrics[ut.VAL_LOSS]
        del (epoch_val_metrics[ut.VAL_LOSS])

        no_improve_counter = no_improve_counter + 1 if min_training_loss < epoch_train_loss else 0

        if min_training_loss > epoch_train_loss:
            # save the model with the lowest training error
            min_training_loss = epoch_train_loss
            best_model = model

        if (train_configuration[ut.REPORT_EPOCH] is not None
                and epoch % train_configuration[ut.REPORT_EPOCH] == 0):
            _report_performance(epoch_train_loss,
                                epoch_val_loss,
                                epoch_train_metrics,
                                epoch_val_metrics)

        # save the model's performance for this epoch
        _track_performance(performance_dict=performance_dict,
                           train_loss=epoch_train_loss,
                           val_loss=epoch_val_loss,
                           train_metric=epoch_train_metrics,
                           val_metrics=epoch_val_metrics)

        _set_summary_writer(writer,
                            epoch_train_loss=epoch_train_loss,
                            epoch_val_loss=epoch_val_loss,
                            epoch_train_metrics=epoch_train_metrics,
                            epoch_val_metrics=epoch_val_metrics,
                            epoch=epoch
                            )

        if epoch % 10 == 9 and checkpoints_path is not None:
            n_checkpoint = len(os.listdir(checkpoints_path))
            save_model(model=model, path=os.path.join(checkpoints_path, f'checkpoint_{n_checkpoint}.pt'))

        # check if the losses reached the minimum thresholds
        if ((train_configuration[ut.MIN_TRAIN_LOSS] is not None and
             train_configuration[ut.MIN_TRAIN_LOSS] >= epoch_train_loss) or

                (train_configuration[ut.MIN_VAL_LOSS] is not None
                 and train_configuration[ut.MIN_VAL_LOSS] >= epoch_val_loss)):
            warnings.warn((f"The validation loss {train_configuration[ut.MIN_VAL_LOSS]} was reached\n"
                           f"aborting training!!"), category=RuntimeWarning)
            # the first state that reaches lower scores than the specified thresholds
            # is consequently the model's best state
            break

        # abort training if 2 conditions were met:
        # 1. NO_IMPROVE_STOP is larger than the minimum value
        # 2. the training loss did not decrease for consecutive NO_IMPROVE_STOP epochs

        if ut.MIN_NO_IMPROVE_STOP <= train_configuration[ut.NO_IMPROVE_STOP] <= no_improve_counter:
            warnings.warn(f"The training loss did not improve for {no_improve_counter} consecutive epochs."
                          f"\naborting training!!", category=RuntimeWarning)
            break

    if log_dir is not None:
        save_info(save_path=log_dir, details=details)

    if save_path is not None:
        save_model(best_model, path=save_path)

    return best_model, performance_dict


##################################################################################################################
# time to set the inference part of the script
_VALID_RETURN_TYPES = ['np', 'pt', 'list']


# relatively small test splits (that can fit th memory)

class InferenceDirDataset(Dataset):
    def __init__(self,
                 test_dir: Union[str, Path],
                 transformations: tr) -> None:
        # the usual base class constructor call
        super().__init__()
        test_data_path = process_save_path(test_dir, file_ok=False, dir_ok=True)
        self.data = [os.path.join(test_data_path, file_name) for file_name in os.listdir(test_data_path)]
        self.t = transformations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> int:
        # don't forget to apply the transformation before returning the index-th element in the directory
        return self.t(Image.open(self.data[index]))


def _set_inference_loader(inference_source_data: Union[DataLoader[torch.tensor], Path, str],
                          transformations: tr = None, 
                          batch_size: int = 100) -> DataLoader:
    # the input to this function should be validated
    if isinstance(inference_source_data, (Path, str)):

        warnings.warn(f"The inference source data was passed as a path to a directory..."
                      f"\nBuilding the dataloader", category=RuntimeWarning)

        # make sure the transformations argument is passed
        if transformations is None:
            raise TypeError("The 'transformations' argument must be passed if the data source is a directory"
                            f"\nFound: {transformations}")
        
        ds = InferenceDirDataset(inference_source_data, transformations)
        dataloader = DataLoader(ds,
                                batch_size=batch_size,
                                shuffle=False,  # shuffle false to keep the original order of the test-split samples
                                num_workers=os.cpu_count() // 2)
        return dataloader

    return inference_source_data


def inference(model: nn.Module,
              inference_source_data: Union[DataLoader, Path, str],
              transformation: tr = None,
              output_layer: Union[nn.Module, callable] = None,
              device: str = None,
              return_tensor: str = 'np'
              ) -> Union[np.ndarray, torch.tensor, List[int]]:
    # first let's make sure our loader is set
    loader = _set_inference_loader(inference_source_data,
                                   transformation)

    device = get_default_device() if device is None else device
    # make sure the return_tensor argument is a set to a valid value
    if return_tensor not in _VALID_RETURN_TYPES:
        raise ValueError(f'the `return_tensor` argument is expected to be among {_VALID_RETURN_TYPES}\n'
                         f'found: {return_tensor}')

    def default_output(x: torch.Tensor):
        return x.argmax(dim=-1)

    # the default output layer is the softmax layer: (reduced to argmax)
    output_layer = default_output if output_layer is None else output_layer
    # set to the inference mode
    model.eval()
    model.to(device)

    with torch.inference_mode():
        result = [output_layer(model.forward(X.to(device))) for X in loader]

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
