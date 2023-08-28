"""
This script contains general functionalities to train image classification models as well
as offers a number of frequent uses cases such as:
    * saving models
    * creating summary writers to visualize the model's performance
    *
"""

import torch
import itertools
from tqdm import tqdm

import numpy as np

from _collections_abc import Sequence
from typing import Union, Dict, Optional
from pathlib import Path

from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from src.pytorch_modular.pytorch_utilities import get_default_device, save_model
from src.pytorch_modular.exp_tracking import save_info


def accuracy(y_pred: torch.tensor, y: torch.tensor) -> float:
    # squeeze values if needed
    value = (y_pred == y).type(torch.float32).mean().item()
    return value


def binary_output(x: torch.Tensor) -> torch.IntTensor:
    sigma = nn.Sigmoid()
    output = sigma(x).to(torch.int32)
    return output


ACCURACY = 'accuracy'


def train_per_epoch(model: nn.Module,
                    train_dataloader: DataLoader[torch.tensor],
                    loss_function: nn.Module,
                    optimizer: torch.optim.optimizer,
                    output_layer: nn.Module,
                    scheduler: lr_scheduler = None,
                    metrics: Dict = None,
                    device: str = None,
                    report_batch: int = None) -> Dict[str, float]:
    # set default arguments
    device = get_default_device() if device is None else device
    metrics = {ACCURACY: accuracy} if metrics is None else metrics
    metrics = dict(metrics) if isinstance(metrics, tuple) and len(metrics) == 2 else metrics

    # set the model to the training model
    model.train()
    model.to(device)
    # set the training loss
    train_loss, train_metrics = 0, dict([(name, 0) for name, _ in metrics.items()])

    # make sure to set the `drop_last` field in the dataloader to True,
    # as it might affect the metrics

    if hasattr(train_dataloader, 'drop_last'):
        train_dataloader.drop_last = True

    for batch_index, (x, y) in enumerate(train_dataloader):
        # report the number of batches reported if `report_batch` is set to a value
        if report_batch is not None and batch_index % report_batch == 0:
            print(f'batch n: {batch_index + 1} is loaded !!')
        # set the data to the suitable device
        # THE LABELS MUST BE SET TO THE LONG DATATYPE
        x, y = x.to(device), y.to(torch.long).to(device)
        # make sure to un-squeeze 'y' if it is one-dimensional
        y = torch.unsqueeze(y, dim=-1) if len(y.shape) == 1 else y

        # set the optimizer
        optimizer.zero_grad()
        # forward pass
        y_pred = model(x)
        # calculate the loss, and backprop
        batch_loss = loss_function(y_pred, y.float())
        batch_loss.backward()
        # optimizer's step
        optimizer.step()

        train_loss += batch_loss.item()
        y_pred_class = output_layer(y_pred)

        batch_metrics = dict([(name, m(y_pred_class)) for name, m in metrics.items()])
        # save the results of the batch
        for name, metric_value in batch_metrics.items():
            train_metrics[name] += metric_value

    batches = len(train_dataloader)
    # adjust metrics to get the average loss and average metrics
    train_loss = train_loss / batches

    # divide all the metrics by the number of batches
    for name, metric_value in train_metrics.items():
        train_metrics[name] = metric_value / batches

    # update the learning rate at the end of each epoch
    if scheduler is not None:
        scheduler.step()

    # save the training loss in the dictionary
    train_metrics['train_loss'] = train_loss
    return train_metrics


def val_per_epoch(model: nn.Module,
                  dataloader: DataLoader[torch.tensor],
                  loss_fn: nn.Module,
                  output_layer: nn.Module,
                  metrics: Union[Sequence[callable], callable] = None,
                  device: str = None) -> Dict[str, float]:
    # set the default arguments
    # let's set some default arguments
    device = get_default_device() if device is None else device
    metrics = {ACCURACY: accuracy} if metrics is None else metrics
    metrics = dict(metrics) if isinstance(metrics, tuple) and len(metrics) == 2 else metrics

    val_loss, val_metrics = 0, dict([(name, 0) for name, _ in metrics.items()])

    # put the model to the evaluation mode
    model.eval()
    model.to(device)

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for _, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            # make sure to add an extra dimension to 'y' if it is uni dimensional
            y = torch.unsqueeze(y, dim=-1) if len(y.shape) == 1 else y
            y_pred = model(x)

            # 2. Calculate and accumulate loss
            loss = loss_fn(y_pred, y.float())
            val_loss += loss.item()

            labels = output_layer(y_pred)
            # calculate the different metrics needed:
            metrics_results = [m(labels, y) for m in metrics]

            # add the batch metrics to the train metrics in general
            for index, mr in enumerate(metrics_results):
                val_metrics[index] += mr

    # adjust metrics to get the average loss and average metrics
    loss = val_loss / len(dataloader)
    for name, metric_value in val_metrics.items():
        val_metrics[name] = metric_value / len(dataloader)

    val_metrics['val_loss'] = val_loss
    return val_metrics


# one possible mean to set the different training parameters is to pass a configuration object: dictionary
# create a constant literals to denote the different training parameters
OPTIMIZER = 'optimizer'
SCHEDULER = 'scheduler'
OUTPUT_LAYER = 'output_layer'
LOSS_FUNCTION = 'loss_function'
METRICS = 'metrics'
MIN_TRAIN_LOSS = 'min_train_loss'
MIN_VAL_LOSS = 'min_val_loss'
MAX_EPOCHS = 'max_epochs'
DEVICE = 'device'
PROGRESS = 'progress'
REPORT_BATCH = 'report_batch'
MIN_EVALUATION_EPOCH = 'min_evaluation_epoch'


def train_model(model: nn.Module,
                train_dataloader: DataLoader[torch.Tensor],
                test_dataloader: DataLoader[torch.Tensor],
                train_configuration: Dict,
                writer: Optional[SummaryWriter] = None,
                save_path: Optional[Union[Path, str]] = None
                ):
    # extract necessary parameters
    optimizer = train_configuration.get(OPTIMIZER, None)
    scheduler = train_configuration.get(SCHEDULER, None)
    output_layer = train_configuration.get(OUTPUT_LAYER, None)
    loss_function = train_configuration.get(LOSS_FUNCTION, None)

    training_params = [(OPTIMIZER, optimizer),
                       (SCHEDULER, scheduler),
                       (OUTPUT_LAYER, output_layer),
                       (LOSS_FUNCTION, loss_function)]

    for name, tp in enumerate(training_params):
        if tp is None:
            raise TypeError(f"The argument {name} is expected to be passed as non-None to the configuration\n"
                            f"Found: {type(tp)}")

    # the default is only the accuracy
    metrics = train_configuration.get(METRICS, [accuracy])
    min_train_loss = train_configuration.get(MIN_TRAIN_LOSS, None)
    min_val_loss = train_configuration.get(MIN_VAL_LOSS, None)
    max_epochs = train_configuration.get(MAX_EPOCHS, 50)
    min_eval_epoch = train_configuration.get(MIN_EVALUATION_EPOCH, max_epochs // 10)

    device = train_configuration.get(SCHEDULER, get_default_device())
    progress = train_configuration.get(SCHEDULER, True)
    report_batch = train_configuration.get(REPORT_BATCH, None)

    results = {"train_loss": [],
               "val_loss": []}

    for index, _ in enumerate(metrics):
        results[f'train_metric_{index + 1}'] = []
        results[f'loss_metric_{index + 1}'] = []

    best_model, best_loss = None, None

    for epoch in tqdm(range(max_epochs)):
        print(f"Epoch n: {epoch + 1} started")
        train_results = train_per_epoch(model=model,
                                        train_dataloader=train_dataloader,
                                        loss_function=loss_function,
                                        optimizer=optimizer,
                                        output_layer=output_layer,
                                        scheduler=scheduler,
                                        device=device,
                                        report_batch=report_batch)

        train_loss, train_metrics = train_results[0], train_results[1:]

        # the test function can have a loss initiated in the call as it doesn't call the backwards function
        # no back propagation takes place
        val_results = val_per_epoch(model=model,
                                    dataloader=test_dataloader,
                                    loss_fn=loss_function,
                                    output_layer=output_layer,
                                    device=device)

        val_loss, val_metrics = val_results[0], val_results[1:]
        # track the best performing model on the validation portion
        if epoch >= min_eval_epoch and (best_loss is None or best_loss >= val_loss):
            best_loss = val_loss
            best_model = model

        if progress:
            print("training metrics")
            print(f"training loss: {train_loss}")
            for i, m in train_metrics:
                print(f"training metric {i}: {m}")

            print("#" * 100)

            print("validation metrics")
            print(f"validation loss: {train_loss}")
            for i, m in val_metrics:
                print(f"validation metric {i}: {m}")

        # save the epoch's statistics
        results['train_loss'].append(train_loss)
        results['val_loss'].append(val_loss)

        for i, m in train_metrics:
            results[f'train_metric_{i}'].append(m)
            print(f"training metric {i}: {m}")

        for i, m in val_metrics:
            print(f"validation metric {i}: {m}")

        if writer is not None:
            # track loss results
            writer.add_scalars(main_tag='Loss',
                               tag_scalar_dict={"train_loss": train_loss, 'val_loss': val_loss},
                               global_step=epoch)


            writer.add_scalars(main_tag='Accuracy',
                               tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc},
                               global_step=epoch)

            writer.close()

        # check if the losses reached the minimum thresholds
        if (min_train_loss is not None and min_train_loss >= train_loss) or (
                min_val_loss is not None and min_val_loss >= val_loss):
            # the model that goes lower than these thresholds is automatically the best model
            break

    # in addition to the model save all the details:
    # build the details:
    details = {OPTIMIZER: optimizer,
               SCHEDULER: scheduler,
               MAX_EPOCHS: max_epochs,
               MIN_TRAIN_LOSS: min_train_loss,
               MIN_VAL_LOSS: min_val_loss,
               'layers': model.num_layers}

    save_info(save_path=save_path, details=details)
    save_model(best_model, path=save_path)
    return results


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
