import os
from _collections_abc import Sequence
from typing import Union
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from src.pytorch_modular.pytorch_utilities import get_default_device
import numpy as np
import itertools

HOME = os.getcwd()


def accuracy(y_pred: torch.tensor, y: torch.tensor) -> float:
    # squeeze values if needed
    value = (y_pred == y).type(torch.float32).mean().item()
    return value


def train_per_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    loss_fn: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    output_layer: nn.Module,
                    scheduler: lr_scheduler = None,
                    metrics: Union[Sequence[callable], callable] = None,
                    device: str = None,
                    report_batch: int = None) -> tuple:
    # first set the default value of the device parameter
    if device is None:
        device = get_default_device()

    # the default metric for classification is accuracy
    if metrics is None:
        metrics = accuracy

    if not isinstance(metrics, Sequence):
        metrics = [metrics]

    # put the mode in train mode
    model.train()

    # set up the train loss of a model
    train_loss = 0
    train_metrics = [0 for _ in metrics]

    for batch, (x, y) in enumerate(dataloader):
        if report_batch is not None and batch % report_batch == 0:
            print(f'batch n {batch + 1} LOADED !!')

        x, y = x.to(device), y.to(torch.long).to(device)  # convert to Long Type

        # make sure to un-squeeze 'y' if it is only 1 dimensions
        if len(y.shape) == 1:
            y = torch.unsqueeze(y, dim=-1)

        optimizer.zero_grad()
        # get the forward pass first
        y_pred = model(x)

        # calculate the loss
        batch_loss = loss_fn(y_pred, y.float())
        batch_loss.backward()
        # add the batch loss to the general training loss
        optimizer.step()

        train_loss += batch_loss.item()
        y_pred_class = output_layer(y_pred)

        # calculate the different metrics needed:
        metrics_results = [m(y_pred_class, y) for m in metrics]

        # add the batch metrics to the train metrics in general
        for index, mr in enumerate(metrics_results):
            train_metrics[index] += mr

    # adjust metrics to get the average loss and average metrics
    train_loss = train_loss / len(dataloader)
    train_metrics = tuple([m / len(dataloader) for m in train_metrics])

    # update the learning rate at the end of each epoch
    if scheduler is not None:
        scheduler.step()

    return (train_loss,) + train_metrics


def test_per_epoch(model: nn.Module,
                   dataloader: DataLoader,
                   loss_fn: nn.Module,
                   output_layer: nn.Module,
                   metrics: Union[Sequence[callable], callable] = None,
                   device: str = None) -> tuple:
    # set the device
    if device is None:
        device = get_default_device()

    # the default metric for classification is accuracy
    if metrics is None:
        metrics = accuracy

    if not isinstance(metrics, Sequence):
        metrics = [metrics]

    # put the model to the evaluation mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss = 0
    metric_values = [0 for _ in metrics]

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for _, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            # make sure to add an extra dimension to 'y' if it is uni dimensional
            if len(y.shape) == 1:
                y = torch.unsqueeze(y, dim=-1)

            y_pred = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(y_pred, y.float())
            test_loss += loss.item()

            labels = output_layer(y_pred)
            # calculate the different metrics needed:
            metrics_results = [m(labels, y) for m in metrics]

            # add the batch metrics to the train metrics in general
            for index, mr in enumerate(metrics_results):
                metric_values[index] += mr

    # adjust metrics to get the average loss and average metrics
    loss = test_loss / len(dataloader)
    metric_values = tuple([m / len(dataloader) for m in metric_values])

    return (loss,) + metric_values


VALID_RETURN_TYPES = ['np', 'pt', 'list']


def predict(model: nn.Module,
            dataloader: DataLoader[torch.tensor],
            output_layer: Union[nn.Module, callable],
            device: str = None,
            return_tensor: str = 'np') -> Union[np.ndarray, torch.Tensor, list[int]]:
    # set the device
    if device is None:
        device = get_default_device()

    # make sure the return_tensor argument is a set to a valid value
    assert return_tensor in VALID_RETURN_TYPES, f"PLEASE SET THE 'return_tensor' argument to one of these values: {VALID_RETURN_TYPES}"

    # set to the inference model
    model.eval()

    # set the model to the same device as the data
    model.to(device)

    with torch.inference_mode():
        result = [output_layer(model(X.to(device))) for X in dataloader]

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


def binary_output(x: torch.Tensor) -> torch.IntTensor:
    sigma = nn.Sigmoid()
    output = sigma(x).to(torch.int32)
    return output
