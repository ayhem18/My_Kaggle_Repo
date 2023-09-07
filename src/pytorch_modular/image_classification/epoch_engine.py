"""
This scripts contains functionality blocks used in each epoch
"""
import torch

from typing import Union, Dict, Tuple
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from src.pytorch_modular.pytorch_utilities import get_default_device
from src.pytorch_modular.image_classification.classification_metrics import accuracy, ACCURACY


def _set_default_parameters(device: str = None,
                            metrics: Union[str, Dict[str, callable]] = None
                            ) -> Tuple[str, Dict[str, callable]]:
    # set default arguments
    device = get_default_device() if device is None else device

    # a None 'metrics' will map only to accuracy
    metrics = {ACCURACY: accuracy} if metrics is None else metrics

    return device, metrics


def train_per_epoch(model: nn.Module,
                    train_dataloader: DataLoader[torch.tensor],
                    loss_function: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    output_layer: Union[nn.Module, callable],
                    scheduler: lr_scheduler,
                    device: str = None,
                    metrics: Union[str, Dict[str, callable]] = None,
                    report_batch: int = None
                    ) -> Dict[str, float]:
    """
    This function update a model's weights during a single pass across the entire dataset
    Args:
        model: the model to train
        train_dataloader: a data loader that iterates through
        loss_function: the loss to minimize
        optimizer: the optimization algorithm
        output_layer: The model is assumed to output logits, this layer converts them to labels (to compute metrics)
        scheduler: responsible for adjusting the learning rate
        metrics: defaults to only accuracy
        device: The device on which the model will run
        report_batch: determines the frequency of reporting the metrics results

    Returns: A dictionary with loss value on the training data as well as the different given metrics
    """

    # set the default arguments
    device, metrics = _set_default_parameters(device, metrics)

    # set the model to correct device and state
    model.train()
    model.to(device)
    # set the train loss and metrics
    train_loss, train_metrics = 0, dict([(name, 0) for name, _ in metrics.items()])

    # make sure to set the `drop_last` field in the dataloader to True,
    # as it might affect the metrics
    if hasattr(train_dataloader, 'drop_last'):
        train_dataloader.drop_last = True
    # make sure the train_dataloader shuffles the data
    if hasattr(train_dataloader, 'shuffle'):
        train_dataloader.shuffle = True

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

        # calculate the metrics for
        for metric_name, metric_func in metrics.items():
            train_metrics[metric_name] += metric_func(y, y_pred_class)

    # update the learning rate at the end of each epoch
    if scheduler is not None:
        scheduler.step()

    # make sure to add the loss before averaging the 'train_loss' variable
    train_metrics['train_loss'] = train_loss
    # average the loss and the metrics
    for metric_name, metric_value in train_metrics.items():
        train_metrics[metric_name] /= len(train_dataloader)

    return train_metrics


def val_per_epoch(model: nn.Module,
                  dataloader: DataLoader[torch.tensor],
                  loss_fn: nn.Module,
                  output_layer: nn.Module,
                  device: str = None,
                  metrics: Union[str, Dict[str, callable]] = None
                  ) -> Dict[str, float]:
    """
    This function evaluates a given model on a given test split of a dataset
    Args:
        model: the given model
        dataloader: the loader guaranteeing access to the test split
        loss_fn: the loss function the model tries to minimize on the test split
        output_layer: The model is assumed to output logits, this layer converts them to labels (to compute metrics)
        metrics: defaults to only accuracy
        device: The device on which the model will run
    Returns: A dictionary with loss value on the training data as well as the different given metrics
    """
    # set the default arguments
    device, metrics = _set_default_parameters(device, metrics)

    val_loss, val_metrics = 0, dict([(name, 0) for name, _ in metrics.items()])

    # set the model to the correct device and state
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

            for metric_name, metric_func in metrics.items():
                val_metrics[metric_name] += metric_func(labels, y)

    # make sure to add the loss without averaging the 'val_loss' variable
    val_metrics['val_loss'] = val_loss
    for name, metric_value in val_metrics.items():
        val_metrics[name] = metric_value / len(dataloader)

    return val_metrics