"""
This scripts contains functionality blocks used in each epoch
"""
import warnings
import random
import torch

from typing import Union, Dict, Tuple
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from src.pytorch_modular.pytorch_utilities import get_default_device
from src.pytorch_modular.image_classification.classification_metrics import accuracy, ACCURACY
from src.pytorch_modular.image_classification.debug import debug_val_epoch
from src.pytorch_modular.visual import plot_images


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
                    debug: bool = False,
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
        debug:

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
        if not train_dataloader.drop_last:
            raise ValueError(f"Please make sure to set the parameter 'drop_last' in the dataloader"
                             f"to {True} to avoid any misleading decrease in performance")

    # make sure the train_dataloader shuffles the data
    if hasattr(train_dataloader, 'shuffle'):
        train_dataloader.shuffle = True

    for _, (x, y) in enumerate(train_dataloader):
        # the idea is quite simple here, visualize the image when needed
        if debug:
            if random.random() <= 0.1:
                batch_size = x.size(0)
                random_index = random.choice(range(batch_size))
                input_as_image = x[random_index].cpu().detach().numpy()
                label_int = y[random_index].cpu().detach().numpy().item()
                plot_images(images=input_as_image, captions=[label_int])

        # depending on the type of the dataset and the dataloader, the labels can be either 1 or 2 dimensional tensors
        # the first step is to squeeze them
        y = torch.squeeze(y, dim=-1)
        # THE LABELS MUST BE SET TO THE LONG DATATYPE
        x, y = x.to(device), y.to(torch.long).to(device)
        # pass the 1-dimensional label tensor to the loss function. In case the loss function expects 2D tensors, then
        # the exception will be caught and the extra dimension will be added
        y_pred = model(x)
        try:
            batch_loss = loss_function(y_pred, y)
        except RuntimeError:
            # un-squeeze the y
            new_y = torch.unsqueeze(y, dim=-1).to(torch.long).to(device)
            warnings.warn(
                f"An extra dimension has been added to the labels vectors"
                f"\nold shape: {y.shape}, new shape: {new_y.shape}")
            batch_loss = loss_function(y_pred, new_y)

        train_loss += batch_loss.item()
        optimizer.zero_grad()
        batch_loss.backward()
        # optimizer's step
        optimizer.step()

        y_pred_class = output_layer(y_pred)

        # calculate the metrics for
        for metric_name, metric_func in metrics.items():
            train_metrics[metric_name] += metric_func(y, y_pred_class)

    # update the learning rate at the end of each epoch
    if scheduler is not None:
        scheduler.step()

    # average the loss and the metrics
    # make sure to add the loss before averaging the 'train_loss' variable
    train_metrics['train_loss'] = train_loss
    for metric_name, _ in train_metrics.items():
        train_metrics[metric_name] /= len(train_dataloader)

    return train_metrics


def val_per_epoch(model: nn.Module,
                  dataloader: DataLoader[torch.tensor],
                  loss_function: nn.Module,
                  output_layer: Union[nn.Module, callable],
                  device: str = None,
                  metrics: Union[str, Dict[str, callable]] = None,
                  debug: bool = False
                  ) -> Dict[str, float]:
    """
    This function evaluates a given model on a given test split of a dataset
    Args:
        debug:
        model: the given model
        dataloader: the loader guaranteeing access to the test split
        loss_function: the loss function the model tries to minimize on the test split
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
            # depending on the type of the dataset and the dataloader, the labels can be either 1 or 2 dimensional tensors
            # the first step is to squeeze them
            y = torch.squeeze(y, dim=-1)
            # THE LABELS MUST BE SET TO THE LONG DATATYPE
            x, y = x.to(device), y.to(torch.long).to(device)
            y_pred = model(x)
            
            # calculate the loss, and backprop
            try:
                loss = loss_function(y_pred, y)
            except RuntimeError:
                # un-squeeze the y
                new_y = torch.unsqueeze(y, dim=-1)
                warnings.warn(
                    f"An extra dimension has been added to the labels vectors"
                    f"\nold shape: {y.shape}, new shape: {new_y.shape}")
                loss = loss_function(y_pred, new_y.float())

            val_loss += loss.item()

            predictions = output_layer(y_pred)

            for metric_name, metric_func in metrics.items():
                val_metrics[metric_name] += metric_func(y, predictions)

            if debug:
                print("#" * 25)
                print()
                print(val_loss)
                for metric_name, metric_func in metrics.items():
                    print(metric_func(y, predictions))
                debug_val_epoch(x, y, predictions)

    # make sure to add the loss without averaging the 'val_loss' variable
    val_metrics['val_loss'] = val_loss
    for name, metric_value in val_metrics.items():
        val_metrics[name] = metric_value / len(dataloader)

    return val_metrics
