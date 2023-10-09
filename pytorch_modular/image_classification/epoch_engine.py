"""
This scripts contains functionality blocks used in each epoch
"""
import warnings
import random
import torch

import src.pytorch_modular.image_classification.utilities as ut 

from typing import Union, Dict, Tuple, Optional
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from copy import deepcopy

from src.pytorch_modular.pytorch_utilities import get_default_device
from src.pytorch_modular.image_classification.classification_metrics import accuracy, ACCURACY
from src.pytorch_modular.image_classification.debug import debug_val_epoch
from src.pytorch_modular.visual import plot_images, display_image


def _set_default_parameters(device: str = None,
                            metrics: Union[str, Dict[str, callable]] = None
                            ) -> Tuple[str, Dict[str, callable]]:
    # set default arguments
    device = get_default_device() if device is None else device

    # a None 'metrics' will map only to accuracy
    metrics = {ACCURACY: accuracy} if metrics is None else metrics

    return device, metrics

def train_per_epoch(model: nn.Module, 
                    train_dataloader: DataLoader[torch.Tensor], 
                    optimizer: torch.optim.Optimizer, 
                    compute_loss: Optional[callable]=None, 
                    output_layer: Union[nn.Module, callable]=None,
                    scheduler: lr_scheduler = None,
                    loss_function: nn.Module = None,
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
    # make sure that either function_loss  and output_layer are both not None or compute_loss is not None
    if compute_loss is None and (output_layer is None or loss_function is None):
        raise ValueError(f"either 'output_layer' and 'loss_function' are not None, or 'compute_loss' is not None")

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

    last_parameters = None 

    for _, (x, y) in enumerate(train_dataloader):
        # the idea is quite simple here, visualize the image when needed
        if debug:
            if random.random() <= 0.1:
                batch_size = x.size(0)
                random_index = random.choice(range(batch_size))
                input_as_image = x[random_index]
                display_image(input_as_image)

            # iterate through the weights of a model to make sure they are indeed changing
            # if last_parameters is None:
            #     last_parameters = list(model.parameters())
            # else: 
            #     changed = False
            #     new_p = deepcopy(list(model.parameters()))
            #     for lp, p in zip(last_parameters, new_p):
            #         changed = not torch.allclose(lp, p)
            #         if changed:
            #             last_parameters = new_p
            #             continue
            #     if not changed:
            #         raise ValueError("The model's weight were not updated!!!")
            #     last_parameters = new_p
                
        optimizer.zero_grad()

        # depending on the type of the dataset and the dataloader, the labels can be either 1 or 2 dimensional tensors
        # the first step is to squeeze them
        # THE LABELS MUST BE SET TO THE LONG DATATYPE
        x, y = x.to(device), y.float().squeeze().to(device)
        y_pred = model(x)
        loss_function = loss_function.to(device)
        if compute_loss is None:
            # pass the 1-dimensional label tensor to the loss function. In case the loss function expects 2D tensors, then
            # the exception will be caught and the extra dimension will be added
            try:
                batch_loss = loss_function(y_pred, y)
            except (RuntimeError, ValueError):
                # un-squeeze the y
                new_y = y.unsqueeze(dim=-1).to(device)
                # new_y = torch.unsqueeze(y, dim=-1).to(device)
                warnings.warn(
                    f"An extra dimension has been added to the labels vectors"
                    f"\nold shape: {y.shape}, new shape: {new_y.shape}")
                batch_loss = loss_function(y_pred, new_y)
        else:
            batch_loss = compute_loss(y_pred, y)
        
        train_loss += batch_loss.item()
        batch_loss.backward()
        # optimizer's step
        optimizer.step()

        y_pred_class = output_layer(y_pred)

        # calculate the metrics for
        for metric_name, metric_func in metrics.items():
            train_metrics[metric_name] += metric_func(y_pred_class, y)

    # update the learning rate at the end of each epoch
    if scheduler is not None:
        scheduler.step()

    # average the loss and the metrics
    # make sure to add the loss before averaging the 'train_loss' variable
    train_metrics[ut.TRAIN_LOSS] = train_loss
    for metric_name, _ in train_metrics.items():
        train_metrics[metric_name] /= len(train_dataloader)

    return train_metrics


def val_per_epoch(model: nn.Module,
                  dataloader: DataLoader[torch.tensor],
                  compute_loss: nn.Module=None,
                  loss_function: nn.Module=None,
                  output_layer: Union[nn.Module, callable]=None,
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

    # make sure that either function_loss  and output_layer are both not None or compute_loss is not None
    if compute_loss is None and (output_layer is None or loss_function is None):
        raise ValueError(f"either 'output_layer' and 'loss_function' are not None, or 'compute_loss' is not None")


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
            x, y = x.to(device), y.to(torch.long).squeeze().to(device)
            y_pred = model(x)

            if compute_loss is None:
                # calculate the loss, and backprop
                try:
                    loss = loss_function(y_pred, y)
                except RuntimeError:
                    # un-squeeze the y
                    new_y = torch.unsqueeze(y, dim=-1)
                    warnings.warn(
                        f"An extra dimension has been added to the labels vectors"
                        f"\nold shape: {y.shape}, new shape: {new_y.shape}")
                    loss = loss_function(y_pred, new_y.squeeze().float())
            else:
                loss = compute_loss(y_pred, y)

            val_loss += loss.item()

            predictions = output_layer(y_pred)

            for metric_name, metric_func in metrics.items():
                val_metrics[metric_name] += metric_func(y, predictions)

            # if debug:
            #     print("#" * 25)
            #     print()
            #     print(val_loss)
            #     for metric_name, metric_func in metrics.items():
            #         print(metric_func(predictions, y))
            #     debug_val_epoch(x, y, predictions)

    # make sure to add the loss without averaging the 'val_loss' variable
    val_metrics[ut.VAL_LOSS] = val_loss
    for name, metric_value in val_metrics.items():
        val_metrics[name] = metric_value / len(dataloader)

    return val_metrics
