"""
"""

import torch

from torch.utils.data import DataLoader
from torch import nn
from typing import Union, Tuple, Dict
from torch.optim import lr_scheduler

from pytorch_modular.CBM.model.CBM import CBModel
from pytorch_modular.CBM.model.loss import CBMLoss


def set_output_layer(model_output: torch.Tensor) -> nn.Module:
    # this function simply inspects the dimension of each single output
    # and return the suitable output layer
    if model_output.dim() != 2:
        raise ValueError(f"The model's output is expected to be 2 dimensional. Found {model_output.dim()}")
    
    num_samples, dim = model_output.shape 
    
    if dim == 1:
        return nn.Sigmoid()

    return nn.Softmax(dim=1)


def train_CBM_per_epoch(model: CBModel,
                        train_dataloader: DataLoader[torch.Tensor, torch.Tensor, torch.Tensor],
                        loss_function: CBMLoss,
                        optimizer: torch.optim.Optimizer,
                        output_layer: Union[nn.Module, callable] = None,
                        scheduler: lr_scheduler = None,
                        device: str = None
                        ) -> Tuple[Dict[str, float], Dict[str, float]]:

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    # set the model to correct device and state
    model.train()
    model.to(device)

    # training metrics
    train_cl_loss = 0
    train_concept_loss = 0
    train_loss = 0
    train_accuracy = 0

    for _, (x, concept_y, y_class) in enumerate(train_dataloader):
        # set optimizer grad to zero
        optimizer.zero_grad()

        # set to device
        x, y_concept, y_class = x.float().to(device), concept_y.float().to(device), y_class.float().to(device)
        loss_function = loss_function.to(device)
        y_pred_class, y_concept_preds = model.forward(x)

        # make sure to return all losses to be tracked
        class_loss, concept_loss, batch_loss = loss_function.forward(y_pred_class=y_pred_class,
                                                                     y_class=y_class,
                                                                     y_concept_preds=y_concept_preds,
                                                                     y_concept=y_concept,
                                                                     return_all=True)

        # backward pass
        batch_loss.backward()

        # make sure to keep track of the losses
        train_loss += batch_loss.item()
        train_concept_loss += concept_loss.item()
        train_cl_loss += class_loss.item()

        # optimizer's step
        optimizer.step()

        y_pred_class = output_layer(y_pred_class)

    # update the learning rate at the end of each epoch
    if scheduler is not None:
        scheduler.step()

    # make sure to average the different metrics
    train_accuracy = train_accuracy / len(train_dataloader)
    train_loss = train_loss / len(train_dataloader)
    train_cl_loss = train_cl_loss / len(train_dataloader)
    train_concept_loss = train_concept_loss / len(train_dataloader)

    return ({"loss": train_loss, "class_loss": train_cl_loss, "concept_loss": train_concept_loss},
            {"accuracy": train_accuracy})


def val_CBM_per_epoch(model: CBModel,
                      val_dataloader: DataLoader[torch.Tensor, torch.Tensor, torch.Tensor],
                      loss_function: CBMLoss,
                      output_layer: Union[nn.Module, callable] = None,
                      device: str = None
                      ) -> tuple[Dict[str, float], Dict[str, float]]:
    # set the device1
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    # set the model to correct device and state
    model.eval()
    model.to(device)
    
    # metrics    
    val_cl_loss = 0
    val_concept_loss = 0
    val_loss = 0
    val_accuracy = 0

    with torch.inference_mode():
        for _, (x, concept_y, y_class) in enumerate(val_dataloader):
            # set to device
            x, y_concept, y_class = x.float().to(device), concept_y.float().to(device), y_class.float().to(device)
            loss_function = loss_function.to(device)
            y_pred_class, y_concept_preds = model.forward(x)

            # track all losses
            class_loss, concept_loss, batch_loss = loss_function.forward(y_pred_class=y_pred_class,
                                                                         y_class=y_class,
                                                                         y_concept_preds=y_concept_preds,
                                                                         y_concept=y_concept,
                                                                         return_all=True)
        
            # make sure to keep track of the losses
            val_loss += batch_loss.item()
            val_concept_loss += concept_loss.item()
            val_cl_loss += class_loss.item()

            # predict the classes
            y_pred_class = output_layer(y_pred_class)

        # make sure to average the different metrics
        val_accuracy = val_accuracy / len(val_dataloader)
        val_loss = val_loss / len(val_dataloader)
        val_cl_loss = val_cl_loss / len(val_dataloader)
        val_concept_loss = val_concept_loss / len(val_dataloader)

    return ({"loss": val_loss, "class_loss": val_cl_loss, "concept_loss": val_concept_loss},
            {"accuracy": val_accuracy})
