import os
import sys

HOME = os.getcwd()
sys.path.append(HOME)
sys.path.append(os.path.join(HOME, 'src'))

# first import the EfficientNet architecture
from torch.utils.tensorboard import SummaryWriter
# let's start small
from torch import nn
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import lr_scheduler
from typing import Optional, Union
from pathlib import Path
from src.pytorch_modular.image_classification.engine_classification \
    import train_per_epoch, val_per_epoch, binary_output
from src.pytorch_modular.pytorch_utilities import input_shape_from_dataloader, save_model, get_default_device
from src.pytorch_modular.exp_tracking import save_info
from torchvision.models import resnet101, vgg11_bn, vgg16, VGG16_Weights


class ClassifierHead(nn.Module):
    def __init__(self, num_classes: int, num_layers: int, num_input_features: int):
        # as usual call the super class constructor
        super().__init__()
        # the shape used in the classifier's output
        self.output = num_classes if num_classes > 2 else 1
        self.num_layers = num_layers
        self.features = num_input_features
        self._build_classifier()

    def _build_classifier(self):
        base_power = int(np.log2(self.features))
        powers = np.linspace(start=int(np.log2(self.output)), stop=base_power, num=self.num_layers)
        # make sure to convert to integers
        num_units = [int(2 ** p) for p in powers][::-1]
        # set the last element to the actual number of classes
        num_units[-1] = self.output
        num_units = [self.features] + num_units

        layers = [nn.Linear(in_features=num_u, out_features=num_units[i + 1]) for i, num_u in enumerate(num_units[:-1])]
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers[:-1]:
            x = F.relu(l(x))

        return self.layers[-1](x)


class DVC_Classifier(nn.Module):

    def __set_feature_extractor(self):
        # this function prepares the feature extractor
        for param in self.net.parameters():
            param.requires_grad = False

    def __init__(self, num_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # save the weights for later use
        self.num_layers = num_layers
        self.weights = VGG16_Weights.DEFAULT
        self.net = vgg16(self.weights)
        # make sure the weights the feature extract is not trainable
        self.__set_feature_extractor()
        # build the classifier
        num_features = self.net.classifier[0].in_features
        classifier = ClassifierHead(num_classes=2, num_layers=num_layers, num_input_features=num_features)
        # set the classifier in the pretrained model
        self.net.classifier = classifier

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def get_transformations(self):
        return self.weights.transforms()

    def __str__(self):
        return f'vgg16_{self.num_layers}'


def train_model(model: DVC_Classifier,
                train_dataloader: DataLoader,
                test_dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: lr_scheduler = None,
                min_train_loss: float = None,
                min_test_loss: float = None,
                epochs: int = 5,
                device: str = None,
                print_progress=False,
                report_batch: int = None,
                writer: Optional[SummaryWriter] = None,
                save_path: Optional[Union[Path, str]] = None
                ):
    # set the device
    if device is None:
        device = get_default_device()

    # 2. Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    # set the model to the current device
    model.to(device)
    # make sure to define the loss function separately for the training part
    # as the loss function should be the exact same object (to save the results of the back propagation each time)
    train_loss_fn = nn.BCEWithLogitsLoss()

    best_model = None
    best_loss = None

    for epoch in tqdm(range(epochs)):
        print(f"Epoch n: {epoch + 1} started")
        train_loss, train_acc = train_per_epoch(model=model,
                                                dataloader=train_dataloader,
                                                loss_fn=train_loss_fn,  # don't forget the parentheses
                                                optimizer=optimizer,
                                                output_layer=lambda x: binary_output(x),
                                                scheduler=scheduler,
                                                device=device,
                                                report_batch=report_batch)
        # the test function can have a loss initiated in the call as it doesn't call the backwards function
        # no back propagation takes place
        test_loss, test_acc = val_per_epoch(model=model, dataloader=test_dataloader, loss_function=,
                                            output_layer=lambda x: binary_output(x), device=device)

        # make sure to track the best performing model on the test portion
        if epoch >= 5 and (best_loss is None or test_loss < best_loss):
            best_loss = test_loss
            best_model = model

        if print_progress:
            # 4. Print out what's happening
            print(
                f"Epoch: {epoch + 1} | "
                f"train_loss: {train_loss:.6f} | "
                f"train_acc: {train_acc:.6f} | "
                f"test_loss: {test_loss:.6f} | "
                f"test_acc: {test_acc:.6f}"
            )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if writer is not None:
            # track loss results
            writer.add_scalars(main_tag='Loss',
                               tag_scalar_dict={"train_loss": train_loss, 'test_loss': test_loss},
                               global_step=epoch)

            writer.add_scalars(main_tag='Accuracy',
                               tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc},
                               global_step=epoch)

            # to get the shape of the model's input, we can use the train_dataloader
            writer.add_graph(model=model,
                             input_to_model=torch.randn(size=input_shape_from_dataloader(train_dataloader)).to(device))

            writer.close()

        # check if the errors reached their the thresholds
        if (min_train_loss is not None and min_train_loss >= train_loss) or (
                min_test_loss is not None and min_test_loss >= test_loss):
            # the model that goes lower than these thresholds is automatically the best model
            break

    # in addition to the model save all the details:
    # build the details:
    details = {'optimization_algo': optimizer,
               'scheduler': scheduler,
               'epochs': epochs,
               'min_train_loss': min_train_loss,
               'min_test_loss': min_test_loss,
               'layers': model.num_layers}

    save_info(save_path=save_path, details=details)
    save_model(best_model, path=save_path)
    return results


if __name__ == '__main__':
    net1 = resnet101()
    net2 = vgg11_bn()
    print(net1)
    print("#" * 100)
    print("\n" * 4)
    print(net2)
