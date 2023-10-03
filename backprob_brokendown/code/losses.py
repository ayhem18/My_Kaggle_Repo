import random
import torch

import numpy as np

from typing import Union
from torch import nn

random.seed(69)
torch.manual_seed(69)
np.random.seed(69)

num = Union[float, int]


def binaryCrossEntropy(x: Union[np.ndarray, num], y: Union[np.ndarray, num]) -> Union[float, np.ndarray]:
    """This is a simple implementation of the binary cross entropy loss

    Args:
        x (Union[np.ndarray, num]): The input to the loss (or more precisely the probabilities) 
        y (Union[np.ndarray, num]): the target

    Returns:
        Union[float, np.ndarray]: The value of the loss(es) depending on the input
    """
    # to avoid writing code for both scalar and vector cases, let's unify the input 
    if isinstance(x, (float, int)):
        x = np.array([x], dtype=np.float32)
    else:
        x = np.squeeze(x)

    if isinstance(y, (float, int)):
        y = np.array([y]).astype(int)
    else:
        y = np.squeeze(y)

    # the target should have only '1' or '0' values
    # make sure to use numpy functions with working with numpy arrays
    if not np.all((y == 0) | (y == 1)):
        raise ValueError((f"The target vector is expected to be all ones and zeros.\n"
                          f"Found: {y[(y != 0) & (y != 1)][0]}"))

    # make sure all elements in the input are probability-like
    if not np.all((1 >= x) & (x >= 0)):
        raise ValueError((f"The input values are expected to represent probabilities.\n"
                          f"Found: {x[(x < 0) | (x > 1)][0]}"))

    # make sure the inputs are of the same shape
    if x.shape != y.shape:
        raise ValueError((f"the input and target are expected to be of the same shape. \n"
                          f"Found shapes  {x.shape} and {y.shape}"))

    # time for the main piece of code
    # numpy '*' will apply element-wise multiplication
    result = -(np.log(x) * y + np.log(1 - x) * (1 - y))
    # make sure to convert back to float if the input was indeed float
    return result.item() if isinstance(y, num) else result


# let's write a function to thoroughly test the custom function using pytorch
def testCustomBCE(num_test: int = 100):
    for _ in range(num_test):
        loss = nn.BCELoss(reduction='none')
        dim = random.randint(1, 20)
        x = torch.rand(dim)
        x_np = x.numpy()
        target = torch.bernoulli(torch.rand(dim))
        target_np = target.numpy()

        torch_loss = loss(x, target)
        custom_loss = binaryCrossEntropy(x_np, target_np)
        # convert torch loss to numpy
        torch_loss = torch_loss.numpy()
        try:
            tl = torch_loss.item()
            assert abs(tl - custom_loss) <= 10 ** -5, "The loss for the scalar case is not correct"
        except:
            # this block of code is reached for the vector case
            assert np.allclose(torch_loss, custom_loss), "The loss for the vectorized case is not correct"


def crossEntropyLoss(y_pred: np.ndarray, y: Union[np.ndarray, num]) -> Union[np.ndarray, num]:
    # first let's check the dimensions
    # y_pred should be of shape (n, K) where K > 2
    if len(y_pred.shape) != 2 or y_pred.shape[-1] <= 2:
        raise ValueError(f'The predictions are expected to be of shape {(None, "n")} where {"n"} is larger than 2')
    num_classes = y_pred.shape[-1]

    if isinstance(y, num):
        y = np.array([y], dtype=np.int32)
    else:
        y = np.squeeze(y)

    # make sure all the target are less than 'num_classes':
    if not np.all((num_classes > y) & (y >= 0)):
        raise ValueError((f"The targets are expected to be in the range [0, number of classes]\n"
                          f"Found: {y[(y < 0) | (y < num_classes)][0]}"))

    # make sure the shapes match
    if y.shape[0] != y_pred.shape[0]:
        raise ValueError((f"The number of predictions and labels must match\n"
                          f"Found {y.shape[0]} predictions and {y_pred.shape[0]} targets"))

    # for a single example say x_i = [p1, p2, ... p_k] where p_i is the probability of the i-th class
    # then the cross entropy loss is -log(pj) where p_j where 'j' is the correct class for x_i
    # so the idea here is to index y_pred accroding to 'y' and then apply -log
    result = -np.log(y_pred[list(range(len(y_pred))), y])
    return result.item() if isinstance(y, num) else result


def test_CEL(num_test: int = 100):
    for _ in range(num_test):
        loss = nn.NLLLoss(reduction='none')
        num_classes = random.randint(3, 20)
        num_samples = random.randint(10, 100)

        x = torch.rand(num_samples, num_classes)
        # make sure to normalize so values are in [0, 1]
        x = x / torch.sum(x, dim=0)
        x_np = x.numpy()

        y_np = np.array([random.randint(0, num_classes - 1) for _ in range(num_samples)])
        y = torch.from_numpy(y_np)

        # torch does not provide functions that would calculate the cross entropy loss directly from predictions.
        # nn.CrossEntropy loss expects logits (the numbers before applying softmax)
        # nn.NLLLoss expects the logarithm of predictions which explains torch.log(x) in the loss call below 

        torch_loss = loss(torch.log(x), y)
        torch_loss = torch_loss.numpy()
        custom_loss = crossEntropyLoss(x_np, y_np)
        # convert torch loss to numpy
        try:
            tl = torch_loss.item()
            assert abs(tl - custom_loss) <= 10 ** -5, "The loss for the scalar case is not correct"
        except:
            # this block of code is reached for the vector case
            assert np.allclose(torch_loss, custom_loss), "The loss for the vectorized case is not correct"


if __name__ == '__main__':
    test_CEL()
