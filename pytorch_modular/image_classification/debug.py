"""
This script contains a number of functionalities to debug the training of image classifier
"""
import torch
from typing import Union
import math
import cv2 as cv


def debug_val_epoch(x: torch.Tensor,
                    y: torch.Tensor,
                    predictions: torch.Tensor,
                    samples: Union[float, int] = 0.01,
                    ) -> None:
    
    batch_size = x.size(0)
    original_shape = tuple(x.shape[1:])
    correct_prediction_mask = (y != predictions)


    x_to_mask = x.permute((1, 2, 3, 0))    
    x_to_mask = x_to_mask.reshape(shape=(-1, batch_size))

    correct_samples = (torch.masked_select(x_to_mask, correct_prediction_mask)
                       .reshape((-1, int(correct_prediction_mask.sum()))))
    
    correct_predictions = torch.masked_select(predictions, correct_prediction_mask)

    # time to flip the dimensions once again
    correct_samples = correct_samples.permute((1, 0))

    correct_display = int(math.ceil(len(correct_samples) * samples)) if isinstance(samples, float) else samples

    for x, label in zip(correct_samples[:correct_display], correct_predictions[:correct_display]):
        x = x.reshape(shape=original_shape) # since numpy uses channels at the end
        # first transpose for the image to make sense
        x = x.cpu().detach().permute((1, 2, 0)).numpy()
        # display 'x'
        cv.imshow(f'correct_image_as_{label}', x)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
