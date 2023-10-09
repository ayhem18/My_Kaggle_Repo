"""
This scripts contains utility functions as well as some constants used across different
scripts in the image_classification directory
"""

# constant to save the string literal of important variables
TRAIN_LOSS = 'train_loss'
VAL_LOSS = 'val_loss'
TEST_LOSS = 'test_loss'

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
REPORT_EPOCH = 'report_epoch'
COMPUTE_LOSS = 'compute_loss'

# the number of epochs to discard before considering the best model
MIN_EVALUATION_EPOCH = 'min_evaluation_epoch'

# if the model does not reach a lower training loss than the current lowest loss after 'n' consecutive epochs,
# the training will stop
NO_IMPROVE_STOP = 'no_improve_stop'
DEBUG = 'debug'

MIN_NO_IMPROVE_STOP = 15

