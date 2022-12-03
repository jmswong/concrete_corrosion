from config import (MAX_NUM_EPOCHS, GRACE_PERIOD, EPOCHS, CPU, GPU,
                    NUM_SAMPLES, DATA_ROOT_DIR, NUM_WORKERS, VALID_SPLIT)
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os


def random_search():
    pass


if __name__ == '__main__':
    random_search()
