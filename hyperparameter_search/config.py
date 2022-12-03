import os

# Training parameters.
EPOCHS = 1000
# Data root.
DATA_ROOT_DIR = os.path.abspath('/home/wongjames/cs230/Project/data_12_2_2022')
# Number of parallel processes for data fetching.
NUM_WORKERS = 4
# Ratio of split to use for validation.
VALID_SPLIT = 0.2
# For ASHA scheduler in Ray Tune.
MAX_NUM_EPOCHS = 1000
GRACE_PERIOD = 1
# For search run (Ray Tune settings).
CPU = 1
GPU = 1
# Number of random search experiments to run.
NUM_SAMPLES = 20
