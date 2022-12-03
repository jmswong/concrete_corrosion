from config import (MAX_NUM_EPOCHS, GRACE_PERIOD, EPOCHS, CPU, GPU,
                    DATA_ROOT_DIR, NUM_WORKERS, VALID_SPLIT)
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
import argparse
import os

# Requires adding model directory to PYTHONPATH
from baseline import CNN1FC1
from baseline import train_epoch
from baseline import validate
from baseline import Data

model_func = CNN1FC1
train_epoch_func = train_epoch
validate_func = validate

parser = argparse.ArgumentParser()

parser.add_argument(
    '--corrosion_path',
    default=
    '/home/wongjames/cs230/Project/data_12_2_2022/corrosion_train_normalized.npy',
    help="Path of saved corrosion numpy array")
parser.add_argument(
    '--label_path',
    default='/home/wongjames/cs230/Project/data_12_2_2022/labels_train.npy',
    help="Path of saved target label numpy array")
parser.add_argument('--num_runs',
					type=int,
					default=10,
					help="Number of runs for random search")
parser.add_argument('--min_batch_size',
					type=int,
					default=64,
					help="Minimum batch size for random search")
parser.add_argument('--max_batch_size',
					type=int,
					default=512,
					help="Maximum batch size for random search")
parser.add_argument('--min_lr',
					type=float,
					default=1e-4,
					help="Minimum Learning rate for random search")
parser.add_argument('--max_lr',
					type=float,
					default=10,
					help="Maximum Learning rate for random search")
parser.add_argument('--min_weight_decay',
					type=float,
					default=1e-5,
					help="Minimum weight decay for random search")
parser.add_argument('--max_weight_decay',
					type=float,
					default=1e-2,
					help="Maximum weight decay for random search")

args = parser.parse_args()


def get_train_and_val_dataloaders(train_batch_size):
    '''
    Loads data from specified arg paths and create DataLoaders. Training loader
    is batched by train_batch_size. Validation loader is always single-batch.

    Args:
        train_batch_size (int): Training batch size.

    Returns:
        train_dataloader (DataLoader)
        val_dataloader (DataLoader)
    '''
    # Load dataset from saved npy
    corrosion_data = np.load(args.corrosion_path, allow_pickle=True)
    target_data = np.load(args.label_path, allow_pickle=False)

    # Split to 80%/20% train/test sets
    random_state = 42
    X_train, X_val, y_train, y_val = train_test_split(
        corrosion_data, target_data, test_size=0.2, random_state=random_state)

    # Instantiate training and test(validation) data
    train_data = Data(X_train, y_train)
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=train_batch_size,
                                  shuffle=True)

    # Create single-batch validation data
    val_data = Data(X_val, y_val)
    val_dataloader = DataLoader(dataset=val_data,
                                batch_size=X_val.shape[0],
                                shuffle=True)

    return train_dataloader, val_dataloader


def train_and_validate(config):
    train_dataloader, val_dataloader = get_train_and_val_dataloaders(
        config["batch_size"])
    # Initialize the model.
    model = model_func()
    # Define optimizer
    optimizer = config['optimizer']
    assert optimizer in ["Adam", "RMSprop", "SGD"]
    if optimizer == "Adam":
        torch_optimizer = torch.optim.Adam(model.parameters(),
                                           lr=config['lr'],
                                           weight_decay=config['weight_decay'])
    elif optimizer == "RMSprop":
        torch_optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=config['lr'],
            momentum=0,
            alpha=0.99,
            eps=1e-8,
            weight_decay=config['weight_decay'])
    elif optimizer == "SGD":
        torch_optimizer = torch.optim.SGD(model.parameters(),
                                          lr=config['lr'],
                                          momentum=0,
                                          weight_decay=config['weight_decay'])

    # Start the training.
    for epoch in range(EPOCHS):
        train_loss, train_precision, train_recall, train_f1, train_roc_auc = train_epoch(
            model=model,
            data_loader=train_dataloader,
            optimizer=torch_optimizer,
            positive_samples_weight=config["pos_weight"])
        val_loss, val_precision, val_recall, val_f1, val_roc_auc = validate(
            model=model,
            data_loader=val_dataloader,
            positive_samples_weight=config["pos_weight"])

        tune.report(loss=val_loss,
                    precision=val_precision,
                    recall=val_recall,
                    f1=val_f1,
                    roc_auc=val_roc_auc)


def random_search():
    '''
    Runs random hyperparameter search over parameters specified by config.
    '''
    # Define the parameter search configuration.
	# Create a set of batch sizes between min and max, in multiples of 2
    batch_sizes = [args.min_batch_size]
    while batch_sizes[-1]*2 <= args.max_batch_size:
        batch_sizes.append(batch_sizes[-1]*2)
    config = {
        "lr": tune.loguniform(args.min_lr, args.max_lr),
        "batch_size": tune.choice(batch_sizes),
        "weight_decay": tune.loguniform(args.min_weight_decay, args.max_weight_decay),
        "optimizer": tune.choice(["Adam", "RMSprop", "SGD"]),
        "pos_weight": tune.choice([1]),
    }

    # Schduler to stop bad performing trails.
    scheduler = ASHAScheduler(metric="f1",
                              mode="min",
                              max_t=MAX_NUM_EPOCHS,
                              grace_period=GRACE_PERIOD,
                              reduction_factor=2)

    # Reporter to show on command line/output window
    reporter = CLIReporter(metric_columns=[
        "loss", "f1", "precision", "recall", "roc_auc", "training_iteration"
    ])

    # Start run/search
    result = tune.run(train_and_validate,
                      resources_per_trial={
                          "cpu": CPU,
                          "gpu": GPU
                      },
                      config=config,
                      num_samples=args.num_runs,
                      scheduler=scheduler,
                      local_dir='../outputs/raytune_result',
                      keep_checkpoints_num=1,
                      checkpoint_score_attr='min-loss_f1_precision_recall',
                      progress_reporter=reporter)

    # Extract the best trial run from the search.
    best_trial = result.get_best_trial('f1', 'min', 'last')

    print(f"Lowest f1 score config: {best_trial.config}")
    print(f"Lowest f1 score: {best_trial.last_result['f1']}")


if __name__ == '__main__':
    random_search()
