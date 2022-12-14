import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from sklearn.model_selection import train_test_split

# Ray resets PYTHONPATH on distributed runs, so we need to explicitlly set the
# path here. See https://github.com/ray-project/ray/issues/5635 for details.
os.environ["PYTHONPATH"] = "/home/wongjames/concrete"

sys.path.append('..')
import models
from data_loader import get_data_loader
from models.Conv1H1FC1 import Conv1H1FC1, train_epoch, validate

model_func = models.Conv1H1FC1.Conv1H1FC1
train_epoch_func = models.Conv1H1FC1.train_epoch
validate_func = models.Conv1H1FC1.validate

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
                    default=1e-3,
                    help="Minimum Learning rate for random search")
parser.add_argument('--max_lr',
                    type=float,
                    default=1.0,
                    help="Maximum Learning rate for random search")
parser.add_argument('--min_weight_decay',
                    type=float,
                    default=0.001,
                    help="Minimum weight decay for random search")
parser.add_argument('--max_weight_decay',
                    type=float,
                    default=0.1,
                    help="Maximum weight decay for random search")
parser.add_argument('--validation_size',
                    type=float,
                    default=0.2,
                    help="Fraction of dataset to use for validation")
parser.add_argument('--cpu',
                    type=int,
                    default=8,
                    help="Number of CPUs requested")
parser.add_argument('--gpu',
                    type=int,
                    default=0,
                    help="Number of GPUs requested")
parser.add_argument('--num_epochs',
                    type=int,
                    default=3000,
                    help="Number of epochs to train each model")
parser.add_argument('--asha_max_epochs',
                    type=int,
                    default=1500,
                    help="Maximum epochs per trial before stopping")
parser.add_argument('--asha_grace_period',
                    type=int,
                    default=1000,
                    help="Don't stop any trials with fewer than this number"
                    "of epochs")
parser.add_argument(
    '--scheduler',
    default='ASHA',
    help=
    "RayTune Scheduler for parameter search. Must be either 'ASHA' or 'PBT'")

args = parser.parse_args()
assert (args.scheduler in ("ASHA", "PBT"))


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
        corrosion_data,
        target_data,
        test_size=args.validation_size,
        random_state=random_state)

    # Instantiate training and test(validation) data
    train_dataloader = get_data_loader(X_train,
                                       y_train,
                                       batch_size=train_batch_size)

    # Create single-batch validation data
    val_dataloader = get_data_loader(X_val, y_val, batch_size=None)

    return train_dataloader, val_dataloader


def train_and_validate(config):
    '''
    Create DataLoaders, instantiate model, train model. Checkpoints and saves
    models for RayTune.

    Args:
        config (dict): RayTune config with hyperparameter values
    '''
    train_dataloader, val_dataloader = get_train_and_val_dataloaders(
        config["batch_size"])

    # Initialize the model.
    model = model_func(conv_kernel_sizes=[config['kernel_size']],
                       pooling_strides=[config['pooling_stride']],
                       hidden_layer_sizes=[config['hidden_layer_size']],
                       output_layer_sizes=[1])

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
    for epoch in range(args.num_epochs):
        train_loss, train_precision, train_recall, train_f1, train_roc_auc = train_epoch(
            model=model,
            data_loader=train_dataloader,
            optimizer=torch_optimizer,
            positive_samples_weight=config["pos_weight"])
        val_loss, val_precision, val_recall, val_f1, val_roc_auc = validate(
            model=model,
            data_loader=val_dataloader,
            positive_samples_weight=config["pos_weight"])

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, 'checkpoint')
            torch.save((model.state_dict(), torch_optimizer.state_dict()),
                       path)

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
    while batch_sizes[-1] * 2 <= args.max_batch_size:
        batch_sizes.append(batch_sizes[-1] * 2)
    config = {
        "lr":
        tune.loguniform(args.min_lr, args.max_lr),
        "batch_size":
        tune.choice(batch_sizes),
        "weight_decay":
        tune.loguniform(args.min_weight_decay, args.max_weight_decay),
        "optimizer":
        tune.choice(["Adam", "RMSprop", "SGD"]),
        "pos_weight":
        tune.choice([1, 2]),
        "kernel_size":
        tune.choice([8, 16, 32]),
        "pooling_stride":
        tune.choice([4, 8, 16]),
        "hidden_layer_size":
        tune.choice([8, 16, 32]),
    }

    # Asynchronous Successive Halving Algorithm (Li et al. 2018)
    asha_scheduler = ASHAScheduler(metric="f1",
                                   mode="max",
                                   max_t=args.asha_max_epochs,
                                   grace_period=args.asha_grace_period,
                                   reduction_factor=2)

    # Population Based Training Scheduler (Jaderberg et al. 2017)
    pbt_scheduler = PopulationBasedTraining(
        time_attr='training_iteration',
        metric='f1',
        mode='max',
        perturbation_interval=100.0,
        resample_probability=0.5,
        hyperparam_mutations=config,
        synch=False,
    )

    # Reporter to show on command line/output window
    reporter = CLIReporter(metric_columns=[
        "loss", "f1", "precision", "recall", "roc_auc", "training_iteration"
    ])

    # Start run/search
    result = tune.run(train_and_validate,
                      resources_per_trial={
                          "cpu": args.cpu,
                          "gpu": args.gpu,
                      },
                      config=config,
                      num_samples=args.num_runs,
                      scheduler=(asha_scheduler if args.scheduler == "ASHA"
                                 else pbt_scheduler),
                      local_dir='../outputs/raytune_result',
                      keep_checkpoints_num=1,
                      checkpoint_score_attr='f1',
                      progress_reporter=reporter)

    # Extract the best trial run from the search.
    best_trial = result.get_best_trial('f1', 'max', 'last')

    print(f"Highest f1 score config: {best_trial.config}")
    print(f"Highest f1 score: {best_trial.last_result['f1']}")


if __name__ == '__main__':
    random_search()
