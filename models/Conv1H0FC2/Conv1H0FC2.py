"""Conv1H0FC2 Concrete cracking classification model"""

import argparse
import copy
import math
import random
import sys

import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

sys.path.append('../..')
from data_loader import get_data_loader
from training_loss_util import weighted_loss

CORROSION_DEPTH_SIZE = 337


class Conv1H0FC2(nn.Module):
    '''
    Convolution Layers on corrosion depths: 1
    Hidden fully connected layers on concrete properties: 0
    Outut fully connected layers on concatenated features: 2
    '''
    def __init__(self,
                 conv_kernel_sizes=[8],
                 pooling_strides=[8],
                 hidden_layer_sizes=[],
                 output_layer_sizes=[16, 1]):
        '''
        Args:
            conv_kernel_sizes (list): Kernel sizes for corrosion depths convolution
            pooling_strides (list): Strides for MaxPool. Pooling uses the same
                kernel size as convolution.
            hidden_layer_sizes (list): Layer sizes for fully connected layers
                on concrete property input.
            output_layer_sizes (list): Must be 1 for single output hidden layer.
        '''
        super(Conv1H0FC2, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.max_pool_layers = nn.ModuleList()
        self.output_fc_layers = nn.ModuleList()

        # 1 input image channel, 1 output channel, kernel_size x 1 convolution kernel
        self.conv1_output_size = math.floor((CORROSION_DEPTH_SIZE -
                                             (conv_kernel_sizes[0] - 1) - 1) /
                                            1 + 1)
        conv1 = nn.Conv1d(in_channels=1,
                          out_channels=1,
                          kernel_size=conv_kernel_sizes[0],
                          stride=1)
        self.conv_layers.append(conv1)

        # 1 max pooling layer
        pool_kernel_size = conv_kernel_sizes[0]
        pool_stride = pooling_strides[0]
        self.pool1_output_size = math.floor((self.conv1_output_size -
                                             (pool_kernel_size - 1) - 1) /
                                            pool_stride + 1)
        pool1 = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride)

        self.max_pool_layers.append(pool1)

        # 2 output fully connected layers, single output node
        output_fc1 = nn.Linear(in_features=self.pool1_output_size + 4,
                               out_features=output_layer_sizes[0],
                               bias=True)
        self.output_fc_layers.append(output_fc1)

        output_fc2 = nn.Linear(in_features=output_layer_sizes[0],
                               out_features=output_layer_sizes[1],
                               bias=True)
        self.output_fc_layers.append(output_fc2)

        self.print_model_architecture()

    def print_model_architecture(self):
        pass

    def forward(self, corrosion_depths, concrete_features):
        '''
        Forward Prop
        Inputs:
            corrosion_depths: tensor of dim (batch_size x 337)
            concrete_features: tensor of dim (batch_size x 4)
        Output:
            predictions: tensor of dim (batch_size x 1)
        '''
        corrosion_depths = torch.unsqueeze(corrosion_depths, 1)

        # Convolution, ReLU, MaxPool
        corrosion_x = self.conv_layers[0](corrosion_depths)
        corrosion_x = torch.nn.ReLU()(corrosion_x)
        corrosion_x = self.max_pool_layers[0](corrosion_x)

        # Flatten to (batch x 19)
        corrosion_x = torch.flatten(corrosion_x, start_dim=1, end_dim=2)

        # Concatenate
        concatenated_x = torch.concat([corrosion_x, concrete_features], dim=1)

        # fully connected layers
        x = self.output_fc_layers[0](concatenated_x)
        x = self.output_fc_layers[1](x)

        return torch.sigmoid(x)


def validate(model, data_loader, positive_samples_weight=1):
    '''
    Validates a given model on a dataset.

    Args:
        model: Torch model.
        data_loader: DataLoader with validation data.
        positive_samples_weight: Weight applied to positive samples.

    Returns: Evaluation metrics averaged over all batches in validation.
        avg_loss (float)
        avg_precision (float)
        avg_recall (float)
        avg_f1 (float)
        avg_roc_auc (float)
    '''
    model.eval()

    loss_fn = nn.BCELoss(reduction='none')

    counter = 0
    total_loss = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_roc_auc = 0
    for input1, input2, y in data_loader:
        counter += 1

        # forward prop
        predictions = model(input1, input2)

        # compute weighted loss
        avg_loss = weighted_loss(predictions, y, loss_fn,
                                 positive_samples_weight)

        # compute evaluation metrics on training set
        binary_preds = predictions > 0.5
        precision_score = sklearn.metrics.precision_score(y, binary_preds)
        recall_score = sklearn.metrics.recall_score(y, binary_preds)
        f1_score = sklearn.metrics.f1_score(y, binary_preds)
        # ROC_AUC is not defined if only 1 class exists
        if (len(set(y.tolist()))) > 1:
            roc_auc = sklearn.metrics.roc_auc_score(
                y,
                predictions.detach().numpy())
        else:
            roc_auc = 0

        # Add metrics to totals
        total_loss += avg_loss
        total_precision += precision_score
        total_recall += recall_score
        total_f1 += f1_score
        total_roc_auc += roc_auc

    avg_loss = total_loss / counter
    avg_precision = total_precision / counter
    avg_recall = total_recall / counter
    avg_f1 = total_f1 / counter
    avg_roc_auc = total_roc_auc / counter

    return avg_loss, avg_precision, avg_recall, avg_f1, avg_roc_auc


def train_epoch(model, data_loader, optimizer, positive_samples_weight=1):
    '''
    Trains a single epoch over the training data.

    Args:
        model: Torch model to train.
        data_loader: DataLoader with input training data.
        optimizer: Torch otpimizer. Should be one of {"Adam", "RMSprop", "SGD"}.
        positive_samples_weight: Weight applied to positive samples.

    Returns: Evaluation metrics averaged over all batches of training.
        avg_loss (float)
        avg_precision (float)
        avg_recall (float)
        avg_f1 (float)
        avg_roc_auc (float)
    '''
    model.train()

    loss_fn = nn.BCELoss(reduction='none')

    counter = 0
    total_loss = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_roc_auc = 0
    for input1, input2, y in data_loader:
        counter += 1
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward prop
        predictions = model(input1, input2)

        # compute weighted loss
        avg_loss = weighted_loss(predictions, y, loss_fn,
                                 positive_samples_weight)

        # compute evaluation metrics on training set
        binary_preds = predictions > 0.5
        precision_score = sklearn.metrics.precision_score(y, binary_preds)
        recall_score = sklearn.metrics.recall_score(y, binary_preds)
        f1_score = sklearn.metrics.f1_score(y, binary_preds)
        # ROC_AUC is not defined if only 1 class exists
        if (len(set(y.tolist()))) > 1:
            roc_auc = sklearn.metrics.roc_auc_score(
                y,
                predictions.detach().numpy())
        else:
            roc_auc = 0

        # Add metrics to totals
        total_loss += avg_loss
        total_precision += precision_score
        total_recall += recall_score
        total_f1 += f1_score
        total_roc_auc += roc_auc

        # compute gradients and update parameters
        avg_loss.backward()
        optimizer.step()

    avg_loss = total_loss / counter
    avg_precision = total_precision / counter
    avg_recall = total_recall / counter
    avg_f1 = total_f1 / counter
    avg_roc_auc = total_roc_auc / counter

    return avg_loss, avg_precision, avg_recall, avg_f1, avg_roc_auc


def train_and_test():
    '''
    Load training data, train model, save model outuputs.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        default='/home/wongjames/cs230/Project/data_12_2_2022/',
        help="Path of saved corrosion numpy array")
    parser.add_argument(
        '--output_path',
        default=
        '/home/wongjames/cs230/Project/saved_models_12_2_2022/Conv1H0FC2.pt',
        help="Path to save trained pytorch model state")
    parser.add_argument(
        '--optimizer',
        default='Adam',
        help="Optimization algorithm. One of {'Adam', 'RMSprop', 'SGD'}")
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help="Batch size to use for training")
    parser.add_argument('--num_epochs',
                        type=int,
                        default=100,
                        help="Number of training epochs")
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.013,
                        help="Learning rate")
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.008,
                        help="Weight decay amount")
    parser.add_argument(
        '--print_every',
        type=int,
        default=100,
        help="Print training and validation loss every this many epochs.")
    parser.add_argument(
        '--validate',
        action='store_true',
        help='True to split dataset and compute validation metrics')
    parser.add_argument('--normalized_data',
                        action='store_true',
                        help='True to use normalized training data')
    parser.add_argument(
        '--max_training_data_size',
        type=int,
        default=1000000,
        help="Truncate the training data if it is larger than this")

    args = parser.parse_args()

    # Load dataset from saved npy
    if args.normalized_data:
        corrosion_path = args.data_dir + 'corrosion_train_normalized.npy'
    else:
        corrosion_path = args.data_dir + 'corrosion_train.npy'
    label_path = args.data_dir + 'labels_train.npy'
    corrosion_data = np.load(corrosion_path, allow_pickle=True)
    target_data = np.load(label_path, allow_pickle=False)

    # Split to 80%/20% train/test sets
    if args.validate:
        random_state = 42
        X_train, X_val, y_train, y_val = train_test_split(
            corrosion_data,
            target_data,
            test_size=0.2,
            random_state=random_state)
    else:
        X_train = corrosion_data
        y_train = target_data

    # Maybe truncate data
    if args.max_training_data_size < corrosion_data.shape[0]:
        X_train = X_train[:args.max_training_data_size]
        y_train = y_train[:args.max_training_data_size]

    print("Training data size: %d" % X_train.shape[0])

    train_dataloader = get_data_loader(X_train,
                                       y_train,
                                       batch_size=args.batch_size)

    # Create single-batch validation data
    if args.validate:
        val_dataloader = get_data_loader(X_val, y_val, batch_size=None)

    model = Conv1H0FC2()

    # Define optimizer
    assert args.optimizer in ["Adam", "RMSprop", "SGD"]
    if args.optimizer == "Adam":
        torch_optimizer = torch.optim.Adam(model.parameters(),
                                           lr=args.learning_rate,
                                           weight_decay=args.weight_decay)
    elif args.optimizer == "RMSprop":
        torch_optimizer = torch.optim.RMSprop(model.parameters(),
                                              lr=args.learning_rate,
                                              momentum=0,
                                              alpha=0.99,
                                              eps=1e-8,
                                              weight_decay=args.weight_decay)
    elif args.optimizer == "SGD":
        torch_optimizer = torch.optim.SGD(model.parameters(),
                                          lr=args.learning_rate,
                                          momentum=0,
                                          weight_decay=args.weight_decay)

    # Amount to upweigh the positive samples in training.
    positive_samples_weight = 1

    for epoch in range(args.num_epochs):
        train_loss, train_precision, train_recall, train_f1, train_roc_auc = train_epoch(
            model=model,
            data_loader=train_dataloader,
            optimizer=torch_optimizer,
            positive_samples_weight=positive_samples_weight)

        if args.validate:
            val_loss, val_precision, val_recall, val_f1, val_roc_auc = validate(
                model=model,
                data_loader=val_dataloader,
                positive_samples_weight=positive_samples_weight)

            if args.print_every is not None and epoch % args.print_every == 0:
                print("Epoch %4d- train_loss:%.3f val_loss:%.3f train_f1:%.3f "
                      "val_f1:%.3f train_roc_auc:%.3f val_roc_auc:%.3f" %
                      (epoch, train_loss, val_loss, train_f1, val_f1,
                       train_roc_auc, val_roc_auc))

        elif args.print_every is not None and epoch % args.print_every == 0:
            print("Epoch %4d- train_loss:%.3f" % (epoch, train_loss))

    # Save trained model
    torch.save(model.state_dict(), args.output_path)


if __name__ == '__main__':
    train_and_test()
