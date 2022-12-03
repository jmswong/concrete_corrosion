"""Baseline Convolution+FullyConnected model"""

import argparse
import copy
import random
import torch

import numpy as np

import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.model_selection import train_test_split


class Data(Dataset):
    '''
    Data class for corrosion data, concrete properties, and target label
    '''
    def __init__(self, corrosion_data, target_labels):
        '''
        The first 2 columns are the simulation_idx and timestep respectively,
        and should not be used in training. Columns 3, 4, 5, and 6 are
        floating-point representations of certain concrete properties
        (in particular rebar, cover, tensile_strength, w_c, respectively).
        Columns 7+ are the corrosion depths along the rebar.
        '''
        self.corrosion_inputs = torch.from_numpy(corrosion_data[:, 6:].astype(
            np.float32))
        self.concrete_inputs = torch.from_numpy(corrosion_data[:, 2:6].astype(
            np.float32))
        self.target_labels = torch.from_numpy(target_labels.astype(np.float32))
        self.len = self.corrosion_inputs.shape[0]

    def __getitem__(self, index):
        return self.corrosion_inputs[index], self.concrete_inputs[
            index], self.target_labels[index]

    def __len__(self):
        return self.len


class CNN1FC1(nn.Module):
    '''
    Baseline Convolution + FC model.
    This model runs corrosion depths through a single-layer 1d convolution with
    one kernel of size 20, followed by ReLU and max-pooling with stride 16.
    Then the output is concatenated with the 4 continuous concrete-property
    features, and fed into a fully connected layer, with sigmoid activation.
    '''
    def __init__(self):
        super(CNN1FC1, self).__init__()
        # 1 input image channel, 1 output channel, 20x1 convolution kernel
        self.conv1 = nn.Conv1d(1, 1, 20)

        # fully connected layer, single output node
        self.fc1 = nn.Linear(in_features=23, out_features=1, bias=True)

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

        # Input: batch x 1 x 337
        # Output: batch x 1 x 318
        x = self.conv1(corrosion_depths)

        x = torch.nn.ReLU()(x)

        # Input: (batch x 1 x 318)
        # Output: (batch x 1 x 19)
        x = torch.nn.MaxPool1d(kernel_size=16, stride=16)(x)

        # Flatten to (batch x 19)
        x = torch.flatten(x, start_dim=1, end_dim=2)

        # Concat (batch x 19) with (batch x 4) -> (batch x 23)
        x = torch.concat([x, concrete_features], dim=1)  #

        # fully connected layer
        # Input: (batch x 23)
        # Output: (batch x 1)
        x = self.fc1(x)

        return torch.sigmoid(x)


def compute_weighted_loss(pred, y, loss_fn, pos_weight=1):
    '''
    Computes a weighted version of a given binary loss_fn, weighing the
    positive class by pos_weight (sometimes referred to as alpha).

    weighted_loss = pos_weight * loss if y == 1 else loss

    Args:
        pred (tensor): Dim (num_samples, 1) tensor of predictions.
        y (tensor): Dim (num_samples) tensor of true labels.
        loss_fn (lambda): Binary loss function (e.g. BCELoss).
        pos_weight: Weight applied to positive samples.

    Returns:
        avg_loss (tensor): Scalar, average weighted loss.
    '''
    y = y.unsqueeze(-1)
    loss = loss_fn(pred, y)

    # This constructs a weight vector where the element is pos_weight when
    # y == 1 and 1 otherwise.
    weights = (y * (pos_weight - 1) + 1)

    avg_loss = torch.sum(loss * weights) / sum(weights)
    return avg_loss


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
        avg_loss = compute_weighted_loss(predictions, y, loss_fn,
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
        avg_loss = compute_weighted_loss(predictions, y, loss_fn,
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


def main():
    '''
    Load training data, train model, save model outuputs.
    '''
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
    parser.add_argument(
        '--output_path',
        default='/home/wongjames/cs230/Project/models/baseline_model.pt',
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
                        default=0.01,
                        help="Learning rate")
    parser.add_argument('--weight_decay',
                        type=float,
                        default=1e-4,
                        help="Weight decay amount")
    parser.add_argument(
        '--print_every',
        type=int,
        default=100,
        help="Print training and validation loss every this many epochs.")

    args = parser.parse_args()

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
                                  batch_size=args.batch_size,
                                  shuffle=True)

    # Create single-batch validation data
    val_data = Data(X_val, y_val)
    val_dataloader = DataLoader(dataset=val_data,
                                batch_size=X_val.shape[0],
                                shuffle=True)

    model = CNN1FC1()

    # Define optimizer
    assert args.optimizer in ["Adam", "RMSprop", "SGD"]
    if args.optimizer == "Adam":
        torch_optimizer = torch.optim.Adam(model.parameters(),
                                           lr=args.learning_rate,
                                           weight_decay=args.weight_decay)
    elif args.optimizer == "RMSprop":
        torch_optimizer = torch.topim.RMSprop(model.parameters(),
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
        val_loss, val_precision, val_recall, val_f1, val_roc_auc = validate(
            model=model,
            data_loader=val_dataloader,
            positive_samples_weight=positive_samples_weight)

        if args.print_every is not None and epoch % args.print_every == 0:
            print("Epoch %4d- train_loss:%.3f val_loss:%.3f train_f1:%.3f "
                  "val_f1:%.3f train_roc_auc:%.3f val_roc_auc:%.3f" %
                  (epoch, train_loss, val_loss, train_f1, val_f1,
                   train_roc_auc, val_roc_auc))

    # Save trained model
    torch.save(model.state_dict(), args.output_path)


if __name__ == '__main__':
    main()
