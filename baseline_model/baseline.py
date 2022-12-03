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
parser.add_argument('--batch_size',
                    type=int,
                    default=1,
                    help="Batch size to use for training")
parser.add_argument('--num_epochs',
                    type=int,
                    default=100,
                    help="Number of training epochs")
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.01,
                    help="Learning rate")

args = parser.parse_args()


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


def train(X_train,
          y_train,
          X_val=None,
          y_val=None,
          batch_size=1,
          optimizer='Adam',
          learning_rate=0.01,
          weight_decay=1e-4,
          positive_samples_weight=1,
          num_epochs=100,
          print_every=None):
    # Instantiate training and test(validation) data
    train_data = Data(X_train, y_train)
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True)

    # If validation data is specified, create single-batch validation data
    val_input1 = val_input2 = val_y = None
    if X_val is not None:
        val_data = Data(X_val, y_val)
        val_dataloader = DataLoader(dataset=val_data,
                                    batch_size=X_val.shape[0],
                                    shuffle=True)
        val_input1, val_input2, val_y = list(val_dataloader)[0]

    model = CNN1FC1()

    loss_fn = nn.BCELoss(reduction='none')

    # Define optimizer
    assert optimizer in ["Adam", "RMSprop", "SGD"]
    if optimizer == "Adam":
        torch_optimizer = torch.optim.Adam(model.parameters(),
                                           lr=learning_rate,
                                           weight_decay=weight_decay)
    elif optimizer == "RMSprop":
        torch_optimizer = torch.topim.RMSprop(model.parameters(),
                                              lr=learning_rate,
                                              momentum=0,
                                              alpha=0.99,
                                              eps=1e-8,
                                              weight_decay=weight_decay)
    elif optimizer == "SGD":
        torch_optimizer = torch.optim.SGD(model.parameters(),
                                          lr=learning_rate,
                                          momentum=0,
                                          weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for input1, input2, y in train_dataloader:
            # zero the parameter gradients
            torch_optimizer.zero_grad()

            # forward prop
            predictions = model(input1, input2)

            # compute weighted loss
            avg_loss = compute_weighted_loss(predictions, y, loss_fn,
                                             positive_samples_weight)

            # also compute validation loss
            validation_predictions = model(val_input1, val_input2)
            validation_avg_loss = compute_weighted_loss(
                validation_predictions, val_y, loss_fn,
                positive_samples_weight)

            if print_every is not None and epoch % print_every == 0:
                print(f"Epoch %4d- Training Loss:%.5f   Validation Loss:%.5f" % \
                      (epoch, avg_loss, validation_avg_loss))

            # compute gradients and update parameters
            avg_loss.backward()
            torch_optimizer.step()

    return model


if __name__ == '__main__':
    # Load dataset from saved npy
    corrosion_data = np.load(args.corrosion_path, allow_pickle=True)
    target_data = np.load(args.label_path, allow_pickle=False)

    # Split to 80%/20% train/test sets
    random_state = 42
    X_train, X_val, y_train, y_val = train_test_split(
        corrosion_data, target_data, test_size=0.2, random_state=random_state)

    model = train(X_train=X_train,
                  y_train=y_train,
                  X_val=X_val,
                  y_val=y_val,
                  batch_size=args.batch_size,
                  optimizer="Adam",
                  learning_rate=args.learning_rate,
                  positive_samples_weight=1,
                  num_epochs=args.num_epochs,
                  print_every=100)

    # Save trained model
    torch.save(model.state_dict(), args.output_path)
