import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
#import torch.optim as optim
#from sklearn.model_selection import train_test_split
import argparse
#import os

import sys

sys.path.append('..')
from baseline_model import Conv1FC1
from baseline_model import train_epoch
from baseline_model import validate
from baseline_model import Data

model_fn = Conv1FC1


def eval_model():
    '''
    Evaluates loss, precision, recall, f1 score, and AUC on test set.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        default='/home/wongjames/cs230/Project/data_12_2_2022/',
        help="Path of saved corrosion numpy array")
    parser.add_argument('--normalized_data',
                        action='store_true',
                        help='True to use normalized training data')
    parser.add_argument(
        '--model_dir',
        default='/home/wongjames/cs230/Project/saved_models_12_2_2022/',
        help="Directory of saved models")
    parser.add_argument('--model_name',
                        default='model.pt',
                        help="Name of saved model.pt")

    args = parser.parse_args()

    # Load dataset from saved npy
    if args.normalized_data:
        corrosion_path = args.data_dir + 'corrosion_test_normalized.npy'
    else:
        corrosion_path = args.data_dir + 'corrosion_test.npy'
    label_path = args.data_dir + 'labels_test.npy'
    test_data_np = np.load(corrosion_path, allow_pickle=True)
    test_labels_np = np.load(label_path, allow_pickle=False)

    # Create DataLoader
    test_data = Data(test_data_np, test_labels_np)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=test_data_np.shape[0])

    # Amount to upweigh the positive samples for loss computation
    positive_samples_weight = 1

    # Load model from saved path
    saved_model_path = args.model_dir + args.model_name
    model = model_fn()
    model.load_state_dict(torch.load(saved_model_path))

    loss, precision, recall, f1, roc_auc = validate(
        model=model,
        data_loader=test_dataloader,
        positive_samples_weight=positive_samples_weight)

    print("Test_loss:%.3f test_precision:%.3f test_recall:%.3f "
          "test_f1:%.3f test_auc:%.3f" %
          (loss, precision, recall, f1, roc_auc))


if __name__ == "__main__":
    eval_model()
