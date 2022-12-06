import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append('..')
from data_loader import get_data_loader
from models.Conv2H1FC1 import Conv2H1FC1, train_epoch, validate
from training_loss_util import weighted_loss

model_fn = Conv2H1FC1


def eval_model(model, test_data_dir, normalized_data=True):
    '''
    Evaluates loss, precision, recall, f1 score, and AUC on test set.
    '''
    # Load dataset from saved npy
    if normalized_data:
        corrosion_path = test_data_dir + 'corrosion_test_normalized.npy'
    else:
        corrosion_path = test_data_dir + 'corrosion_test.npy'

    label_path = args.data_dir + 'labels_test.npy'
    test_data_np = np.load(corrosion_path, allow_pickle=True)
    test_labels_np = np.load(label_path, allow_pickle=False)

    # Create DataLoader
    test_dataloader = get_data_loader(test_data_np,
                                      test_labels_np,
                                      batch_size=None)

    # Amount to upweigh the positive samples for loss computation
    positive_samples_weight = 1

    loss, precision, recall, f1, roc_auc = validate(
        model=model,
        data_loader=test_dataloader,
        positive_samples_weight=positive_samples_weight)

    print("Test_loss:%.3f test_precision:%.3f test_recall:%.3f "
          "test_f1:%.3f test_auc:%.3f" %
          (loss, precision, recall, f1, roc_auc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        default='/home/wongjames/cs230/Project/data_12_2_2022/',
        help="Path of saved corrosion numpy array")
    parser.add_argument(
        '--model_dir',
        default='/home/wongjames/cs230/Project/saved_models_12_2_2022/',
        help="Directory of saved models")
    parser.add_argument('--model_name',
                        default='model.pt',
                        help="Name of saved model.pt")
    parser.add_argument('--normalized_data',
                        action='store_true',
                        help='True to use normalized training data')

    args = parser.parse_args()

    # Load model from saved path
    saved_model_path = args.model_dir + args.model_name
    model = model_fn()
    state = torch.load(saved_model_path)
    model.load_state_dict(state, strict=False)

    eval_model(model,
               test_data_dir=args.data_dir,
               normalized_data=args.normalized_data)
