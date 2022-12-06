"""
data_augmentation.py: Augments generated and processed training data.
"""
import argparse
import copy
import random

import numpy as np


def flip(row):
    """
    Flips the corrosion pattern of a given training data row. In one row, the
    corrosion pattern is elements 6 - 343.

    Args:
        row: dim (343, ) row from training data.

    Returns:
        new_row: dim (343, ) np array, with corrosion depths flipped.
    """
    return np.concatenate([row[:6], np.flip(row[6:])])


def gaussian_noise(row, std_dev=0.01):
    """
    Adds random gaussian noise to all input features, with mean 0 and specified
    standard deviation.

    Args:
        row: dim (343, ) row from training data.
        std_dev (float): Standard deviation for gaussian noise.

    Returns:
        new_row: dim (343, ) np array, with added noise.
    """
    # Construct a gaussian noise vector
    noise = np.random.normal(loc=0.0, scale=std_dev, size=row.shape[0] - 2)

    # Pad 0s for the first 2 elements, which should not be augmented
    noise = np.concatenate([np.zeros(2), noise])

    # Add noise to row
    return row + noise


def monotonic_scaling(row, label, amount):
    """
    Scales corrosion depth inputs by amount, in the direction based on label.
    If label is False (no crack), then decrease the corrosion depths.
    If label is True (crack), then increase the corrosion depths.

    Args:
        row: dim (343, ) row from training data.
        label (int): 0 or 1, label for surface cracking.
        amount (float): Increase depths by this percent for positive samples.

    Returns:
        new_row: dim (343, ) np array, with corrosion depth scaling.
    """
    assert 0 < amount < 1, "Monotonic Scaling Factor must be between 0 and 1"
    if label == 1:
        scaling_factor = 1 + amount
    elif label == 0:
        scaling_factor = 1 - amount
    else:
        assert False, "Label %d is not 0 or 1" % label

    scaling_vector = np.full(row.shape[0], scaling_factor)

    # Only scale corrosion
    scaling_vector[0:6] = 1.0

    return row * scaling_vector


def augment_data(data_dir,
                 normalized,
                 num_samples,
                 monotonic_scaling_amount=0.1):
    """
    Loads input data, runs num_samples data augmentation, and saves output.

    Args:
        data_dir: dim (num_samples, 343) array of training data.
        normalized (bool): True to load and save as normalized data.
        num_samples (int): Number of new data points to generate.
        monotonic_scaling_amount (float): For monotonic scaling only, how much
            to scale the corrosion depths up/down.

    Returns:
        None
    """
    # Load existing processed data
    if normalized:
        corrosion_filepath = data_dir + "corrosion_train_normalized.npy"
    else:
        corrosion_filepath = data_dir + "corrosion_train.npy"
    label_filepath = data_dir + "labels_train.npy"

    train_data = np.load(corrosion_filepath, allow_pickle=True)
    labels = np.load(label_filepath, allow_pickle=False)

    original_train_shape = train_data.shape
    original_labels_shape = labels.shape

    flip_counter = 0
    gaussian_noise_counter = 0
    monotonic_scaling_positive_counter = 0
    monotonic_scaling_negative_counter = 0

    for i in range(num_samples):
        # choose a random row
        row_idx = random.randint(0, original_train_shape[0])
        row = train_data[row_idx, :]
        # Copy the label
        label = labels[row_idx]
        labels = np.append(labels, label)

        # choose a type of augmentation
        augmentation_type = random.choices(
            ["flip", "gaussian_noise", "monotonic_scaling"], weights=[1, 1, 1])
        augmented_row = None
        if augmentation_type[0] == "flip":
            augmented_row = flip(row)
            flip_counter += 1
        elif augmentation_type[0] == "gaussian_noise":
            augmented_row = gaussian_noise(row)
            gaussian_noise_counter += 1
        elif augmentation_type[0] == "monotonic_scaling":
            augmented_row = monotonic_scaling(row, label,
                                              monotonic_scaling_amount)
            if label == 1:
                monotonic_scaling_positive_counter += 1
            else:
                monotonic_scaling_negative_counter += 1

        assert augmented_row is not None, "Failed to handle augmentation type %s" % augmentation_type

        # add to training data
        train_data = np.vstack([train_data, copy.deepcopy(augmented_row)])

    print("Data Augmenting %d samples" % num_samples)
    print("  Flipped corrosion depths: %d" % flip_counter)
    print("  Added Gaussian Noise: %d" % gaussian_noise_counter)
    print("  Scaled Increase corrosion depth: %d" %
          monotonic_scaling_positive_counter)
    print("  Scaled Decrease corrosion depth: %d" %
          monotonic_scaling_negative_counter)
    print("Input data size: " + str(original_train_shape))
    print("Output data size: " + str(train_data.shape))
    print("Input label size: " + str(original_labels_shape))
    print("Output labels size: " + str(labels.shape))

    # Save data
    if normalized:
        output_corrosion_filepath = data_dir + "corrosion_train_normalized_augmented%d.npy" % num_samples
    else:
        output_corrosion_filepath = data_dir + "corrosion_train_augmented%d.npy" % num_samples
    output_label_filepath = data_dir + "labels_train_augmented%d.npy" % num_samples

    with open(output_corrosion_filepath, "wb") as f:
        np.save(f, train_data)
    with open(output_label_filepath, "wb") as f:
        np.save(f, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        default='/home/wongjames/cs230/Project/data_12_2_2022/',
        help="Path of saved corrosion numpy array")
    parser.add_argument('--normalized_data',
                        action='store_true',
                        default=True,
                        help='True to use normalized training data')
    parser.add_argument('--num_samples',
                        type=int,
                        default=10000,
                        help='Number of augmented samples to add')

    args = parser.parse_args()

    augment_data(args.data_dir, args.normalized_data, args.num_samples)
