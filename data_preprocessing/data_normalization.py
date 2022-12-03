import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Normalize training features.")
parser.add_argument(
    '--training_data_dir',
    help='Path to location of corrosion_train.npy and corrorsion_test.npy')

args = parser.parse_args()


def normalize_data(training_features, test_features):
    '''
	Normalize corrosion depth and concrete property features on training set.
    Apply the same normalization to the test set. This is stable so for
    inference, this function should be called with the same training data,
    and the inference dataset.

    Args:
		training_features: Numpy array of shape (num_train_samples, 343).
            Each row contains: simulation_idx, timestep, 4 concrete property
            featurs, and 337 corrosion depth features.
		test_features: Numpy array of shape (num_test_samples, 343). Same
            as training_features.

	Returns:
		training_features_normalized: Numpy array of the same shape as
            training_features, with normalization.
		test_features_normalized: Numpy array of the same shape as
            test_features, with normalization.
    '''
    return training_features, test_features


if __name__ == "__main__":
    # Load train and test datasets
    corrosion_data_train = np.load(args.training_data_dir +
                                   '/corrosion_train.npy',
                                   allow_pickle=True)
    corrosion_data_test = np.load(args.training_data_dir +
                                  '/corrosion_test.npy',
                                  allow_pickle=True)
    normalized_data_train, normalized_data_test = normalize_data(
        corrosion_data_train, corrosion_data_test)
