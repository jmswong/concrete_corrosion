import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Normalize training features.")
parser.add_argument(
    '--training_data_dir',
    help='Path to location of corrosion_train.npy and corrorsion_test.npy')

args = parser.parse_args()

# Number of values in a single corrosion input
CORROSION_SIZE = 337


def normalize_data(training_features, test_features):
    '''
	Normalize corrosion depth and concrete property features on training set.
    Apply the same normalization to the test set. This is stable so for
    inference, this function should be called with the same training data,
    and the inference dataset.

	The training features are normalized by subtracting the mean and dividing
    by the standard deviation. For the corrosion depth inputs, mean and stdev
    are computed across all rebar locations and all training examples (total
    of num_training_samples * 337 values). For each of the 4 concrete
    properties input features, mean and stdev are computed on that feature only
    (total of num_training_samples values).

    This produces a total of 5 mean and 5 stdev values computed on the training
    set. The same values are used to normalize the test data and for inference.

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
    params = compute_statistics(training_features)

    # Construct 1d vector of mean and standard deviations.
    # The first 2 columns on training_features should not be modified, so set
    # the mean to 0 and standard deviation to 1.
    means = [0, 0]
    stdevs = [1, 1]

    # Concrete property features.
    for i in range(4):
        means.append(params[f'mu_{i}'])
        stdevs.append(params[f'std_{i}'])

    # Corrosion depth features. The values are duplicated 337 times.
    for i in range(CORROSION_SIZE):
        means.append(params['mu_c'])
        stdevs.append(params['std_c'])

    means = np.array(means)
    stdevs = np.array(stdevs)

    # Normalize training and test sets by subtracting the mean and dividing by
    # standard deviation.from training and test features. The same vector
    # computed above can be used in both, due to numpy broadcasting.
    training_features_normalized = (training_features - means) / stdevs
    test_features_normalized = (test_features - means) / stdevs

    return training_features_normalized, test_features_normalized


def compute_statistics(features):
    '''
    Computes mean and standard deviation on input features.

    Args:
	features: Numpy array of shape (num_train_samples, 343).
            Each row contains: simulation_idx, timestep, 4 concrete property
            featurs, and 337 corrosion depth features.

    Returns:
        params (dict): Contains 10 values: mu_0 to mu_4, std_0 to std_4 are
            the mean and standard deviation of the 4 concrete properties.
            mu_c and std_c are the mean and standard deviation across all
            corrosion depth inputs.
    '''
    params = {}

    for i in range(4):
        feature = features[:, i + 2]
        mean = np.mean(feature)
        std = np.std(feature)
        params[f"mu_{i}"] = mean
        params[f"std_{i}"] = std

    concrete_features = features[:, 6:]
    params[f"mu_c"] = np.mean(concrete_features)
    params[f"std_c"] = np.std(concrete_features)

    return params


if __name__ == "__main__":
    # Load train and test datasets
    corrosion_data_train = np.load(args.training_data_dir +
                                   '/corrosion_train.npy',
                                   allow_pickle=True)
    corrosion_data_test = np.load(args.training_data_dir +
                                  '/corrosion_test.npy',
                                  allow_pickle=True)
    corrosion_train_normalized, corrosion_test_normalized = normalize_data(
        corrosion_data_train, corrosion_data_test)

    # Save normalized datasets
    with open(args.training_data_dir + "/corrosion_train_normalized.npy",
              "wb") as f:
        np.save(f, corrosion_train_normalized)
    with open(args.training_data_dir + "/corrosion_test_normalized.npy",
              "wb") as f:
        np.save(f, corrosion_test_normalized)
