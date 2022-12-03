# data_normalization

Normalizes nparray input data that was generated in data_preprocessing.

Example Usage:
```
python3 data_normalization.py --training_data_dir=/home/wongjames/cs230/Project/data_12_2_2022
```

The training features are normalized by subtracting the mean $\mu$ and dividing by the standard deviation $\sigma$. For the corrosion depth inputs, $\mu$ and $\sigma$ are computed across all rebar locations and all training examples (total of num_training_samples * 337 values). For each of the 4 concrete properties input features, $\mu$ and $\sigma$ are computed on that feature only (total of num_training_samples values).

This produces a total of 5 $\mu$ and 5 $\sigma$ values computed on the training set. The same values are used to normalize the test data and used for inference.

Args:
- training_data_dir (str): Path to location of training data. Should contain files `corrosion_train.npy` and `corrosion_test.npy`.

Outputs: 
- training_features_normalized: Numpy array of the same shape as training_features, with normalization.
- test_features_normalized: Numpy array of the same shape as test_features, with normalization.
