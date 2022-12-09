# Concrete Corrision
These models predict corrosion-induced cracking using PyTorch. The models takes in corrosion patterns and concrete material properties and outputs whether there is a crack on the concrete’s surface.

# Dependencies
numpy==1.21.5
ray==2.1.0
scikit_learn==1.2.0
scipy==1.9.1
torch==1.13.0

# Contents
`data_analysis/`: Notebook for analyzing corrosion and concrete input features.
`data_augmentation/`: Generates augmented training data by flipping, adding noise, and monotonic scaling.
`data_generation/`: FEM simulation for generating data.
`data_loader/`: Helper functions for loading datasets.
`data_normalization/`: Custom normalization for corrosion depth features.
`hyperparameter_search/`: RayTune for random hyperparameter search.
`model_evaluation/`: Helper function for evaluating trained models on test set. Also contains a notebook for more detailed evaluation metrics and plots.
`models/`: Directory containing several model architecture configurations.
`sample_data_11_09_2022/`: Sample corrosion patterns from COMSOL.
`training_loss_util/`: Helper function for computing loss.

# Usage
1. Generate zipped corrosion data.
```
matlab -nodesktop -r generateCrackData.m
```
2. Extract, preprocess, and join data:
```
python3 data_preprocessing/preprocess_data.py --output_path=/path/to/data --extract
```
3. Optionally normalize data:
```
python3 data_normalization/data_normalization.py --training_data_dir=/path/to/data
```
4. Train a model:
```
python3 models/baseline_model/baseline.py --batch_size=128 --num_epochs=10000 --data_dir=/path/to/data/ --output_path=/path/to/model --print_every=10
```
5. Tune model hyperparameters:
```
python3 hyperparameter_search/hyperparameter_search.py --num_runs=100 --corrosion_path=/path/to/data --label_path=/path/to/data
```
6. Evaluate models on test set:
```
python3 model_evaluation/evaluate.py --model_name=baseline_model.py --data_dir=/path/to/data/ --model_dir=/path/to/model --normalized_data
```
