# Concrete Corrision
Predicting crack patterns of reinforced concrete from corrosion patterns using CNNs.

# Dependencies
- PyTorch

# Contents


# Usage
1. Generate zipped corrosion data.
2. Preprocess data:
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
