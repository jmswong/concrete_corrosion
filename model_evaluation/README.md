# Dependencies
- Pytorch
- Numpy

# Model Evaluation
Evaluates loss, precision, recall, f1 score, and AUC on test set.

Args:
- normalized_data: True to test on normalized data

## Example Usage
```
python3 evaluate.py --model_name=baseline_model.py --data_dir=/path/to/data/ --model_dir=/path/to/model --normalized_data
```
