# Dependencies
- PyTorch
- RayTune
- Numpy
- Sklearn

# Hyperparameter Tuning
This executes parallelized random search using RayTune over parameters specified by `config.py`.  Config specifies the following search parameters:
- Epochs: Number of training epochs per model
- Valid_split: Percent of dataset to hold out for validation
- Scheduler: Either "ASHA" or "PBT"
-  Max_num_epochs: Max epochs for ASHA.

Validation loss, precision, recall, f1 score, and AUC metrics are reported for each model.  F1 score is used as the optimization objective.

## Example Usage
First you need to modify `os.environ["PYTHONPATH"]` in `hyperparameter_search.py` to point to your repo home directory. This is required as RayTune does not pass env variables to downstream worker jobs. Then run:
```
python3 hyperparameter_search.py --num_runs=100 --corrosion_path=/path/to/corrosion.npy --label_path=/path/to/target_labels.npy 
```
