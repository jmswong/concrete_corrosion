# data_preprocessing

Processes zipped COMSOL output and target labels, outputs into numpy arrays.  Then normalize the input features.

Example Usage:
```
python3 preprocess_data.py --output_path=/home/wongjames/cs230/Project/data_12_02_2022 --extract
python3 data_normalization.py --training_data_dir=/home/wongjames/cs230/Project/data_12_2_2022
```

Args:
- output_path (str): Path to location of corrosion data. If --extract=True, the extracted files will also be extracted to here.
- num_simulations (int): Number of simulations to process.
- extract (bool): If true, unzip and extract files from zipped filepath.
- corrosion_zipped_filename (str): Name of zipped corrosion data file. This must exist in output_path.
- output_zipped_filename (str): Name of zipped output data file. This must exist in output_path.

Outputs: 
- corrosion.npy: numpy array of shape (num_samples, 343). Each row specifies the simulation_index, timestep, then 4 features of the concrete (rebar, cover, tensile_strength, w_c), then 337 floating point numbers representing concrete corrosion depths at uniformly distributed points along the rebar for that experiment.
- target_labels.npy: numpy array of shape (num_samples,). Each row is a boolean indicating whether there was a surface crack on this experiment (1 if cracked).

# Contents
- corrosion_lib.py: Helper functions for extracting and processing corrosion depth data.
- output_lib.py: Helper functions for extracting and processing FEM output data.
- preprocess_data.py: Calls corrosion_lib and output_lib. Extracts, processes, and joins corrosion and output datasets. Outputs numpy arrays.
- data_normalization.py: Normalizes data from preprocess_data.py.
