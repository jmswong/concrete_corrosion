import argparse
import glob
import os.path
import re
import scipy.io
import zipfile


NUM_TIMESTEPS_PER_SIMULATION = 9

# Returns a list of FEM filenames for the first num_simulation runs of FEM.
def get_FEM_filenames(num_simulations = 1):
  filenames = []
  for simulation_idx in range(1, num_simulations + 1):
    for timestep in range(1, NUM_TIMESTEPS_PER_SIMULATION + 1):
      filename = "output_%d_%d.mat" % (simulation_idx, timestep)
      filenames.append(filename)
  return filenames

# Unzip specified number of FEM output files into the same directory. Each
# simulation will have NUM_TIMESTEPS_PER_SIMULATION output files.
def extract_FEM_output(zipped_path, num_simulations = 1):
  FEM_filenames = get_FEM_filenames(num_simulations)

  FEM_path_lst = zipped_path.split('/')
  FEM_dir, zip_filename = '/'.join(FEM_path_lst[:-1]), FEM_path_lst[-1]
  
  with zipfile.ZipFile(zipped_path, 'r') as zip_obj:
    file_names = zip_obj.namelist()
    for file_name in file_names:
      if "MACOSX" in file_name:
        continue
      file_name = file_name.split('/')[-1]
      if file_name not in FEM_filenames:
        continue
      full_file_name = zip_filename.split(".")[0] + "/" + file_name
      print("extracting " + full_file_name + " to " + FEM_dir)
      zip_obj.extract(full_file_name, FEM_dir)

# Extract concrete properties and surface cracking target label from specified
# output file. Returns a dictionary of concrete statistics and the target label.
def extract_concrete_outputs_from_filepath(filepath):
  mat = scipy.io.loadmat(filepath)
  rebar = mat['rebar'][0][0]
  cover = mat['cover'][0][0]
  tensile_srength = mat['tensile_strength'][0][0]
  w_c = mat['w_c'][0][0]
  theta = mat['theta'][0][0]
  z = mat['z'][0][0]
  label = mat['ind'][0][0]
  height_override = mat['exp_corr_layer'][0][0] if 'exp_corr_layer' in mat else None
  return {
      'rebar' : rebar,
      'cover' : cover,
      'tensile_strength' : tensile_srength,
      'w_c' : w_c,
      'theta' : theta,
      'z' : z,
      'label' : label,
      'height_override' : height_override,
  }

# Extract num_simulation number of output files. Returns a list of pairs, each
# containing the output filename and the concrete output dictionary.
def extract_concrete_outputs(path, num_simulations = 1):
  FEM_filenames = get_FEM_filenames(num_simulations)

  file_and_outputs = []
  for filename in get_FEM_filenames(num_simulations):
    filepath = path + '/Data_outputs/' + filename
    if not os.path.isfile(filepath):
      print("skipping %s since output file does not exist!" % filepath)
      continue
    concrete_outputs = extract_concrete_outputs_from_filepath(filepath)
    file_and_outputs.append((filepath, concrete_outputs))
  return file_and_outputs
