import argparse
import glob
import re
import zipfile


NUM_TIMESTEPS_PER_SIMULATION = 9

# Unzip specified number of corrosion files into the same directory,
# ignoring any edge files. If extract_all is true, ignore num_simulations
# and extract every file in the specified path.
def extract_corrosion_output(path, num_simulations = 1, extract_all = False):
  import pdb; pdb.set_trace()
  if extract_all:
    corrosion_filenames = get_all_filenames_from_zip(path)
  else:
    corrosion_filenames = get_corrosion_filenames(num_simulations)

  corrosion_path_lst = path.split('/')
  corrosion_dir_base = '/'.join(corrosion_path_lst[:-1])
  corrosion_dir = corrosion_dir_base

  with zipfile.ZipFile(path, 'r') as zip_obj:
    file_names = zip_obj.namelist()
    for file_name in file_names:
      file_name = file_name.split("/")[-1]
      if file_name not in corrosion_filenames:
        continue
      full_file_name = path.split("/")[-1].split(".")[0] + "/" + file_name
      print("extracting " + full_file_name + " to " + corrosion_dir)
      zip_obj.extract(full_file_name, corrosion_dir)

# Returns a list of corrosion simulation filenames, for the first num_simulation
# datapoints. Note that the files are 1-indexed, so this returns filenames from
# simulation 1 ... (num_simulations+1). Each simulation contains 
# NUM_TIMESTEPS_PER_SIMULATION files, one per timestep.
def get_corrosion_filenames(num_simulations = 1):
  filenames = []
  # file names from COMSOL are 1-indexed
  for simulation_idx in range(1, num_simulations + 1):
    for timestep in range(1, NUM_TIMESTEPS_PER_SIMULATION + 1):
      filename = "Corrosion_simulation_%d_timeStep_%d.txt" % (simulation_idx, timestep)
      filenames.append(filename)
  return filenames

# Similar to get_corrosion_filenames, but takes in a list of
# (simulation, timestep) tuples and only extracts the specified samples.
def get_corrosion_filenames_subset(simulation_timesteps):
  filenames = []
  for simulation_idx, timestep in simulation_timesteps:
    filename = "Corrosion_simulation_%d_timeStep_%d.txt" % (simulation_idx, timestep)
    filenames.append(filename)
  return filenames

# Returns a list of corrosion simulation filenames which match any output file
# from the given corrosion.zip path.
def get_all_filenames_from_zip(path):
  filenames = []
  with zipfile.ZipFile(path, 'r') as zip_obj:
    filenames = zip_obj.namelist()

  # filter out MACOSX
  filenames = [file for file in filenames if "MACOSX" not in file]

  # filter out directories
  filenames = [file for file in filenames if file[-1] != '/']

  # extract simulation id and timestep
  simulation_timesteps = []
  for filepath in filenames:
    filename = filepath.split("/")[1]
    matches = re.match(r'output_(\d+)_(\d+).mat', filename)
    simulation_idx = int(matches.group(1))
    timestep = int(matches.group(2))
    simulation_timesteps.append((simulation_idx, timestep))

  # construct corrosion filenames
  return get_corrosion_filenames_subset(simulation_timesteps)

# Returns a list of corrosion simulation filenames, for the first num_simulation
# datapoints. Note that the files are 1-indexed, so this returns filenames from

# Returns a 1d corrosion map given a single filepath. A 1d corrosion map is
# represented as a python dictionary, with keys representing the location on the
# x-axis along a horizontal rebar, and the values representing corrosion depth
# at that point.
def extract_1d_corrosion_map_from_filepath(filepath):
  with open(filepath, 'r') as f:
    lines = f.readlines()
  corrosion = {}
  for line in lines:
    if line.startswith("%"):
      continue
    spl = re.split(r'\s+', line.strip())
    assert len(spl) == 2, spl
    rebar_location = float(spl[0])
    corrosion_depth = float(spl[1])
    corrosion[rebar_location] = corrosion_depth
  return corrosion

# Returns a list of pairs, each containing the filename of the simulation, and
# a 1-d corrosion map.
def extract_1d_corrosion_maps(output_dir, num_simulations = 1, simulation_timesteps = None):
  corrosion_dir = output_dir + "/corrosion"

  if simulation_timesteps is not None:
    corrosion_filenames = get_corrosion_filenames_subset(simulation_timesteps)
  else:
    corrosion_filenames = get_corrosion_filenames(num_simulations)

  file_and_corrosion_map = []
  for filename in corrosion_filenames:
    filepath = corrosion_dir + '/' + filename
    corrosion_map = extract_1d_corrosion_map_from_filepath(filepath)
    file_and_corrosion_map.append((filepath, corrosion_map))
  return file_and_corrosion_map

# Asserts that all corrosion simulations are sampled from the same rebar
# locations- that is, the points along the x-axis of the rebar are the same for
# all samples. If they are not, then we will need to rescale the inputs before
# training.
def verify_rebar_locations(file_and_corrosion_map):
  rebar_locations = [tuple(x[1].keys()) for x in file_and_corrosion_map]
  return all(x == rebar_locations[0] for x in rebar_locations)

# The corrosion depth data generated from COMSOL are on different scales. This
# rescales them based on a hard-coded scaling factor.
def remap_output_scales(file_and_corrosion_maps, output_maps):
  output = []
  for file_path, corrosion_map in file_and_corrosion_maps:
    file_name = file_path.split("/")[-1]
    m = re.search("Corrosion_simulation_(\d+)_timeStep_(\d+).txt", file_name)
    simulation_idx, timestep = int(m.group(1)), int(m.group(2))
    if 1 <= simulation_idx <= 5:
      scaling_factor = (10 ** 6) / 5
    elif 6 <= simulation_idx <= 15:
      scaling_factor = (10 ** 5)
    elif 16 <= simulation_idx <= 22:
      scaling_factor = (10 ** 5) / 5
    elif simulation_idx == 23 and timestep <= 4:
      scaling_factor = (10 ** 5) / 5
    else:
      # For certain simulations, the corrosion depths are stored in the output
      # file.
      replacement_corrosion_map = {}
      output_filename = "output_%d_%d.mat" % (simulation_idx, timestep)
      for output_path, output_map in output_maps:
        if output_path.split("/")[-1] == output_filename:
          assert output_map['height_override'] is not None
          corrosion_depths_from_output = list(output_map['height_override'])
          corrosion_map_points = list(corrosion_map.keys())
          for i in range(len(corrosion_map_points)):
            replacement_corrosion_map[corrosion_map_points[i]] = float(corrosion_depths_from_output[i])
      assert replacement_corrosion_map, "Failed to find matching output for corrosion file %s" % file_path
      print("Replacing corrosion map from %s with %s" % (file_path, output_path))
      output.append((file_path, replacement_corrosion_map))
    
    scaled_corrosion_map = {k : v * scaling_factor for k, v in corrosion_map.items()}
    output.append((file_path, scaled_corrosion_map))
  return output 
