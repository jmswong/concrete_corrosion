import argparse
import glob
import os.path
import re
import zipfile

import scipy.io

NUM_TIMESTEPS_PER_SIMULATION = 9


def get_FEM_filenames(num_simulations=1):
    '''
    Returns a list of FEM filenames for the first num_simulation runs of FEM.

    Args:
        num_simulations (int): Get filenames for this many simulations.

    Returns:
        list (str): List of output filenames containing output labels.
    '''
    filenames = []
    for simulation_idx in range(1, num_simulations + 1):
        for timestep in range(1, NUM_TIMESTEPS_PER_SIMULATION + 1):
            filename = "output_%d_%d.mat" % (simulation_idx, timestep)
            filenames.append(filename)
    return filenames


def get_FEM_filenames_from_list(simulation_timesteps):
    '''
    Returns a list of FEM filenames for the specified (simulation_idx, timestep) pairs.

    Args:
        simulation_timesteps: List of (simulation index, timestep) pairs. If
            this is specified, ignore num_simulations and extract all corrosion
            files specified by the list.

    Returns:
        list (str): List of output filenames containing output labels.
    '''
    filenames = []
    for simulation_idx, timestep in simulation_timesteps:
        filename = "output_%d_%d.mat" % (simulation_idx, timestep)
        filenames.append(filename)
    return filenames


def get_all_simulation_timesteps_from_zip(path):
    '''
    Returns a list of (simulation_idx, timestep) pairs for all FEM outputs in the given zip file.

    Args:
        path (str): Path to zipped output data.

    Returns:
        list: List of (simulation_index, timesteps) that were successfully extracted.
    '''
    output = []
    with zipfile.ZipFile(path, 'r') as zip_obj:
        file_names = zip_obj.namelist()
        for file_name in file_names:
            if "MACOSX" in file_name:
                continue
            if file_name[-1] == '/':
                continue
            file_name = file_name.split('/')[-1]
            matches = re.match(r'output_(\d+)_(\d+).mat', file_name)
            simulation_idx = int(matches.group(1))
            timestep = int(matches.group(2))
            output.append((simulation_idx, timestep))
    # returns list sorted first by increasing simulation_idx, then by increasing timestep
    return sorted(output)


def extract_FEM_output(zipped_path, num_simulations=1, extract_all=False):
    '''
    Unzip specified number of FEM output files into the same directory. Each
    simulation will have NUM_TIMESTEPS_PER_SIMULATION output files. If
    extract_all is true, ignore num_simulations and extract every file in the
    specified path.

    Args:
        zipped_path (str): Path to zipped output data.
        num_simulations (int): If set, extract all timesteps for the first
            num_simulation experiments.
        extract_all (bool): If true, extract all files in zipped_path.

    Returns:
        list: List of (simulation_index, timesteps) that were successfully extracted.
    '''
    if extract_all:
        simulation_timesteps = get_all_simulation_timesteps_from_zip(
            zipped_path)
        FEM_filenames = get_FEM_filenames_from_list(simulation_timesteps)
    else:
        FEM_filenames = get_FEM_filenames(num_simulations)

    FEM_path_lst = zipped_path.split('/')
    FEM_dir, zip_filename = '/'.join(FEM_path_lst[:-1]), FEM_path_lst[-1]

    output = []
    with zipfile.ZipFile(zipped_path, 'r') as zip_obj:
        file_names = zip_obj.namelist()
        for file_name in file_names:
            if "MACOSX" in file_name:
                continue

            file_name = file_name.split('/')[-1]
            if file_name not in FEM_filenames:
                continue

            full_file_name = zip_filename.split(".")[0] + "/" + file_name
            # print("extracting " + full_file_name + " to " + FEM_dir)
            zip_obj.extract(full_file_name, FEM_dir)

            # parse for simulation_idx and timestep
            matches = re.match(r'output_(\d+)_(\d+).mat', file_name)
            simulation_idx = int(matches.group(1))
            timestep = int(matches.group(2))

            output.append((simulation_idx, timestep))

    # returns list sorted first by increasing simulation_idx, then by increasing timestep
    return sorted(output)


def extract_concrete_outputs_from_filepath(filepath):
    '''
    Extract concrete properties and surface cracking target label from specified
    output file. Returns a dictionary of concrete statistics and the target label.
    
    Args:
        filepath (str): Path to single output file.

    Returns:
        dict: dictionary of concrete statistics and the target label.
    '''
    mat = scipy.io.loadmat(filepath)
    rebar = mat['rebar'][0][0]
    cover = mat['cover'][0][0]
    tensile_srength = mat['tensile_strength'][0][0]
    w_c = mat['w_c'][0][0]
    theta = mat['theta'][0][0]
    z = mat['z'][0][0]
    label = mat['ind'][0][0]
    height_override = mat['exp_corr_layer'][0][
        0] if 'exp_corr_layer' in mat else None
    return {
        'rebar': rebar,
        'cover': cover,
        'tensile_strength': tensile_srength,
        'w_c': w_c,
        'theta': theta,
        'z': z,
        'label': label,
        'height_override': height_override,
    }


def extract_concrete_outputs(path,
                             num_simulations=1,
                             simulation_timesteps=None):
    '''
    Extract concrete output files. Returns a list of pairs, each
    containing the output filename and the concrete output dictionary.

    Args:
        path (str): Path to zipped output data.
        num_simulations (int): If set, extract all timesteps for the first
            num_simulation experiments.
        simulation_timesteps: List of (simulation index, timestep) pairs. If
            this is specified, ignore num_simulations and extract all corrosion
            files specified by the list.

    Returns:
        list (tuple): List of (filename, concrete output) pairs.
    '''
    if simulation_timesteps is not None:
        FEM_filenames = get_FEM_filenames_from_list(simulation_timesteps)
    else:
        FEM_filenames = get_FEM_filenames(num_simulations)

    file_and_outputs = []
    for filename in FEM_filenames:
        filepath = path + '/Data_outputs/' + filename
        if not os.path.isfile(filepath):
            print("skipping %s since output file does not exist!" % filepath)
            continue
        concrete_outputs = extract_concrete_outputs_from_filepath(filepath)
        file_and_outputs.append((filepath, concrete_outputs))
    return file_and_outputs
