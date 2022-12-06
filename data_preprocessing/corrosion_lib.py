import argparse
import glob
import re
import zipfile

NUM_TIMESTEPS_PER_SIMULATION = 9


def extract_all_corrosion_output(path, verbose=False):
    """
    Unzip and extract all corrosion depth files.
    Args:
        path (str): Path to zipped corrosion data.
        verbose (bool): If true, print detailed debug.

    Returns:
        simulation_timesteps: List of (simulation index, timestep) pairs.
    """
    corrosion_path_lst = path.split('/')
    corrosion_dir_base = '/'.join(corrosion_path_lst[:-1])
    corrosion_dir = corrosion_dir_base + "_full"

    simulation_timesteps = []

    with zipfile.ZipFile(path, 'r') as zip_obj:
        file_names = zip_obj.namelist()
        for file_name in file_names:
            if "edge" in file_name:
                continue
            
            matches = re.match(r'corrosion/Corrosion_simulation_(\d+)_timeStep_(\d+).txt', file_name)
            simulation_idx = int(matches.group(1))
            timestep = int(matches.group(2))

            # skip the first 25 experiments which have a scaling issue
            if simulation_idx < 25:
                continue
            
            simulation_timesteps.append((simulation_idx, timestep))

            if verbose:
                print("extracting " + file_name + " to " + corrosion_dir)
            zip_obj.extract(file_name, corrosion_dir)
    return simulation_timesteps

def extract_corrosion_output(path,
                             num_simulations=1,
                             simulation_timesteps=None,
                             verbose=False):
    '''
    Unzip specified corrosion files into the same directory.

    Args:
        path (str): Path to zipped corrosion data.
        num_simulations (int): If set, extract all timesteps for the first
            num_simulation experiments.
        simulation_timesteps: List of (simulation index, timestep) pairs. If
            this is specified, ignore num_simulations and extract all corrosion
            files specified by the list.
        verbose (bool): If true, print detailed debug.
    '''
    if simulation_timesteps is not None:
        corrosion_filenames = get_all_filenames_from_list(simulation_timesteps)
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
            full_file_name = path.split("/")[-1].split(
                ".")[0] + "/" + file_name
            if verbose:
                print("extracting " + full_file_name + " to " + corrosion_dir)
            zip_obj.extract(full_file_name, corrosion_dir)


def get_all_filenames_from_list(simulation_timesteps):
    '''
    Constructs list of filenames given a list of simulation and timestep pars.

    Args:
        simulation_timesteps: List of (simulation index, timestep) pairs. If
            this is specified, ignore num_simulations and extract all corrosion
            files specified by the list.

    Returns:
        list (str): List of filenames for corrosion input files.
    '''
    filenames = []
    for simulation_idx, timestep in simulation_timesteps:
        filename = "Corrosion_simulation_%d_timeStep_%d.txt" % (simulation_idx,
                                                                timestep)
        filenames.append(filename)
    return filenames


def get_corrosion_filenames(num_simulations=1):
    '''
    Returns a list of corrosion simulation filenames, for the first num_simulation
    datapoints. Note that the files are 1-indexed, so this returns filenames from
    simulation 1 ... (num_simulations+1). Each simulation contains
    NUM_TIMESTEPS_PER_SIMULATION files, one per timestep.

    Args:
        num_simulations (int): If set, extract all timesteps for the first
            num_simulation experiments.

    Returns:
        list (str): List of filenames for corrosion input files.
    '''
    filenames = []
    # file names from COMSOL are 1-indexed
    for simulation_idx in range(1, num_simulations + 1):
        for timestep in range(1, NUM_TIMESTEPS_PER_SIMULATION + 1):
            filename = "Corrosion_simulation_%d_timeStep_%d.txt" % (
                simulation_idx, timestep)
            filenames.append(filename)
    return filenames


def get_corrosion_filenames_subset(simulation_timesteps):
    '''
    Similar to get_corrosion_filenames, but takes in a list of
    (simulation, timestep) tuples and only extracts the specified samples.

    Args:
        simulation_timesteps: List of (simulation index, timestep) pairs. If
            this is specified, ignore num_simulations and extract all corrosion
            files specified by the list.

    Returns:
        list (str): List of filenames for corrosion input files.
    '''
    filenames = []
    for simulation_idx, timestep in simulation_timesteps:
        filename = "Corrosion_simulation_%d_timeStep_%d.txt" % (simulation_idx,
                                                                timestep)
        filenames.append(filename)
    return filenames


def get_all_filenames_from_zip(path):
    '''
    Returns a list of corrosion simulation filenames which match any file
    from the given corrosion.zip path.
    
    Args:
        path (str): Path to zipped corrosion data.

    Returns:
        list: of tuples (simulation_index, timestep)
    '''
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


def extract_1d_corrosion_map_from_filepath(filepath):
    '''
    Returns a 1d corrosion map given a single filepath. A 1d corrosion map is
    represented as a python dictionary, with keys representing the location on the
    x-axis along a horizontal rebar, and the values representing corrosion depth
    at that point.

    Args:
        filepath (str): Path to single corrosion file.

    Returns:
        dict: Mapping from rebar location to corrosion depth.
    '''
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


def extract_1d_corrosion_maps(output_dir,
                              num_simulations=1,
                              simulation_timesteps=None):
    '''
    Returns a list of pairs, each containing the filename of the
    simulation, and a 1-d corrosion map.

    Args:
        output_dir (str): Path to unzipped corrosion data.
        num_simulations (int): If set, extract all timesteps for the first
            num_simulation experiments.
        simulation_timesteps: List of (simulation index, timestep) pairs. If
            this is specified, ignore num_simulations and extract all corrosion
            files specified by the list.
        
    Returns:
        list (tuple): List of (filename, corrosion_map), where corrosion_map
            is a dictionary mapping from points on the x-axis of a horizontal
            to corrosion depths.
    '''
    corrosion_dir = output_dir + "/corrosion"

    if simulation_timesteps is not None:
        corrosion_filenames = get_corrosion_filenames_subset(
            simulation_timesteps)
    else:
        corrosion_filenames = get_corrosion_filenames(num_simulations)

    file_and_corrosion_map = []
    for filename in corrosion_filenames:
        filepath = corrosion_dir + '/' + filename
        corrosion_map = extract_1d_corrosion_map_from_filepath(filepath)
        file_and_corrosion_map.append((filepath, corrosion_map))
    return file_and_corrosion_map


def verify_rebar_locations(file_and_corrosion_map):
    '''
    Asserts that all corrosion simulations are sampled from the same rebar
    locations- that is, the points along the x-axis of the rebar are the same for
    all samples. If they are not, then we will need to rescale the inputs before
    training.

    Args:
        file_and_corrosion_maps (list): List of (filename, corrosion_map) tuples.
            corrosion_map is a dictionary mapping from points on the x-axis of a
            horizontal rebar to corrosion depths.

    Returns:
        bool: True if all corrosion maps are sampled from the same locations.
    '''
    rebar_locations = [tuple(x[1].keys()) for x in file_and_corrosion_map]
    return all(x == rebar_locations[0] for x in rebar_locations)


def remap_output_scales(file_and_corrosion_maps, output_maps, verbose=False):
    '''
    The corrosion depth data generated from COMSOL are on different scales. This
    rescales them based on a hard-coded scaling factor.

    Args:
        file_and_corrosion_maps (list): List of (filename, corrosion_map) tuples.
            corrosion_map is a dictionary mapping from points on the x-axis of a
            horizontal rebar to corrosion depths.
        output_maps (list): List of (filename, output_map) tuples. output_map is
            a dictionary containing various properties of the output data.
        verbose (bool): If true, print detailed debug.

    Returns:
        list: New list of corrosion_map, scaled accordingly.   
    '''
    output = []
    for file_path, corrosion_map in file_and_corrosion_maps:
        file_name = file_path.split("/")[-1]
        m = re.search("Corrosion_simulation_(\d+)_timeStep_(\d+).txt",
                      file_name)
        simulation_idx, timestep = int(m.group(1)), int(m.group(2))
        scaling_factor = None
        if 1 <= simulation_idx <= 5:
            scaling_factor = (10**6) / 5
        elif 6 <= simulation_idx <= 15:
            scaling_factor = (10**5)
        elif 16 <= simulation_idx <= 22:
            scaling_factor = (10**5) / 5
        elif simulation_idx == 23 and timestep <= 4:
            scaling_factor = (10**5) / 5
        else:
            # For other simulations, the corrosion depths are stored in the output
            # file.
            replacement_corrosion_map = {}
            output_filename = "output_%d_%d.mat" % (simulation_idx, timestep)
            for output_path, output_map in output_maps:
                if output_path.split("/")[-1] == output_filename:
                    assert output_map['height_override'] is not None
                    corrosion_depths_from_output = list(
                        output_map['height_override'])
                    corrosion_map_points = list(corrosion_map.keys())
                    for i in range(len(corrosion_map_points)):
                        replacement_corrosion_map[
                            corrosion_map_points[i]] = float(
                                corrosion_depths_from_output[i][0])
            assert replacement_corrosion_map, "Failed to find matching output for corrosion file %s" % file_path
            if verbose:
                print("Replacing corrosion map from %s with %s" %
                      (file_path, output_path))
            output.append((file_path, replacement_corrosion_map))

        if scaling_factor is not None:
            scaled_corrosion_map = {
                k: v / scaling_factor
                for k, v in corrosion_map.items()
            }
            output.append((file_path, scaled_corrosion_map))

    return output
