"""
Extracts and process full unlabeled corrosion dataset.
"""
import argparse
import re

import numpy as np

import corrosion_lib
import output_lib

parser = argparse.ArgumentParser(
    description="Extract and process corrosion and output files")
parser.add_argument('--output_path',
                    help='Path to location of corrosion data.',
                    default='/home/wongjames/cs230/Project/data_12_2_2022')
parser.add_argument('--corrosion_zipped_filename', default='corrosion.zip')
parser.add_argument('--verbose',
                    action='store_true',
                    help='True to print detailed debug')

args = parser.parse_args()


def construct_numpy_output(corrosion_maps):
    """
    Constructs a numpy matrix given corrosion map input.

    Args:
        corrosion_maps (list): List of dicts mapping from points on the x-axis
            of a horizontal rebar to corrosion depths.

    Returns:
        corrosion output (numpy array): Array of shape (num_samples, 343). Each
            row contains: simulation_idx, timestep, 4 concrete property
            featurs, and 337 corrosion depth features.
    """
    corrosion_output = []
    for file_path, corrosion_map in corrosion_maps:
        file_name = file_path.split("/")[-1]
        m = re.search("Corrosion_simulation_(\d+)_timeStep_(\d+).txt",
                      file_name)
        simulation_idx, timestep = int(m.group(1)), int(m.group(2))

        # Since we've verified the x-axis evenly distributed and are are all on the
        # same scale, we can drop the points on the x-axis and represent the
        # corrosion depths for a single timestep as a vector in 1d or matrix in 2d.
        corrosion_depths = list(corrosion_map.values())

        # Sanity check corrosion depths
        max_corrosion_depth = max(corrosion_depths)
        if (max_corrosion_depth > 0.1):
            print(
                "Skipping simulation %d timestep %d since max depth %f too high"
                % (simulation_idx, timestep, max_corrosion_depth))
            continue

        corrosion_output.append(corrosion_depths)
    print(len(corrosion_output))
    print(len(corrosion_maps))
    return np.array(corrosion_output)


def preprocess():
    simulation_timesteps = corrosion_lib.extract_all_corrosion_output(
        args.output_path + "/" + args.corrosion_zipped_filename,
        verbose=args.verbose)

    corrosion_maps = corrosion_lib.extract_1d_corrosion_maps(
        args.output_path + "_full", simulation_timesteps=simulation_timesteps)

    # check that all corrosion datapoints are on the same rebar scale
    assert corrosion_lib.verify_rebar_locations(corrosion_maps)

    output = construct_numpy_output(corrosion_maps)
    print("Processed %d unlabeled corrosion depths" % output.shape[0])

    # save ndarray to file
    with open(args.output_path + "/corrosion_unlabeled.npy", "wb") as f:
        np.save(f, output)


if __name__ == "__main__":
    preprocess()
