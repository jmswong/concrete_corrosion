import argparse
import corrosion_lib
import numpy as np
import output_lib
import re

parser = argparse.ArgumentParser(
    description="Extract and process corrosion and output files")
parser.add_argument(
    '--output_path',
    help=
    'Path to location of corrosion data. If --extract=True, the extracted files will also be extracted to here.'
)
parser.add_argument('--num_simulations',
                    type=int,
                    help='Number of simulations to process')
parser.add_argument(
    '--extract',
    action='store_true',
    help='If true, unzip and extract files from zipped filepath.')
parser.add_argument('--corrosion_zipped_filename', default='corrosion.zip')
parser.add_argument('--output_zipped_filename', default='Data_outputs.zip')

args = parser.parse_args()


def join_corrosion_and_outputs(corrosion_maps, output_maps):
    corrosion_output = []
    target_output = []
    for file_path, corrosion_map in corrosion_maps:
        file_name = file_path.split("/")[-1]
        m = re.search("Corrosion_simulation_(\d+)_timeStep_(\d+).txt",
                      file_name)
        simulation_idx, timestep = int(m.group(1)), int(m.group(2))
        output_filename = "output_%d_%d.mat" % (simulation_idx, timestep)

        # find the corresponding output map
        target_label = None
        for output_path, output_map in output_maps:
            if output_path.split("/")[-1] == output_filename:
                target_label = output_map['label']
                corrosion_output.append([simulation_idx, timestep])

                # append concrete properties from output
                corrosion_output[-1] += [
                    output_map['rebar'], output_map['cover'],
                    output_map['tensile_strength'], output_map['w_c']
                ]

        if target_label is None:
            print(
                "Skipping simulation %d timestep %d since output is missing" %
                (simulation_idx, timestep))
            continue

        target_output.append(target_label)

        # Since we've verified the x-axis evenly distributed and are are all on the
        # same scale, we can drop the points on the x-axis and represent the
        # corrosion depths for a single timestep as a vector in 1d or matrix in 2d.
        corrosion_depths = list(corrosion_map.values())

        # We will let the first and second columns of the output represent the simulation
        # number and timestep respectively.
        corrosion_output[-1] += corrosion_depths

    assert len(corrosion_output) == len(target_output)
    return np.array(corrosion_output), np.array(target_output)


def shuffle_numpy_pair(a, b):
    assert len(a) == len(b)
    np.random.seed(42)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def preprocess():
    # Extract the first num_simulations experiments to output_path.
    if (args.extract):
        simulation_timesteps = output_lib.extract_FEM_output(
            args.output_path + '/' + args.output_zipped_filename,
            args.num_simulations,
            extract_all=True)
        corrosion_lib.extract_corrosion_output(
            args.output_path + '/' + args.corrosion_zipped_filename,
            args.num_simulations,
            simulation_timesteps=simulation_timesteps)

    corrosion_maps = corrosion_lib.extract_1d_corrosion_maps(
        args.output_path,
        args.num_simulations,
        simulation_timesteps=simulation_timesteps)
    output_maps = output_lib.extract_concrete_outputs(
        args.output_path,
        args.num_simulations,
        simulation_timesteps=simulation_timesteps)

    # Rescale corrosion depths to all be on the same scale. Also replace any
    # corrosion depths from outputs.
    corrosion_maps = corrosion_lib.remap_output_scales(corrosion_maps,
                                                       output_maps)

    # check that all corrosion datapoints are on the same rebar scale
    assert corrosion_lib.verify_rebar_locations(corrosion_maps)

    # join corrosion and output data
    corrosion_array, labels_array = join_corrosion_and_outputs(
        corrosion_maps, output_maps)
    print("corrosion_array shape: ", corrosion_array.shape)
    print("labels_array shape: ", labels_array.shape)

    # shuffle and split to 80%/20% train/test sets
    corrosion_array, labels_array = shuffle_numpy_pair(corrosion_array,
                                                       labels_array)
    index_80 = int(corrosion_array.shape[0] * 0.8)
    corrosion_train = corrosion_array[:index_80]
    corrosion_test = corrosion_array[index_80:]
    labels_train = labels_array[:index_80]
    labels_test = labels_array[index_80:]
    print("corrosion_train shape: ", corrosion_train.shape)
    print("labels_train shape: ", labels_train.shape)
    print("corrosion_test shape: ", corrosion_test.shape)
    print("labels_test shape: ", labels_test.shape)

    # save ndarray to file
    with open(args.output_path + "/corrosion_train.npy", "wb") as f:
        np.save(f, corrosion_train)
    with open(args.output_path + "/labels_train.npy", "wb") as f:
        np.save(f, labels_train)
    with open(args.output_path + "/corrosion_test.npy", "wb") as f:
        np.save(f, corrosion_test)
    with open(args.output_path + "/labels_test.npy", "wb") as f:
        np.save(f, labels_test)


if __name__ == "__main__":
    preprocess()
