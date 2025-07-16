import csv
import os
import random
import shutil


def setup_directories(problem, mesh_type, base_dir=None, subdirs=None, dir_format=None):
    """
    Set up directories for storing data, plots, and logs.

    Args:
        base_dir (str): Base directory for the project.
        parameters (dict): Dictionary of parameters, including "mesh_type" and "problem".
            - "mesh_type" (int): Type of mesh used in the simulation (default: 0).
            - "problem" (str): Name of the problem (e.g., "burgers" or "helmholtz") (default: "default_problem").
        subdirs (list, optional): List of subdirectories to create. Defaults to:
            ["data", "plot", "log", "mesh", "mesh_fine"].
            Additional subdirectories like "plot_compare", "train", "test", and "val" are added for "helmholtz".
        dir_format (str, optional): Format string for the problem-specific directory. Must use placeholders
            matching keys in the `parameters` dictionary. Example:
            "lc={lc}_ngrid_{n_grid}_n={n_case}_{data_type}_{scheme}_meshtype_{mesh_type}".
            If not provided, raises a ValueError.

    Returns:
        dict: A dictionary mapping subdirectory names to their full paths.

    Raises:
        ValueError: If `dir_format` is not provided or is invalid.
    """

    # Define the project directory
    if base_dir:
        project_dir = os.path.abspath(base_dir)
    else:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # QC:
    print(f"Project Directory: {project_dir}")

    # Define the dataset directory
    dataset_dir = os.path.join(
        project_dir, "data", f"dataset_meshtype_{mesh_type}", problem
    )

    # Use the provided format string for the problem-specific directory
    if dir_format is None:
        problem_specific_dir = os.path.join(
            dataset_dir, f"{problem}_meshtype_{mesh_type}"
        )
    else:
        # check if dir_format is a valid string format
        if not isinstance(dir_format, str):
            raise ValueError("dir_format must be a string.")
        problem_specific_dir = os.path.join(dataset_dir, dir_format)

    # Define default subdirectories if not provided
    if subdirs is None:
        subdirs = [
            "data",
            "plot",
            "log",
            "mesh",
            "mesh_fine",
            "plot_compare",
            "train",
            "test",
            "val",
        ]

    # Create and clear directories
    directories = {}
    for subdir in subdirs:
        dir_path = os.path.join(problem_specific_dir, subdir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            # Clear the directory by removing all files
            for file in os.listdir(dir_path):
                os.remove(os.path.join(dir_path, file))
        directories[subdir] = dir_path

    # QC:
    # print(f"Subdirectories created: {directories}")

    return directories


def output_csv(parameters, key_list, output_dir):
    """
    Write selected parameters to a CSV file.

    Args:
        parameters (dict): Dictionary of parameters to write.
        key_list (list): List of keys to include in the CSV.
        output_dir (str): Directory where the CSV file will be saved.
    """
    # Filter parameters based on key_list
    csv_keys = [key for key in key_list if key in parameters]
    csv_data = [parameters[key] for key in csv_keys]

    # Define the output file path
    csv_file_path = os.path.join(output_dir, "info.csv")

    # Write to CSV
    with open(csv_file_path, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header (keys)
        csv_writer.writerow(csv_keys)
        # Write data (values)
        csv_writer.writerow(csv_data)

    print(f"Parameters saved to {csv_file_path}")


def move_data(target, source, start, num_files):
    """
    Move data files from the source directory to the target directory.

    Args:
        target (str): The path to the target directory.
        source (str): The path to the source directory.
        start (int): The starting index of the files to move.
        num_files (int): The total number of files to move.

    Raises:
        FileNotFoundError: If the source directory does not exist.
        ValueError: If the start index or num_files is invalid.
    """
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source directory '{source}' does not exist.")

    if start < 0 or num_files <= 0:
        raise ValueError("Invalid start index or number of files to move.")

    # Create the target directory if it doesn't exist
    if not os.path.exists(target):
        os.makedirs(target)
    else:
        # Clear the target directory by removing all files
        for file in os.listdir(target):
            os.remove(os.path.join(target, file))

    # Copy files sequentially starting from the specified index
    for i in range(start, start + num_files):
        try:
            # Copy the data file
            shutil.copy(
                os.path.join(source, f"data_{i:04d}.npy"),
                os.path.join(target, f"data_{i:04d}.npy"),
            )
        except FileNotFoundError:
            print(f"File data_{i:04d}.npy not found in {source}. Skipping.")
            continue
        except Exception as e:
            print(f"An error occurred while copying data_{i:04d}.npy: {e}")
            continue


def split_data(
    source_dir,
    train_dir,
    test_dir,
    val_dir,
    train_ratio=0.75,
    test_ratio=0.15,
    val_ratio=0.1,
):
    """
    Split files in a source directory into train, test, and validation directories.

    Args:
        source_dir (str): Path to the source directory containing files.
        train_dir (str): Path to the train directory.
        test_dir (str): Path to the test directory.
        val_dir (str): Path to the validation directory.
        train_ratio (float): Proportion of files to allocate to the train set.
        test_ratio (float): Proportion of files to allocate to the test set.
        val_ratio (float): Proportion of files to allocate to the validation set.

    Raises:
        ValueError: If the sum of train_ratio, test_ratio, and val_ratio is not 1.
    """
    # Validate ratios
    if not (0 <= train_ratio <= 1 and 0 <= test_ratio <= 1 and 0 <= val_ratio <= 1):
        raise ValueError("Ratios must be between 0 and 1.")
    if train_ratio + test_ratio + val_ratio != 1:
        raise ValueError(
            "The sum of train_ratio, test_ratio, and val_ratio must equal 1."
        )

    # Get all files in the source directory
    files = [
        f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))
    ]
    random.shuffle(files)  # Shuffle files for unbiased distribution

    # QC:
    # print(f'files {files}')

    # Calculate split indices - preference train > test > val
    total_files = len(files)
    num_train = int(total_files * train_ratio)
    num_test = max(int(total_files * test_ratio), total_files - num_train)
    num_val = total_files - num_train - num_test

    # Distribute files
    train_files = files[:num_train]
    test_files = files[num_train : num_train + num_test]
    val_files = files[num_train + num_test :]

    for datafiles, target_dir in zip(
        [train_files, test_files, val_files], [train_dir, test_dir, val_dir]
    ):
        for datafile in datafiles:
            shutil.copy(
                os.path.join(source_dir, datafile), os.path.join(target_dir, datafile)
            )

    print(
        f"Data split complete: {num_train} train, {num_test} test, {num_val} validation files."
    )
