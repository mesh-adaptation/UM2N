"""
Utilities for handling WarpMesh configurations.
"""
import yaml
import argparse


def load_yaml_to_namespace(yaml_file_path):
    """
    Read the WarpMesh configuration from a YAML file as a Namespace.

    :arg yaml_file_path: path to the YAML file to read from
    :return: WarpMesh configuration as a Namespace
    """
    # Read the YAML file
    with open(yaml_file_path + ".yaml", 'r') as file:
        yaml_dict = yaml.safe_load(file)

    # Convert the dictionary to an argparse.Namespace
    return argparse.Namespace(**yaml_dict)


def save_namespace_to_yaml(namespace, yaml_file_path):
    """
    Save the WarpMesh configuration Namespace to a YAML file.

    :arg namespace: the WarpMesh configuration as a Namespace
    :arg yaml_file_path: path to the YAML file to be written to
    """
    # Convert the Namespace to a dictionary
    namespace_dict = vars(namespace)

    # Write the dictionary to a YAML file
    with open(yaml_file_path + ".yaml", 'w') as file:
        yaml.dump(namespace_dict, file, default_flow_style=False)
