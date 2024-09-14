import yaml
import argparse


def load_yaml_to_namespace(yaml_file_path):
    # Read the YAML file
    with open(yaml_file_path + ".yaml", "r") as file:
        yaml_dict = yaml.safe_load(file)

    # Convert the dictionary to an argparse.Namespace
    namespace = argparse.Namespace(**yaml_dict)

    return namespace


def save_namespace_to_yaml(namespace, yaml_file_path):
    # Convert the Namespace to a dictionary
    namespace_dict = vars(namespace)

    # Write the dictionary to a YAML file
    with open(yaml_file_path + ".yaml", "w") as file:
        yaml.dump(namespace_dict, file, default_flow_style=False)
