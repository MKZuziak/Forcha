"""This scripts contain a data generation from a associated library fedata."""
from fedata.hub.generate_dataset import generate_dataset
import os

def generate():
    # Configuration for the generation script.
    data_config = {
    "dataset_name" : "mnist",
    "split_type" : "homogeneous",
    "shards": 5,
    "local_test_size": 0.3,
    "transformations": {0: {"transformation_type": "noise", "noise_multiplyer": 0.4},
                        1: {"transformation_type": "noise", "noise_multiplyer": 0.15}},
    "imbalanced_clients": {},
    "save_dataset": True,
    "save_transformations": True,
    "save_blueprint": True,
    "agents": 5,
    "shuffle": True,
    "save_path": os.getcwd()}
    # Execution
    generate_dataset(config=data_config)


if __name__ == "__main__":
    generate()