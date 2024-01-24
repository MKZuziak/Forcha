"""This example envisages traning a common Convolutional Neural Network on a number of distributed datasets using
classical FedOpt algorithm. Local datasets are uniformly distributed, but some of them are transformed (noised) 
according to predefined schema. During the training, three contribution metrics are defined: LOO, LSAA and EXLSAA."""

import pickle
import os
import datasets
from forcha.components.orchestrator.mcfl_orchestrator import MCFL_Orchestrator
from forcha.components.settings.fedopt_settings import FedoptSettings
from forcha.models.templates.fmnist import create_FashionMnistNet
import numpy as np

def simulation():
    cwd = os.getcwd()
    settings = FedoptSettings(simulation_seed=42,
                        global_epochs=20,
                        local_epochs=3,
                        number_of_nodes=20,
                        sample_size=10,
                        optimizer='SGD',
                        batch_size=32,
                        learning_rate=0.01)
    
    # with open(r'/home/mzuziak/snap/snapd-desktop-integration/83/Documents/Forcha/forcha/tests/end_to_end/datasets/dataset_2/FMNIST_20_dataset_pointers', 'rb') as path:
    #     data = pickle.load(path)
    with open(f'/home/maciejzuziak/raid/documents/Forcha/forcha/tests/end_to_end/datasets/dataset_2/FMNIST_20_dataset_pointers', 'rb') as file:
        data = pickle.load(file)
    # DATA: Selecting data for the orchestrator
    orchestrator_data = data[0]
    # DATA: Selecting data for nodes
    nodes_data = data[1]
    model = create_FashionMnistNet()
    
    # Two gropus
    trans_matric = {
        0: np.array([
            [0.1, 0.8, 0.1],
            [0.1, 0.8, 0.1],
            [0.0, 0.9, 0.1]
        ]),
        1: np.array([
            [0.4, 0.3, 0.3],
            [0.2, 0.6, 0.2],
            [0.4, 0.1, 0.5]
        ])
    }
    groups_p = np.array([0.7, 0.3])
    
    orchestrator = MCFL_Orchestrator(
        settings=settings, 
        full_debug=True)
    
    orchestrator.prepare_orchestrator(
         model=model, 
         validation_data=orchestrator_data)
    orchestrator.prepare_training(nodes_data=nodes_data)
    signal = orchestrator.train_protocol(transition_matrices=trans_matric,
                                         group_probabilities=groups_p)
    return signal

if __name__ == "__main__":
       simulation()