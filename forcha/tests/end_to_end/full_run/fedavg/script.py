"""This example envisages traning a common Convolutional Neural Network on a number of distributed datasets using
classical FedOpt algorithm. Local datasets are uniformly distributed, but some of them are transformed (noised) 
according to predefined schema. During the training, three contribution metrics are defined: LOO, LSAA and EXLSAA."""

import pickle
import os
import datasets
from forcha.components.orchestrator.fedopt_orchestrator import Fedopt_Orchestrator
from forcha.components.settings.settings import Settings
from forcha.models.templates.mnist import MNIST_Expanded_CNN

def simulation():
    cwd = os.getcwd()
    settings = Settings(simulation_seed=42,
                        global_epochs=20,
                        local_epochs=3,
                        number_of_nodes=5,
                        sample_size=3,
                        optimizer='SGD',
                        batch_size=32,
                        learning_rate=0.01,
                        force_cpu=False)
    
    # with open(r'/home/mzuziak/snap/snapd-desktop-integration/83/Documents/Forcha/forcha/tests/end_to_end/datasets/dataset_2/FMNIST_20_dataset_pointers', 'rb') as path:
    #     data = pickle.load(path)
    with open(f'/home/mzuziak/snap/snapd-desktop-integration/83/Documents/Forcha/forcha/tests/end_to_end/datasets/dataset_2/FMNIST_20_dataset_pointers', 'rb') as file:
        data = pickle.load(file)
    # DATA: Selecting data for the orchestrator
    orchestrator_data = data[0]
    # DATA: Selecting data for nodes
    nodes_data = data[1]
    model = MNIST_Expanded_CNN()
    orchestrator = Fedopt_Orchestrator(settings=settings, full_debug=True)
    
    orchestrator.prepare_orchestrator(
         model=model, 
         validation_data=orchestrator_data)
    orchestrator.prepare_training(nodes_data=nodes_data)
    signal = orchestrator.train_protocol()
    return signal

if __name__ == "__main__":
       simulation()