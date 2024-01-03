"""This example envisages traning a common Convolutional Neural Network on a number of distributed datasets using
classical FedOpt algorithm. Local datasets are uniformly distributed, but some of them are transformed (noised) 
according to predefined schema. During the training, three contribution metrics are defined: LOO, LSAA and EXLSAA."""

import pickle
import os
from forcha.components.orchestrator.evaluator_orchestrator import Evaluator_Orchestrator
from forcha.templates.generate_template import basic_evaluator
from forcha.components.settings.init_settings import init_settings
from forcha.models.templates.mnist import MNIST_MLP



def simulation():
    cwd = os.getcwd()
    config = basic_evaluator(iterations=4,
                             number_of_nodes=5,
                             sample_size=5,
                             root_path=cwd,
                             local_lr=0.1,
                             central_lr=1.0,
                             local_epochs=2,
                             batch_size=32,
                             force_cpu=True,
                             ALPHA=True)
    
    settings = init_settings(
         orchestrator_type='evaluator',
         initialization_method='dict',
         dict_settings = config,
         allow_default=True)
    
    with open(r'/home/mzuziak/snap/snapd-desktop-integration/83/Documents/Forcha/forcha/tests/end_to_end/datasets/dataset_2/FMNIST_20_dataset_pointers', 'rb') as path:
        data = pickle.load(path)
    # DATA: Selecting data for the orchestrator
    orchestrator_data = data[0]
    # DATA: Selecting data for nodes
    nodes_data = data[1]
    model = MNIST_MLP()
    orchestrator = Evaluator_Orchestrator(settings=settings, full_debug=True)
    
    orchestrator.prepare_orchestrator(
         model=model, 
         validation_data=orchestrator_data)
    orchestrator.prepare_training(nodes_data=nodes_data)
    signal = orchestrator.train_protocol()
    return signal

if __name__ == "__main__":
       simulation()