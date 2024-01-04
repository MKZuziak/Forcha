"""This example envisages traning a common Convolutional Neural Network on a number of distributed datasets using
classical FedOpt algorithm. Local datasets are uniformly distributed, but some of them are transformed (noised) 
according to predefined schema. During the training, three contribution metrics are defined: LOO, LSAA and EXLSAA."""

import pickle
import os
from forcha.components.orchestrator.fedopt_orchestrator import Fedopt_Orchestrator
from forcha.templates.generate_template import basic_fedopt
from forcha.components.settings.init_settings import init_settings
from forcha.models.templates.mnist import MNIST_Expanded_CNN



def simulation():
    cwd = os.getcwd()
    config = {
        "orchestrator": {
            "iterations": 1500,
            "number_of_nodes": 20,
            "sample_size": 20,
            'enable_archiver': True,
            'dispatch_model': True,
            "archiver":{
                "root_path": cwd,
                "orchestrator": True,
                "clients_on_central": True,
                "central_on_local": True,
                "log_results": True,
                "save_results": True,
                "save_orchestrator_model": False,
                "save_nodes_model": False,
                "form_archive": True
                },
            "optimizer": {
                "name": "FedAdagard",
                "learning_rate": 0.1,
                'b1': 0.0,
                'tau': 0.01}
            },
        "nodes":{
            "local_epochs": 3,
            "model_settings": {
                "optimizer": "SGD",
                "batch_size": 20,
                "learning_rate": 10**(-3/2),
                "FORCE_CPU": False}}}
    
    settings = init_settings(
         orchestrator_type='fed_opt',
         initialization_method='dict',
         dict_settings = config,
         allow_default=True)
    
    with open(r'/home/mzuziak/snap/snapd-desktop-integration/83/Documents/Forcha/forcha/tests/end_to_end/datasets/dataset_2/FMNIST_20_dataset_pointers', 'rb') as path:
        data = pickle.load(path)
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