"""This example envisages traning a common Convolutional Neural Network on a number of distributed datasets using
classical FedOpt algorithm. Local datasets are uniformly distributed, but some of them are transformed (noised) 
according to predefined schema. During the training, three contribution metrics are defined: LOO, LSAA and EXLSAA."""

import pickle
import os
from forcha.components.orchestrator.evaluator_orchestrator import Evaluator_Orchestrator
from forcha.components.settings.init_settings import init_settings
from forcha.models.pytorch.mnist import MNIST_CNN



def simulation():
    cwd = os.getcwd()
    config = {
         "orchestrator": {
            "iterations": 5,
            "number_of_nodes": 3,
            "sample_size": 3,
            'enable_archiver': True,
            "archiver":{
                "root_path": os.getcwd(),
                "orchestrator": True,
                "clients_on_central": True,
                "central_on_local": True,
                "log_results": True,
                "save_results": True,
                "save_orchestrator_model": True,
                "save_nodes_model": True,
                "form_archive": True
                },
            "optimizer": {
                "name": "FedAdagard",
                "learning_rate": 0.1,
                "tau": 0.01,
                "b1": 0.1},
            "evaluator" : {
            "LOO_OR": False,
            "Shapley_OR": False,
            "IN_SAMPLE_LOO": True,
            "IN_SAMPLE_SHAP": False,
            "LSAA": True,
            "EXTENDED_LSAA": True,
            "line_search_length": 3,
            "preserve_evaluation": {
                "preserve_partial_results": True,
                "preserve_final_results": True},
            "full_debug": True,
            "number_of_workers": 50}},
        "nodes":{
        "local_epochs": 2,
        "model_settings": {
            "optimizer": "RMS",
            "batch_size": 32,
            "learning_rate": 0.001}}}
    
    settings = init_settings(
         orchestrator_type='evaluator',
         initialization_method='dict',
         dict_settings = config,
         allow_default=True)
    with open(r'C:\Users\macie\OneDrive\Dokumenty\GitHub\Forcha\forcha\tests\examples\mnist_lsaa\MNIST_3_dataset', 'rb') as path:
        data = pickle.load(path)
    # DATA: Selecting data for the orchestrator
    orchestrator_data = data[0]
    # DATA: Selecting data for nodes
    nodes_data = data[1]
    model = MNIST_CNN()
        
    orchestrator = Evaluator_Orchestrator(settings)
    
    orchestrator.prepare_orchestrator(
         model=model, 
         validation_data=orchestrator_data)
    
    signal = orchestrator.train_protocol(nodes_data=nodes_data)

if __name__ == "__main__":
       simulation()