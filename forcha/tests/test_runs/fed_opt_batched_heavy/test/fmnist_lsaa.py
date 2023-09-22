"""This example envisages traning a common Convolutional Neural Network on a number of distributed datasets using
classical FedOpt algorithm. Local datasets are uniformly distributed, but some of them are transformed (noised) 
according to predefined schema. During the training, three contribution metrics are defined: LOO, LSAA and EXLSAA."""

import pickle
import os
from forcha.components.orchestrator.fedopt_orchestrator import Fedopt_Orchestrator
from forcha.components.settings.init_settings import init_settings
from forcha.models.pytorch.fmnist import create_FashionMnistNet



def simulation():
    cwd = os.getcwd()
    config = {
         "orchestrator": {
            "iterations": 20,
            "number_of_nodes": 12,
            "sample_size": 12,
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
                "name": "Simple",
                "learning_rate": 0.5},
            "evaluator" : {
            "LOO_OR": False,
            "Shapley_OR": False,
            "IN_SAMPLE_LOO": True,
            "IN_SAMPLE_SHAP": False,
            "LSAA": True,
            "EXTENDED_LSAA": True,
            "ADAPTIVE_LSAA": False,
            "line_search_length": 1,
            "preserve_evaluation": {
                "preserve_partial_results": True,
                "preserve_final_results": True},
            "full_debug": True,
            "number_of_workers": 50}},
        "nodes":{
        "local_epochs": 3,
        "model_settings": {
            "optimizer": "Adam",
            "betas": (0.9, 0.8),
            "weight_decay": 1e-4,
            "amsgrad": True,
            "batch_size": 32,
            "learning_rate": 0.001,
            "gradient_clip": 2,
            "FORCE_CPU": False}}}
    
    settings = init_settings(
         orchestrator_type='fed_opt',
         initialization_method='dict',
         dict_settings = config,
         allow_default=True)
    with open(r'/home/mzuziak/documents/Forcha/forcha/tests/test_runs/fed_opt_batched_heavy/test/dataset/FMNIST_12_dataset', 'rb') as path:
        data = pickle.load(path)
    # DATA: Selecting data for the orchestrator
    orchestrator_data = data[0]
    # DATA: Selecting data for nodes
    nodes_data = data[1]
    model = create_FashionMnistNet()
    orchestrator = Fedopt_Orchestrator(settings, 
                                       batch_job = True,
                                       batch = 6,
                                       full_debug = True)
    
    orchestrator.prepare_orchestrator(
         model=model, 
         validation_data=orchestrator_data)
    
    signal = orchestrator.train_protocol(nodes_data=nodes_data)

if __name__ == "__main__":
       simulation()