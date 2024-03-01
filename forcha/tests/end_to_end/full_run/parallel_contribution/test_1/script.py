"""This example envisages traning a common Convolutional Neural Network on a number of distributed datasets using
classical FedOpt algorithm. Local datasets are uniformly distributed, but some of them are transformed (noised) 
according to predefined schema. During the training, three contribution metrics are defined: LOO, LSAA and EXLSAA."""

import pickle
import os
import datasets
import time
from forcha.components.orchestrator.evaluator_orchestrator import Evaluator_Orchestrator
from forcha.components.settings.evaluator_settings import EvaluatorSettings
from forcha.models.templates.mnist import MNIST_Expanded_CNN

def simulation():
    t1 = time.time()
    cwd = os.getcwd()
    settings = EvaluatorSettings(simulation_seed=42,
                        global_epochs=30,
                        local_epochs=2,
                        number_of_nodes=5,
                        sample_size=5,
                        optimizer='SGD',
                        batch_size=32,
                        learning_rate=0.01,
                        in_sample_loo=True,
                        in_sample_shap=True,
                        in_sample_alpha=True,
                        line_search_length=0)
    
    with open(f'/home/mzuziak/snap/snapd-desktop-integration/83/Documents/Forcha/forcha/tests/end_to_end/datasets/dataset/MNIST_5_dataset_pointers', 'rb') as file:
        data = pickle.load(file)
    # DATA: Selecting data for the orchestrator
    orchestrator_data = data[0]
    # DATA: Selecting data for nodes
    nodes_data = [data[1][3] for _ in range(5)]
    model = MNIST_Expanded_CNN()
    orchestrator = Evaluator_Orchestrator(
        settings=settings, 
        full_debug=True,
        parallelization=True,
        number_of_workers=10
        )
    
    orchestrator.prepare_orchestrator(
         model=model, 
         validation_data=orchestrator_data)
    orchestrator.prepare_training(nodes_data=nodes_data)
    signal = orchestrator.train_protocol()
    
    t2 = time.time()
    with open('execution_time.txt', 'a') as file:
        file.write(f"{t2-t1} seconds -> {(t2-t1) / 60} minutes")
    return signal

if __name__ == "__main__":
       simulation()