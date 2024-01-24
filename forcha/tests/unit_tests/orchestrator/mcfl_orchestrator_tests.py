import unittest
from forcha.components.orchestrator.mcfl_orchestrator import MCFL_Orchestrator
from forcha.components.settings.fedopt_settings import FedoptSettings
from forcha.models.templates.mnist import MNIST_Expanded_CNN
from datasets import load_dataset
import numpy as np

class TestSettingsClass(unittest.TestCase):
    
    def test_init(self):
        orchestrator_data = load_dataset('mnist', split="test")
        settings = FedoptSettings()
        model = MNIST_Expanded_CNN()
        orchestrator = MCFL_Orchestrator(settings=settings)
        orchestrator.prepare_orchestrator(model=model, validation_data=orchestrator_data)
    
    
    def test_prepare_training(self):
        orchestrator_data = load_dataset('mnist', split="test")
        nodes_traindata = load_dataset('mnist', split='train')
        nodes_testdata = load_dataset('mnist', split='test')
        nodes_data = [[nodes_traindata, nodes_testdata]]

        settings = FedoptSettings()
        model = MNIST_Expanded_CNN()
        orchestrator = MCFL_Orchestrator(settings=settings)
        orchestrator.prepare_orchestrator(model=model, validation_data=orchestrator_data)
        orchestrator.prepare_training(nodes_data=nodes_data)
    
    
    def test_training(self):
        orchestrator_data = load_dataset('mnist', split="test")
        nodes_traindata = load_dataset('mnist', split='train')
        nodes_testdata = load_dataset('mnist', split='test')
        nodes_data = [[nodes_traindata, nodes_testdata]]

        settings = FedoptSettings(
            number_of_nodes=1,
            sample_size=1,
            local_epochs=1)
        model = MNIST_Expanded_CNN()
        orchestrator = MCFL_Orchestrator(settings=settings)
        orchestrator.prepare_orchestrator(model=model, validation_data=orchestrator_data)
        orchestrator.prepare_training(nodes_data=nodes_data)
        transition_matrices = {
            0: np.array([
            [0.1, 0.4, 0.5],
            [0.1,0.8,0.1],
            [0.1,0.8,0.1]
        ]),
            1: np.array([
            [0.1, 0.4, 0.5],
            [0.1,0.8,0.1],
            [0.1,0.8,0.1]
        ]),
            2: np.array([
            [0.1, 0.4, 0.5],
            [0.1,0.8,0.1],
            [0.1,0.8,0.1]
        ])
        }
        group_probabilities = np.array([0.1, 0.8, 0.1])
        
        
        orchestrator.train_protocol(
            transition_matrices=transition_matrices,
            group_probabilities=group_probabilities
        )


if __name__ == '__main__':
    unittest.main()