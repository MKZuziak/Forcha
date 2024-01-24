import unittest
from forcha.components.orchestrator.fedopt_orchestrator import Fedopt_Orchestrator
from forcha.components.settings.fedopt_settings import FedoptSettings
from forcha.models.templates.mnist import MNIST_Expanded_CNN
from datasets import load_dataset

class TestSettingsClass(unittest.TestCase):
    
    
    def test_init(self):
        settings = FedoptSettings(force_cpu=True)
        orchestrator = Fedopt_Orchestrator(settings=settings)
    
    
    def test_prepare_orchestrator(self):
        orchestrator_data = load_dataset('mnist', split="test")
        settings = FedoptSettings(force_cpu=True)
        model = MNIST_Expanded_CNN()
        orchestrator = Fedopt_Orchestrator(settings=settings)
        orchestrator.prepare_orchestrator(model=model, validation_data=orchestrator_data)
    
    
    def test_prepare_training(self):
        orchestrator_data = load_dataset('mnist', split="test")
        nodes_traindata = load_dataset('mnist', split='train')
        nodes_testdata = load_dataset('mnist', split='test')
        nodes_data = [[nodes_traindata, nodes_testdata]]

        settings = FedoptSettings(force_cpu=True)
        model = MNIST_Expanded_CNN()
        orchestrator = Fedopt_Orchestrator(settings=settings)
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
            local_epochs=1,
            force_cpu=True)
        model = MNIST_Expanded_CNN()
        orchestrator = Fedopt_Orchestrator(settings=settings)
        orchestrator.prepare_orchestrator(model=model, validation_data=orchestrator_data)
        orchestrator.prepare_training(nodes_data=nodes_data)
        orchestrator.train_protocol()


if __name__ == '__main__':
    unittest.main()