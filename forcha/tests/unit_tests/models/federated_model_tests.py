from forcha.models.federated_model import FederatedModel
from forcha.components.settings.settings import Settings
import unittest
from datasets import load_dataset
from forcha.models.templates.mnist import MNIST_Expanded_CNN
import copy
import numpy as np
from collections import OrderedDict

class TestSettingsClass(unittest.TestCase):
    
    
    def test_init(self):
        train_dataset = load_dataset("mnist", split="train")
        test_dataset = load_dataset('mnist', split="test")
        data = [train_dataset, test_dataset]
        settings = Settings()
        net = MNIST_Expanded_CNN()
        test_model = FederatedModel(settings=settings,
                                    net = net,
                                    local_dataset=data,
                                    node_name=0)
        self.assertIsNotNone(test_model)
        self.assertIsNotNone(test_model.net)
        self.assertIsNotNone(test_model.device)
        self.assertIsNotNone(test_model.optimizer)
        self.assertEqual(test_model.node_name, 0)
    
    
    def test_train(self):
        train_dataset = load_dataset("mnist", split="train")
        test_dataset = load_dataset('mnist', split="test")
        data = [train_dataset, test_dataset]
        settings = Settings()
        net = MNIST_Expanded_CNN()
        test_model = FederatedModel(settings=settings,
                                    net = net,
                                    local_dataset=data,
                                    node_name=0)
        for i in range(2):
            test_model.train(iteration=i, epoch=0)
    
    
    def test_evaluation(self):
        train_dataset = load_dataset("mnist", split="train")
        test_dataset = load_dataset('mnist', split="test")
        data = [train_dataset, test_dataset]
        settings = Settings()
        net = MNIST_Expanded_CNN()
        test_model = FederatedModel(settings=settings,
                                    net = net,
                                    local_dataset=data,
                                    node_name=0)
        results = test_model.evaluate_model()
        print(results)
        self.assertIsNotNone(results)
    
    
    def test_getgradients(self):
        train_dataset = load_dataset("mnist", split="train")
        test_dataset = load_dataset('mnist', split="test")
        data = [train_dataset, test_dataset]
        settings = Settings()
        net = MNIST_Expanded_CNN()
        test_model = FederatedModel(settings=settings,
                                    net = net,
                                    local_dataset=data,
                                    node_name=0)
        test_model.preserve_initial_model()
        
        initial_weights = copy.deepcopy(test_model.get_weights())
        
        # Checks if weights are correctly modified during the training
        for i in range(2):
            test_model.train(iteration=i, epoch=0)
        trained_weights = test_model.get_weights()
        preserved_initial_weights = test_model.initial_model.state_dict()
        for k in initial_weights.keys():
            self.assertTrue(np.allclose(initial_weights[k], preserved_initial_weights[k]))
            self.assertFalse(np.allclose(initial_weights[k], trained_weights[k]))
            self.assertFalse(np.allclose(preserved_initial_weights[k], trained_weights[k]))
        
        # Checks if gradients are correctly modified during the training
        obtained_gradients = test_model.get_gradients()
        weights_t1 = trained_weights
        weights_t2 = initial_weights
        calculated_gradients = OrderedDict.fromkeys(weights_t1.keys(), 0)
        for key in weights_t1:
            calculated_gradients[key] =  weights_t1[key] - weights_t2[key]
        for key in obtained_gradients.keys():
            self.assertTrue(np.allclose(obtained_gradients[k], calculated_gradients[k]))
        
        # Checks if the values are not modified after the gradient calculation operation
        for k in initial_weights.keys():
            self.assertTrue(np.allclose(initial_weights[k], preserved_initial_weights[k]))
            self.assertFalse(np.allclose(initial_weights[k], trained_weights[k]))
            self.assertFalse(np.allclose(preserved_initial_weights[k], trained_weights[k]))
        
        
    def test_footprint(self):
        train_dataset = load_dataset("mnist", split="train")
        test_dataset = load_dataset('mnist', split="test")
        data = [train_dataset, test_dataset]
        settings = Settings()
        net = MNIST_Expanded_CNN()
        test_model = FederatedModel(settings=settings,
                                    net = net,
                                    local_dataset=data,
                                    node_name=0)
        _ = test_model.print_model_footprint()


if __name__ == '__main__':
    unittest.main()