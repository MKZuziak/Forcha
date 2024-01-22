from forcha.components.nodes.federated_node import FederatedNode
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

        test_node = FederatedNode(node_id=2,
                                  settings=settings,
                                  model=net,
                                  data=data)
        
        self.assertIsNotNone(test_node)
        self.assertIsNotNone(test_node.node_id)
        self.assertIsNotNone(test_node.settings)
        self.assertIsNotNone(test_node.model)
        self.assertIsNotNone(test_node.train_data)
        self.assertIsNotNone(test_node.test_data)
    
    
    def test_train(self):
        train_dataset = load_dataset("mnist", split="train")
        test_dataset = load_dataset('mnist', split="test")
        data = [train_dataset, test_dataset]
        settings = Settings()
        net = MNIST_Expanded_CNN()

        test_node = FederatedNode(node_id=2,
                                  settings=settings,
                                  model=net,
                                  data=data)
        results = test_node.train_local_model(iteration=0,
                                        mode='gradients')
        self.assertIsNotNone(results)


if __name__ == '__main__':
    unittest.main()