from forcha.components.nodes.federated_node import FederatedNode
from forcha.components.settings.settings import Settings
import unittest
from datasets import load_dataset
from forcha.models.templates.mnist import MNIST_Expanded_CNN
import copy
import numpy as np
from collections import OrderedDict


class TestSettingsClass(unittest.TestCase):
    
    
    def test_transition(self):
        train_dataset = load_dataset("mnist", split="train")
        test_dataset = load_dataset('mnist', split="test")
        data = [train_dataset, test_dataset]
        settings = Settings()
        net = MNIST_Expanded_CNN()

        transition_matrix = np.array([
            [0.1, 0.4, 0.5],
            [0.1,0.8,0.1],
            [0.1,0.8,0.1]
        ])
        test_node = FederatedNode(node_id=2,
                                  settings=settings,
                                  model=net,
                                  data=data)
        test_node.load_transition_matrix(transition_matrix=transition_matrix)
        for iteration in range(20):
            test_node.update_state(iteration=iteration)


if __name__ == '__main__':
    unittest.main()