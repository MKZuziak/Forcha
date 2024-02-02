from forcha.components.evaluator.evaluation_manager import Evaluation_Manager
from forcha.components.settings.evaluator_settings import EvaluatorSettings
from forcha.models.federated_model import FederatedModel
from forcha.components.nodes.federated_node import FederatedNode
from forcha.utils.optimizers import Optimizers
from forcha.models.templates.mnist import MNIST_Expanded_CNN
import unittest
from datasets import load_dataset

import numpy as np
import copy


class TestEvaluatorClass(unittest.TestCase):
    
    
    def test_init(self):
        # Settings
        settings = EvaluatorSettings()
        
        # Dataset
        train_dataset = load_dataset("mnist", split="train")
        test_dataset = load_dataset('mnist', split="test")
        data = [train_dataset, test_dataset]
        
        # Model
        net = MNIST_Expanded_CNN()
        
        
        # Nodes
        node_1 = FederatedNode(
            node_id = 0,
            settings = settings,
            model = net,
            data = data
        )
        
        node_2 = FederatedNode(
            node_id = 1,
            settings = settings,
            model = net,
            data = data
        )
        
        # Orchestrator
        optimizer = Optimizers(
            weights = node_1.model.get_weights(),
            settings=settings)
        
        evaluation_manager = Evaluation_Manager(
            settings = settings,
            model_template = net,
            optimizer_template = optimizer,
            nodes = [0, 1],
            iterations=20)

        self.assertIsNotNone(evaluation_manager)


if __name__ == '__main__':
    unittest.main()