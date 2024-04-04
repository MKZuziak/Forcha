from forcha.components.evaluator.parallel.parallel_manager import Parallel_Manager
from forcha.components.settings.evaluator_settings import EvaluatorSettings
from forcha.components.nodes.federated_node import FederatedNode
from forcha.utils.optimizers import Optimizers
from forcha.models.templates.mnist import MNIST_Expanded_CNN
from forcha.models.federated_model import FederatedModel
import unittest
from datasets import load_dataset
from torch import nn

import numpy as np
import copy


class TestParallelEvaluatorClass(unittest.TestCase):

    def test_init(self):
        # Settings
        settings = EvaluatorSettings()

        # Dataset
        train_dataset = load_dataset("mnist", split="train")
        test_dataset = load_dataset('mnist', split="test")
        data = [train_dataset, test_dataset]

        # Model
        module = MNIST_Expanded_CNN()
        net = FederatedModel(
            settings=settings,
            net=module,
            local_dataset=data,
            node_name=0
            )

        # Nodes
        node_1 = FederatedNode(
            node_id=0,
            settings=settings,
            model=module,
            data=data
        )


        # Orchestrator
        optimizer = Optimizers(
            weights=node_1.model.get_weights(),
            settings=settings)

        evaluation_manager = Parallel_Manager(
            settings=settings,
            optimizer_template=optimizer,
            model_template=net,
            nodes=[0, 1],
            iterations=20)

        self.assertIsNotNone(evaluation_manager)


if __name__ == '__main__':
    unittest.main()