import unittest
import numpy as np
from collections import OrderedDict
from torch import rand, zeros
from forcha.utils.optimizers import Optimizers
from forcha.components.settings.fedopt_settings import FedoptSettings
import copy

class TestSettingsClass(unittest.TestCase):
    
    def test_init(self):
        settings = FedoptSettings()
        layers = ['l1', 'l2', 'l3', 'l4', 'lout']
        fake_weights = OrderedDict((key, rand(3, 4)) for key in layers)
        optimizer = Optimizers(weights=fake_weights,
                   settings=settings)
        self.assertIsNotNone(optimizer)
        
    
    def test_simple_update_one(self):
        settings = FedoptSettings()
        layers = ['l1', 'l2', 'l3', 'l4', 'lout']
        original_weights = OrderedDict((key, rand(3, 4)) for key in layers)
        original_weights_copy = copy.deepcopy(original_weights)
        optimizer = Optimizers(
            weights=original_weights,
            settings=settings)
        
        # One FedUpdate test
        delta = OrderedDict((key, zeros(original_weights[key].size())) for key in original_weights.keys())
        for key in original_weights.keys():
            delta[key] = original_weights[key] / 1.5 + 0.2
        delta_copy = copy.deepcopy(delta)
        new_weights = optimizer.fed_optimize(
            weights=original_weights,
            delta = delta
        )
        
        for key in layers:
            # Original weights should not be changed by the operation
            self.assertTrue((original_weights[key] == original_weights_copy[key]).all())
            # Delta should not be changed by the operation
            self.assertTrue((delta[key] == delta_copy[key]).all())
            # new weights should be different from the previous weights
            self.assertFalse((new_weights[key] == original_weights[key]).all())

    def test_simple_update_two(self):
        settings = FedoptSettings()
        layers = ['l1', 'l2', 'l3', 'l4', 'lout']
        original_weights = OrderedDict((key, rand(3, 4)) for key in layers)
        original_weights_copy = copy.deepcopy(original_weights)
        optimizer = Optimizers(
            weights=original_weights,
            settings=settings)
        
        # One FedUpdate test
        delta = OrderedDict((key, zeros(original_weights[key].size())) for key in original_weights.keys())
        for key in original_weights.keys():
            delta[key] = original_weights[key] / 1.5 + 0.2
        delta_copy = copy.deepcopy(delta)
        new_weights = optimizer.fed_optimize(
            weights=original_weights,
            delta = delta
        )
        
        for key in layers:
            # Original weights should not be changed by the operation
            self.assertTrue((original_weights[key] == original_weights_copy[key]).all())
            # Delta should not be changed by the operation
            self.assertTrue((delta[key] == delta_copy[key]).all())
            # new weights should be different from the previous weights
            self.assertFalse((new_weights[key] == original_weights[key]).all())
        
        for key in new_weights.keys():
            delta[key] = new_weights[key] / 1.5 + 0.2
        delta_copy = copy.deepcopy(delta)
        new_weights = optimizer.fed_optimize(
            weights=new_weights,
            delta = delta
        )
        
        for key in layers:
            # Original weights should not be changed by the operation
            self.assertTrue((original_weights[key] == original_weights_copy[key]).all())
            # Delta should not be changed by the operation
            self.assertTrue((delta[key] == delta_copy[key]).all())
            # new weights should be different from the previous weights
            self.assertFalse((new_weights[key] == original_weights[key]).all())
        
        
    def test_fedadagard(self):
        settings = FedoptSettings(global_optimizer='FedAdagard',
                                  b1=0.4,
                                  b2=0.6,
                                  tau=0.3)
        layers = ['l1', 'l2', 'l3', 'l4', 'lout']
        original_weights = OrderedDict((key, rand(3, 4)) for key in layers)
        original_weights_copy = copy.deepcopy(original_weights)
        optimizer = Optimizers(
            weights=original_weights,
            settings=settings)

        self.assertEqual(optimizer.optimizer, 'FedAdagard')
        
        # One FedUpdate test
        delta = OrderedDict((key, zeros(original_weights[key].size())) for key in original_weights.keys())
        for key in original_weights.keys():
            delta[key] = original_weights[key] / 1.5 + 0.2
        delta_copy = copy.deepcopy(delta)
        # last_momentum = copy.deepcopy(optimizer.previous_momentum)
        # last_delta = copy.deepcopy(optimizer.previous_delta)
        last_momentum = optimizer.previous_momentum
        last_delta = optimizer.previous_delta
        
        new_weights = optimizer.fed_optimize(
            weights=original_weights,
            delta = delta
        )
        
        for key in layers:
            # Original weights should not be changed by the operation
            self.assertTrue((original_weights[key] == original_weights_copy[key]).all())
            # Delta should not be changed by the operation
            self.assertTrue((delta[key] == delta_copy[key]).all())
            # new weights should be different from the previous weights
            self.assertFalse((new_weights[key] == original_weights[key]).all())
            # new momentum and delta should be different
            self.assertFalse((optimizer.previous_momentum[key] == last_momentum[key]).all())
            self.assertFalse((optimizer.previous_momentum[key] == last_delta[key]).all())
        
        for key in new_weights.keys():
            delta[key] = new_weights[key] / 1.5 + 0.2
        delta_copy = copy.deepcopy(delta)
        last_momentum = copy.deepcopy(optimizer.previous_momentum)
        last_delta = copy.deepcopy(optimizer.previous_delta)
        new_weights = optimizer.fed_optimize(
            weights=new_weights,
            delta = delta
        )
        
        for key in layers:
            # Original weights should not be changed by the operation
            self.assertTrue((original_weights[key] == original_weights_copy[key]).all())
            # Delta should not be changed by the operation
            self.assertTrue((delta[key] == delta_copy[key]).all())
            # new weights should be different from the previous weights
            self.assertFalse((new_weights[key] == original_weights[key]).all())
            self.assertFalse((optimizer.previous_momentum[key] == last_momentum[key]).all())
            self.assertFalse((optimizer.previous_momentum[key] == last_delta[key]).all())
    
    
    def test_fedyogi(self):
        settings = FedoptSettings(global_optimizer='FedYogi',
                                  b1=0.4,
                                  b2=0.6,
                                  tau=0.3)
        layers = ['l1', 'l2', 'l3', 'l4', 'lout']
        original_weights = OrderedDict((key, rand(3, 4)) for key in layers)
        original_weights_copy = copy.deepcopy(original_weights)
        optimizer = Optimizers(
            weights=original_weights,
            settings=settings)
        
        self.assertEqual(optimizer.optimizer, 'FedYogi')
        
        # One FedUpdate test
        delta = OrderedDict((key, zeros(original_weights[key].size())) for key in original_weights.keys())
        for key in original_weights.keys():
            delta[key] = original_weights[key] / 1.5 + 0.2
        delta_copy = copy.deepcopy(delta)
        # last_momentum = copy.deepcopy(optimizer.previous_momentum)
        # last_delta = copy.deepcopy(optimizer.previous_delta)
        last_momentum = optimizer.previous_momentum
        last_delta = optimizer.previous_delta
        
        new_weights = optimizer.fed_optimize(
            weights=original_weights,
            delta = delta
        )
        
        for key in layers:
            # Original weights should not be changed by the operation
            self.assertTrue((original_weights[key] == original_weights_copy[key]).all())
            # Delta should not be changed by the operation
            self.assertTrue((delta[key] == delta_copy[key]).all())
            # new weights should be different from the previous weights
            self.assertFalse((new_weights[key] == original_weights[key]).all())
            # new momentum and delta should be different
            self.assertFalse((optimizer.previous_momentum[key] == last_momentum[key]).all())
            self.assertFalse((optimizer.previous_momentum[key] == last_delta[key]).all())
        
        for key in new_weights.keys():
            delta[key] = new_weights[key] / 1.5 + 0.2
        delta_copy = copy.deepcopy(delta)
        last_momentum = copy.deepcopy(optimizer.previous_momentum)
        last_delta = copy.deepcopy(optimizer.previous_delta)
        new_weights = optimizer.fed_optimize(
            weights=new_weights,
            delta = delta
        )
        
        for key in layers:
            # Original weights should not be changed by the operation
            self.assertTrue((original_weights[key] == original_weights_copy[key]).all())
            # Delta should not be changed by the operation
            self.assertTrue((delta[key] == delta_copy[key]).all())
            # new weights should be different from the previous weights
            self.assertFalse((new_weights[key] == original_weights[key]).all())
            self.assertFalse((optimizer.previous_momentum[key] == last_momentum[key]).all())
            self.assertFalse((optimizer.previous_momentum[key] == last_delta[key]).all())


    def test_fedadam(self):
        settings = FedoptSettings(global_optimizer='FedAdam',
                                  b1=0.4,
                                  b2=0.6,
                                  tau=0.3)
        layers = ['l1', 'l2', 'l3', 'l4', 'lout']
        original_weights = OrderedDict((key, rand(3, 4)) for key in layers)
        original_weights_copy = copy.deepcopy(original_weights)
        optimizer = Optimizers(
            weights=original_weights,
            settings=settings)
        
        self.assertEqual(optimizer.optimizer, 'FedAdam')
        
        # One FedUpdate test
        delta = OrderedDict((key, zeros(original_weights[key].size())) for key in original_weights.keys())
        for key in original_weights.keys():
            delta[key] = original_weights[key] / 1.5 + 0.2
        delta_copy = copy.deepcopy(delta)
        # last_momentum = copy.deepcopy(optimizer.previous_momentum)
        # last_delta = copy.deepcopy(optimizer.previous_delta)
        last_momentum = optimizer.previous_momentum
        last_delta = optimizer.previous_delta
        
        new_weights = optimizer.fed_optimize(
            weights=original_weights,
            delta = delta
        )
        
        for key in layers:
            # Original weights should not be changed by the operation
            self.assertTrue((original_weights[key] == original_weights_copy[key]).all())
            # Delta should not be changed by the operation
            self.assertTrue((delta[key] == delta_copy[key]).all())
            # new weights should be different from the previous weights
            self.assertFalse((new_weights[key] == original_weights[key]).all())
            # new momentum and delta should be different
            self.assertFalse((optimizer.previous_momentum[key] == last_momentum[key]).all())
            self.assertFalse((optimizer.previous_momentum[key] == last_delta[key]).all())
        
        for key in new_weights.keys():
            delta[key] = new_weights[key] / 1.5 + 0.2
        delta_copy = copy.deepcopy(delta)
        last_momentum = copy.deepcopy(optimizer.previous_momentum)
        last_delta = copy.deepcopy(optimizer.previous_delta)
        new_weights = optimizer.fed_optimize(
            weights=new_weights,
            delta = delta
        )
        
        for key in layers:
            # Original weights should not be changed by the operation
            self.assertTrue((original_weights[key] == original_weights_copy[key]).all())
            # Delta should not be changed by the operation
            self.assertTrue((delta[key] == delta_copy[key]).all())
            # new weights should be different from the previous weights
            self.assertFalse((new_weights[key] == original_weights[key]).all())
            self.assertFalse((optimizer.previous_momentum[key] == last_momentum[key]).all())
            self.assertFalse((optimizer.previous_momentum[key] == last_delta[key]).all())


if __name__ == '__main__':
    unittest.main()
        