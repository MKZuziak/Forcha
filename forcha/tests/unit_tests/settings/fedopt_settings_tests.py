import unittest
from forcha.components.settings.fedopt_settings import FedoptSettings
import time

class TestSettingsClass(unittest.TestCase):

    def test_default(self):
        test_object = FedoptSettings()
        self.assertEqual(test_object.batch_size, 32)
        self.assertEqual(test_object.simulation_seed, 42)
        self.assertEqual(test_object.global_epochs, 10)
        self.assertEqual(test_object.local_epochs, 2)
        self.assertEqual(test_object.number_of_nodes, 10)
        self.assertEqual(test_object.optimizer, 'RMS')
        self.assertEqual(test_object.b1, 0)
        self.assertEqual(test_object.b2, 0)
        self.assertEqual(test_object.tau, 0)
        self.assertEqual(test_object.global_learning_rate, 1.0)
    
    
    def test_cystom(self):
        simulation_seed = 52
        global_epochs = 20
        local_epochs = 2
        number_of_nodes = 25
        optimizer = 'SGD'
        momentum = 1e4
        nesterov = True
        batch_size = 64
        global_learning_rate = 0.5
        b1 = 0.5
        b2 = 0.5
        tau = 0.2
        
        test_object = FedoptSettings(
            simulation_seed=simulation_seed,
            global_epochs=global_epochs,
            local_epochs=local_epochs,
            number_of_nodes=number_of_nodes,
            optimizer=optimizer,
            batch_size=batch_size,
            b1=b1,
            b2=b2,
            tau=tau,
            global_learning_rate=global_learning_rate,
            momentum = momentum,
            nesterov = nesterov)    
        
        self.assertEqual(test_object.batch_size, batch_size)
        self.assertEqual(test_object.simulation_seed, simulation_seed)
        self.assertEqual(test_object.global_epochs, global_epochs)
        self.assertEqual(test_object.local_epochs, local_epochs)
        self.assertEqual(test_object.number_of_nodes, number_of_nodes)
        self.assertEqual(test_object.optimizer, optimizer)
        self.assertEqual(test_object.nesterov, nesterov)
        self.assertEqual(test_object.batch_size, batch_size)
        self.assertEqual(test_object.momentum, momentum)
        self.assertEqual(test_object.b1, b1)
        self.assertEqual(test_object.b2, b2)
        self.assertEqual(test_object.tau, tau)
        self.assertEqual(test_object.global_learning_rate, global_learning_rate)
        
        
if __name__ == '__main__':
    unittest.main()