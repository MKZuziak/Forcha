import unittest
from forcha.components.settings.settings import Settings
import time

class TestSettingsClass(unittest.TestCase):

    def test_default(self):
        test_object = Settings()
        self.assertEqual(test_object.batch_size, 32)
        self.assertEqual(test_object.simulation_seed, 42)
        self.assertEqual(test_object.global_epochs, 10)
        self.assertEqual(test_object.local_epochs, 2)
        self.assertEqual(test_object.number_of_nodes, 10)
        self.assertEqual(test_object.optimizer, 'RMS')
    
    
    def test_cystom(self):
        simulation_seed = 52
        global_epochs = 20
        local_epochs = 2
        number_of_nodes = 25
        optimizer = 'SGD'
        momentum = 1e4
        nesterov = True
        batch_size = 64
        
        time.sleep(0.02)
        test_object = Settings(simulation_seed=simulation_seed,
                               global_epochs=global_epochs,
                               local_epochs=local_epochs,
                               number_of_nodes=number_of_nodes,
                               optimizer=optimizer,
                               batch_size=batch_size,
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
        
        
if __name__ == '__main__':
    unittest.main()