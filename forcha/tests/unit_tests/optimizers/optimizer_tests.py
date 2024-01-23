import unittest
from collections import OrderedDict
from torch import rand
from forcha.utils.optimizers import Optimizers
from forcha.components.settings.fedopt_settings import FedoptSettings
from forcha.models.templates.mnist import MNIST_Expanded_CNN

class TestSettingsClass(unittest.TestCase):
    
    def test_init(self):
        settings = FedoptSettings()
        layers = ['l1', 'l2', 'l3', 'l4', 'lout']
        fake_weights = OrderedDict((key, rand(3, 4)) for key in layers)
        Optimizers(weights=fake_weights,
                   settings=settings)


if __name__ == '__main__':
    unittest.main()
        