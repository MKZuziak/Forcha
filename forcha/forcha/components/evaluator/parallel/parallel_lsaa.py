from forcha.components.evaluator.lsaa_evaluator import LSAA
import numpy as np
import copy
from forcha.models.federated_model import FederatedModel
from forcha.utils.optimizers import Optimizers
from forcha.utils.computations import Aggregators
from collections import OrderedDict
from multiprocessing import Pool

def calculate_lsaa(node_id: int,
                   gradients: OrderedDict,
                   optimizer: Optimizers,
                   previous_model: FederatedModel,
                   search_length: int) -> tuple[int, dict, float]:
    recorded_values = {}
    node_gradient = copy.deepcopy(gradients[node_id])
    del gradients[node_id]
    
    # Calculating baseline score
    optim_copy = copy.deepcopy(optimizer)
    model_copy = copy.deepcopy(previous_model)
    delta = Aggregators.compute_average(copy.deepcopy(gradients))
    weights = optim_copy.fed_optimize(weights=model_copy.get_weights(),
                                      delta=delta)
    model_copy.update_weights(weights)
    baseline_score = model_copy.quick_evaluate()[1]
    recorded_values[tuple(gradients.keys())] = baseline_score
    
    # Creating 'appended' gradients    
    for phi in range(search_length):
        gradients[(f"{phi + 1}_of_{node_id}")] = copy.deepcopy(node_gradient)
    
    # Calculating new score form appended gradients
    delta = Aggregators.compute_average(gradients)
    weights = optimizer.fed_optimize(weights=previous_model.get_weights(),
                                     delta=delta)
    previous_model.update_weights(weights)
    new_score = previous_model.quick_evaluate()[1]
    recorded_values[tuple(gradients.keys())] = new_score
    lsaa = new_score - baseline_score
    
    return (node_id, recorded_values, lsaa)



class Parallel_LSAA(LSAA):
    def __init__(self, 
                 nodes: list, 
                 iterations: int) -> None:
        super().__init__(nodes, iterations)
    
    
    def update_lsaa(self,
        gradients: OrderedDict,
        nodes_in_sample: list,
        optimizer: Optimizers,
        search_length: int,
        iteration: int,
        previous_model: FederatedModel,
        return_coalitions: bool = True):
        
        recorded_values = {}
        
        with Pool(len(nodes_in_sample)) as pool:
            results = [pool.apply_async(calculate_lsaa, (node.node_id, copy.deepcopy(gradients), copy.deepcopy(optimizer), \
                copy.deepcopy(previous_model), search_length)) for node in nodes_in_sample]
            for result in results:
                node_id, recorded, lsaa_score = result.get()
                recorded_values.update(recorded)
                self.partial_lsaa[iteration][node_id] = lsaa_score
        
        if return_coalitions == True:
            return recorded_values