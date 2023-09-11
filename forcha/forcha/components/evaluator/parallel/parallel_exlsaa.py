from forcha.components.evaluator.exlsaa_evaluator import EXLSAA
import numpy as np
import copy
from forcha.models.pytorch.federated_model import FederatedModel
from forcha.utils.optimizers import Optimizers
from forcha.utils.computations import Aggregators
from collections import OrderedDict
from multiprocessing import Pool

def calculate_exlsaa(node_id: int,
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
    
    exlsaa = 0
    
    # Creating 'appended' gradients    
    for phi in range(search_length):
        gradients[(f"{phi + 1}_of_{node_id}")] = copy.deepcopy(node_gradient)
    
        # Calculating new score form appended gradients
        delta = Aggregators.compute_average(copy.deepcopy(gradients))
        temp_optim = copy.deepcopy(optimizer)
        temp_model = copy.deepcopy(previous_model)
        
        weights = temp_optim.fed_optimize(weights=temp_model.get_weights(),
                                         delta=delta)
        temp_model.update_weights(weights)
        score = temp_model.quick_evaluate()[1]
        recorded_values[tuple(gradients.keys())] = score
        exlsaa += score - baseline_score
    
    exlsaa = exlsaa / search_length
    
    return (node_id, recorded_values, exlsaa)



class Parallel_EXLSAA(EXLSAA):
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
            results = [pool.apply_async(calculate_exlsaa, (node.node_id, copy.deepcopy(gradients), copy.deepcopy(optimizer), \
                copy.deepcopy(previous_model), search_length)) for node in nodes_in_sample]
            for result in results:
                node_id, recorded, lsaa_score = result.get()
                recorded_values.update(recorded)
                self.partial_lsaa[iteration][node_id] = lsaa_score
        
        if return_coalitions == True:
            return recorded_values