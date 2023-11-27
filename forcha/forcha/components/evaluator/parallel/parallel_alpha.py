from forcha.components.evaluator.alpha_evaluator import Alpha_Amplified
import numpy as np
import copy
from forcha.models.federated_model import FederatedModel
from forcha.utils.optimizers import Optimizers
from forcha.utils.computations import Aggregators
from collections import OrderedDict
from multiprocessing import Pool

def calculate_alpha(node_id: int,
                   gradients: OrderedDict,
                   optimizer: Optimizers,
                   previous_model: FederatedModel,
                   baseline_score: float,
                   search_length: int) -> tuple[int, dict, float]:
    recorded_values = {}
    node_gradient = copy.deepcopy(gradients[node_id])
    del gradients[node_id]
        
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
    lsaa = baseline_score - new_score
    
    return (node_id, recorded_values, lsaa)



class Parallel_Alpha(Alpha_Amplified):
    def __init__(self, 
                 nodes: list, 
                 iterations: int) -> None:
        super().__init__(nodes, iterations)
    
    
    def update_alpha(self,
        gradients: OrderedDict,
        nodes_in_sample: list,
        optimizer: Optimizers,
        search_length: int,
        iteration: int,
        previous_model: FederatedModel,
        final_model: FederatedModel,
        return_coalitions: bool = True):
        
        recorded_values = {}
        baseline_score = final_model.quick_evaluate()[1]
        recorded_values[tuple(gradients.keys())] = baseline_score
        
        with Pool(len(nodes_in_sample)) as pool:
            results = [pool.apply_async(calculate_alpha, (node.node_id, copy.deepcopy(gradients), copy.deepcopy(optimizer), \
                copy.deepcopy(previous_model), baseline_score, search_length)) for node in nodes_in_sample]
            for result in results:
                node_id, recorded, alpha_score = result.get()
                recorded_values.update(recorded)
                self.partial_alpha[iteration][node_id] = alpha_score
        
        if return_coalitions == True:
            return recorded_values