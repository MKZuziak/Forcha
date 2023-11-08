from forcha.components.evaluator.sample_evaluator import Sample_Evaluator
import numpy as np
import copy
from forcha.models.federated_model import FederatedModel
from forcha.utils.optimizers import Optimizers
from forcha.utils.computations import Aggregators
from collections import OrderedDict
from multiprocessing import Pool

def calculate_psi(node_id: int,
                   gradients: OrderedDict,
                   optimizer: Optimizers,
                   previous_model: FederatedModel,
                   baseline_score: float) -> tuple[int, dict, float]:
    recorded_values = {}
    del gradients[node_id]
    
    delta = Aggregators.compute_average(gradients)
    weights = optimizer.fed_optimize(weights=previous_model.get_weights(),
                                      delta=delta)
    previous_model.update_weights(weights)
    score = previous_model.quick_evaluate()[1]
    recorded_values[tuple(gradients.keys())] = score
    psi = baseline_score - score
    
    return (node_id, recorded_values, psi)


class Parallel_PSI(Sample_Evaluator):
    def __init__(self, 
                 nodes: list, 
                 iterations: int) -> None:
        super().__init__(nodes, iterations)
    
    
    def update_psi(self,
        gradients: OrderedDict,
        nodes_in_sample: list,
        optimizer: Optimizers,
        iteration: int,
        previous_model: FederatedModel,
        final_model: FederatedModel,
        return_coalitions: bool = True):
        
        recorded_values = {}
        baseline_score = final_model.quick_evaluate()[1]
        recorded_values[tuple(gradients.keys())] = baseline_score
        
        with Pool(len(nodes_in_sample)) as pool:
            results = [pool.apply_async(calculate_psi, (node.node_id, copy.deepcopy(gradients), copy.deepcopy(optimizer), \
                copy.deepcopy(previous_model), baseline_score)) for node in nodes_in_sample]
            for result in results:
                node_id, recorded, psi_score = result.get()
                recorded_values.update(recorded)
                self.partial_psi[iteration][node_id] = psi_score
        
        if return_coalitions == True:
            return recorded_values