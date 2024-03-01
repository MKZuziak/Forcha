import copy
from multiprocessing import Pool
from collections import OrderedDict

import numpy as np

from forcha.components.evaluator.loo_evaluator import Sample_LOO_Evaluator
from forcha.models.federated_model import FederatedModel
from forcha.utils.optimizers import Optimizers
from forcha.utils.computations import Aggregators


def calculate_psi(
    node_id: int,
    gradients: OrderedDict,
    optimizer: Optimizers,
    previous_model: FederatedModel,
    baseline_score: float,
    model_template: FederatedModel,
    optimizer_template: Optimizers,
    ) -> tuple[int, dict, float]:
    
    recorded_values = {}
    del gradients[node_id]
    optimizer_template.set_weights(
        previous_delta=copy.deepcopy(optimizer[0]),
        previous_momentum=copy.deepcopy(optimizer[1]),
        learning_rate=copy.deepcopy(optimizer[2])
        )
    grad_avg = Aggregators.compute_average(gradients)
    weights = optimizer_template.fed_optimize(
        weights=copy.deepcopy(previous_model),
        delta = grad_avg
        )
    model_template.update_weights(weights)
    score = model_template.evaluate_model()[1]
    recorded_values[tuple(sorted(gradients.keys()))] = score
    psi = baseline_score - score
    
    return (node_id, recorded_values, psi)


class Parallel_Sample_LOO_Evaluator(Sample_LOO_Evaluator):
    def __init__(
        self, 
        nodes: list, 
        iterations: int
        ) -> None:
        """Constructor for the Parallel Sample LOO Evaluator. Initializes empty
        hash tables for LOO value for each iteration as well as hash table
        for final LOO values.
        
        Parameters
        ----------
        nodes: list
            A list containing ids of all the nodes engaged in the training.
        iterations: int
            A number of training iterations
        
        Returns
        -------
        None
        """
        super().__init__(nodes, iterations)
    
    
    def evaluate_round(
        self,
        model_template: FederatedModel,
        optimizer_template: Optimizers,
        gradients: OrderedDict,
        nodes_in_sample: list,
        optimizer: Optimizers,
        iteration: int,
        final_model: OrderedDict,
        previous_model: OrderedDict,
        return_coalitions: bool = True
        ):
        """Method used to track_results after each training round.
        Given the graidnets, ids of the nodes included in sample,
        last version of the optimizer, previous version of the model
        and the updated version of the model, it calculates values of
        all the marginal contributions using Leave-one-out method.
        
        Parameters
        ----------
        model_temmplate: FederatedModel
            A template of the FederatedModel object used during the simulation.
        optimizer_template:
            A template of the Optimizer object used during the simulation.     
        gradients: OrderedDict
            An OrderedDict containing gradients of the sampled nodes.
        nodes_in_sample: list
            A list containing FederatedNodes that participated in the training.
        previous_optimizer: Optimizers
            An instance of the forcha.Optimizers class.
        iteration: int
            The current iteration.
        final_model: FederatedModel
            An instance of the FederatedModel object.
        previous_model: FederatedModel
            An instance of the FederatedModel object.
        return_coalitions: bool
            A bool flag indicating whether to score of every coalition.
        Returns
        -------
        None
        """
        
        print("Calculating LOO score in parallel")
        recorded_values = {}
        
        model_template.update_weights(final_model)
        final_model_score = model_template.evaluate_model()[1]
        recorded_values[tuple(gradients.keys())] = final_model_score
        
        with Pool(len(nodes_in_sample)) as pool:
            results = [pool.apply_async(
                calculate_psi, 
                (node.node_id, 
                 copy.deepcopy(gradients), 
                 copy.deepcopy(optimizer),
                 copy.deepcopy(previous_model), 
                 final_model_score, 
                 model_template, 
                 optimizer_template)) \
                for node in nodes_in_sample]
      
            for result in results:
                node_id, recorded, psi_score = result.get()
                recorded_values.update(recorded)
                self.partial_psi[iteration][node_id] = psi_score
        
        if return_coalitions == True:
            return recorded_values