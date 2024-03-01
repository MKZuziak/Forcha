import math
import copy
from collections import OrderedDict

import numpy as np

from forcha.utils.computations import Aggregators
from forcha.models.federated_model import FederatedModel
from forcha.utils.optimizers import Optimizers


class Sample_LOO_Evaluator():
    """Sample evaluator is used to establish the marginal contribution of each sampled
    client to the general value of the global model. Basic Sample Evaluator is able to
    assess the Leave-one-out value for every client included in the sample. It is also
    able to sum the marginal values to obain a final Leave-one-out value."""
    
    def __init__(
        self,
        nodes: list,
        iterations: int
        ) -> None:
        """Constructor for the Sample Evaluator Class. Initializes empty
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
        
        self.psi = {node: np.float64(0) for node in nodes} # Hash map containing all the nodes and their respective marginal contribution values.
        self.partial_psi = {round:{node: np.float64(0) for node in nodes} for round in range(iterations)} # Hash map containing all the partial psi for each sampled subset.


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
        optimizer_template: Optimizers
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
        
        recorded_values = {}
        
        model_template.update_weights(final_model)
        final_model_score = model_template.evaluate_model()[1]
        recorded_values[tuple(gradients.keys())] = final_model_score
        
        for node in nodes_in_sample:
            node_id = node.node_id
            gradients_copy = copy.deepcopy(gradients)
            del gradients_copy[node_id]
            optimizer_template.set_weights(previous_delta=copy.deepcopy(optimizer[0]),
                                           previous_momentum=copy.deepcopy(optimizer[1]),
                                           learning_rate=copy.deepcopy(optimizer[2]))
            grad_avg = Aggregators.compute_average(gradients_copy)
            weights = optimizer_template.fed_optimize(
                weights=copy.deepcopy(previous_model),
                delta = grad_avg
                )
            model_template.update_weights(weights)
            score = model_template.evaluate_model()[1]
            
            self.partial_psi[iteration][node_id] = final_model_score - score
            recorded_values[tuple(sorted(gradients_copy.keys()))] = score
            print(f"Evaluated LOO score of client {node_id}")
        if return_coalitions == True:
            return recorded_values


    def calculate_final_result(self) -> tuple[dict[int: dict], dict[int: float]]:
        """Method used to sum up all the partial LOO scores to obtain
        a final LOO score for each client.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        tuple[dict[int: dict], dict[int: float]]
        """
        
        for iteration_results in self.partial_psi.values():
            for node, value in iteration_results.items():
                self.psi[node] += np.float64(value)
        return (self.partial_psi, self.psi)