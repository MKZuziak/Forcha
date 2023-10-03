import numpy as np
import copy
from forcha.models.pytorch.federated_model import FederatedModel
from forcha.utils.optimizers import Optimizers
from forcha.utils.computations import Aggregators
from collections import OrderedDict


class LSAA():
    """LSAA is used to establish the marginal contribution of each sampled
    client to the general value of the global model. LSAA is based on the assumption
    that we can detect the influence that a sampled client has on a general model
    by testing a scenario in which we have more-alike clients included in the sample."""
    
    def __init__(self,
                 nodes: list,
                 iterations: int) -> None:
        """Constructor for the LSAA. Initializes empty
        hash tables for LSAA value for each iteration as well as hash table
        for final LSAA values.
        
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
        
        self.lsaa = {node: np.float64(0) for node in nodes} # Hash map containing all the nodes and their respective marginal contribution values.
        self.partial_lsaa = {round:{node: np.float64(0) for node in nodes} for round in range(iterations)} # Hash map containing all the partial psi for each sampled subset.
    

    def update_lsaa(self,
                    model_template: FederatedModel,
                    optimizer_template: Optimizers,
                    gradients: OrderedDict,
                    nodes_in_sample: list,
                    optimizer: OrderedDict,
                    search_length: int,
                    iteration: int,
                    previous_model: OrderedDict,
                    return_coalitions: bool = True):
        """Method used to track_results after each training round.
        Given the graidnets, ids of the nodes included in sample,
        last version of the optimizer, previous version of the model
        and the updated version of the model, it calculates values of
        all the marginal contributions using LSAA.
        
        Parameters
        ----------
        gradients: OrderedDict
            An OrderedDict containing gradients of the sampled nodes.
        nodes_in_sample: list
            A list containing id's of the nodes that were sampled.
        optimizer: Optimizers
            An instance of the forcha.Optimizers class.
        search length: int,
            A number of replicas that should be included in LSA search.
        iteration: int
            The current iteration.
        previous_model: FederatedModel
            An instance of the FederatedModel object.
        updated_model: FederatedModel
            An instance of the FederatedModel object.
        Returns
        -------
        None
        """
        
        recorded_values = {}

        for node in nodes_in_sample:
            # Baseline case
            optimizer_template.set_weights(previous_delta=copy.deepcopy(optimizer[0]),
                                           previous_momentum=copy.deepcopy(optimizer[1]),
                                           learning_rate=copy.deepcopy(optimizer[2]))
            copy_gradients = copy.deepcopy(gradients)
            del copy_gradients[node.node_id]
            grad_avg = Aggregators.compute_average(copy_gradients)
            updated_weights = optimizer_template.fed_optimize(
                weights=copy.deepcopy(previous_model),
                delta=grad_avg)
            model_template.update_weights(updated_weights)
            baseline_score = model_template.evaluate_model()[1]
            recorded_values[tuple(copy_gradients.keys())] = baseline_score
            
            
            # Appended case
            optimizer_template.set_weights(previous_delta=copy.deepcopy(optimizer[0]),
                                           previous_momentum=copy.deepcopy(optimizer[1]),
                                           learning_rate=copy.deepcopy(optimizer[2]))
            copy_gradients = copy.deepcopy(gradients)
            del copy_gradients[node.node_id]
            
            for phi in range(search_length):
                copy_gradients[(f"{phi + 1}_of_{node.node_id}")] = copy.deepcopy(gradients[node.node_id])
            
            grad_avg = Aggregators.compute_average(copy_gradients)
            updated_weights = optimizer_template.fed_optimize(
                weights=copy.deepcopy(previous_model),
                delta=grad_avg)
            model_template.update_weights(updated_weights)
            appended_score = model_template.evaluate_model()[1]
            recorded_values[tuple(copy_gradients.keys())] = appended_score 
            
            lsaa_score = appended_score - baseline_score
            self.partial_lsaa[iteration][node.node_id] = lsaa_score
            print(f"Evaluated LSAA of client {node.node_id}") #TODO
       
        if return_coalitions == True:
            return recorded_values
    
    def return_last_value(self,
                          iteration:int) -> dict:
        """Method used to return the results of the last evaluation round.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        tuple[dict[int: dict], dict[int: float]]
        """
        values = self.partial_lsaa[iteration]
        return values
        
    
    def calculate_final_lsaa(self) -> tuple[dict[int: dict], dict[int: float]]:
        """Method used to sum up all the partial LOO scores to obtain
        a final LOO score for each client.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        tuple[dict[int: dict], dict[int: float]]
        """
        
        for iteration_results in self.partial_lsaa.values():
            for node, value in iteration_results.items():
                self.lsaa[node] += np.float64(value)
        return (self.partial_lsaa, self.lsaa)

