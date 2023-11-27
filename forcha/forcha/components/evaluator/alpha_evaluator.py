import numpy as np
import copy
from forcha.models.federated_model import FederatedModel
from forcha.utils.optimizers import Optimizers
from forcha.utils.computations import Aggregators
from collections import OrderedDict


class Alpha_Amplified():
    """Alpha-amplification is used to establish the marginal contribution of each sampled
    client to the general value of the global model. Amplification is based on the assumption
    that we can detect the influence that a sampled client has on a general model
    by testing a scenario in which we have more-alike clients included in the sample."""
    
    def __init__(self,
                 nodes: list,
                 iterations: int) -> None:
        """Constructor for the Alpha-Amplification. Initializes empty
        hash tables for Amplification value for each iteration as well as hash table
        for final values.
        
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
        
        self.alpha = {node: np.float64(0) for node in nodes} # Hash map containing all the nodes and their respective marginal contribution values.
        self.partial_alpha = {round:{node: np.float64(0) for node in nodes} for round in range(iterations)} # Hash map containing all the partial psi for each sampled subset.
    

    def update_alpha(self,
                    model_template: FederatedModel,
                    optimizer_template: Optimizers,
                    gradients: OrderedDict,
                    nodes_in_sample: list,
                    optimizer: OrderedDict,
                    search_length: int,
                    iteration: int,
                    previous_model: OrderedDict,
                    final_model: OrderedDict,
                    return_coalitions: bool = True):
        """Method used to track_results after each training round.
        Given the graidnets, ids of the nodes included in sample,
        last version of the optimizer, previous version of the model
        and the updated version of the model, it calculates values of
        all the marginal contributions using alpha-amplification.
        
        Parameters
        ----------
        gradients: OrderedDict
            An OrderedDict containing gradients of the sampled nodes.
        nodes_in_sample: list
            A list containing id's of the nodes that were sampled.
        optimizer: Optimizers
            An instance of the forcha.Optimizers class.
        search length: int,
            A number of replicas that should be included in search.
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
            
            for phi in range(search_length):
                gradients_copy[(f"{phi + 1}_of_{node.node_id}")] = copy.deepcopy(gradients[node.node_id])
            
            grad_avg = Aggregators.compute_average(gradients_copy)
            weights = optimizer_template.fed_optimize(
                weights=copy.deepcopy(previous_model),
                delta=grad_avg)
            model_template.update_weights(weights)
            appended_score = model_template.evaluate_model()[1]
            
            self.partial_alpha[iteration][node_id] = final_model_score - appended_score
            recorded_values[tuple(gradients_copy.keys())] = appended_score 
            print(f"Evaluated alpha-amplication score of client {node_id}")
       
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
        values = self.partial_alpha[iteration]
        return values
        
    
    def calculate_final_alpha(self) -> tuple[dict[int: dict], dict[int: float]]:
        """Method used to sum up all the partial LOO scores to obtain
        a final LOO score for each client.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        tuple[dict[int: dict], dict[int: float]]
        """
        
        for iteration_results in self.partial_alpha.values():
            for node, value in iteration_results.items():
                self.alpha[node] += np.float64(value)
        return (self.partial_alpha, self.alpha)

