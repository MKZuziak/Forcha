from forcha.components.evaluator.lsaa_evaluator import LSAA
import copy
from forcha.models.pytorch.federated_model import FederatedModel
from forcha.utils.optimizers import Optimizers
from forcha.utils.computations import Aggregators
from collections import OrderedDict

class EXLSAA(LSAA):
    """EXLSAA is used to establish the marginal contribution of each sampled
    client to the general value of the global model. EXLSAA is an expanded version
    of an LSAA algorithm."""
    def __init__(self, nodes: list, iterations: int) -> None:
        """Constructor for the EXLSAA. Initializes empty
        hash tables for EXLSAA value for each iteration as well as hash table
        for final EXLSAA values.
        
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
    

    def update_lsaa(self,
                    gradients: OrderedDict,
                    nodes_in_sample: list,
                    optimizer: Optimizers,
                    search_length: int,
                    iteration: int,
                    previous_model: FederatedModel,
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
                lsaa_score = 0
                node_id = node.node_id
                # Deleting gradients of node i from the sample.
                marginal_gradients = copy.deepcopy(gradients)
                del marginal_gradients[node_id] 

                # Reconstrcuting the marginal model
                marginal_optim = copy.deepcopy(optimizer)
                marginal_model = copy.deepcopy(previous_model)
                marginal_grad_avg = Aggregators.compute_average(marginal_gradients) # AGGREGATING FUNCTION -> CHANGE IF NEEDED
                marginal_weights = marginal_optim.fed_optimize(weights=marginal_model.get_weights(),
                                                               delta=marginal_grad_avg)
                marginal_model.update_weights(marginal_weights)
                marginal_model_score = marginal_model.quick_evaluate()[1]

                recorded_values[tuple(marginal_gradients.keys())] = marginal_model_score
                
                for phi in range(search_length):
                    # Creating copies for the appended version
                    appended_gradients = copy.deepcopy(gradients)
                    del appended_gradients[node_id] 
                    appended_model = copy.deepcopy(previous_model)
                    appended_optimizer = copy.deepcopy(optimizer)
            
                    # Appending one client at a time
                    for j in range(phi+1):
                        appended_gradients[(f"{j + 1}_of_{node_id}")] = copy.deepcopy(gradients[node_id])                
                    
                    # Reconstructing the model
                    appended_grad_avg = Aggregators.compute_average(appended_gradients)
                    appended_weights = appended_optimizer.fed_optimize(weights=appended_model.get_weights(),
                                                                       delta = appended_grad_avg)
                    appended_model.update_weights(appended_weights)
                    appended_model_score = appended_model.quick_evaluate()[1]
                    lsaa_score += appended_model_score - marginal_model_score # Appending the EXLSAA value
                
                
                    recorded_values[tuple(appended_gradients.keys())] = appended_model_score
                
                self.partial_lsaa[iteration][node_id] = lsaa_score / search_length
                
                print(f"Evaluated EXLSAA of client {node_id}") #TODO
                del appended_model, appended_gradients, appended_optimizer, appended_weights, appended_grad_avg
                del marginal_model, marginal_gradients, marginal_optim, marginal_weights, marginal_grad_avg
        
            if return_coalitions == True:
                    return recorded_values
    