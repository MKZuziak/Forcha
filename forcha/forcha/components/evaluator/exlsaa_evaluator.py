from forcha.components.evaluator.lsaa_evaluator import LSAA
import copy
from forcha.models.federated_model import FederatedModel
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
                exlsaa_score = 0
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
                
                for zeta in range(search_length):
                    # Appended case
                    optimizer_template.set_weights(previous_delta=copy.deepcopy(optimizer[0]),
                                                previous_momentum=copy.deepcopy(optimizer[1]),
                                                learning_rate=copy.deepcopy(optimizer[2]))
                    copy_gradients = copy.deepcopy(gradients)
                    del copy_gradients[node.node_id]
                    for phi in range((zeta + 1)):
                        copy_gradients[(f"{phi + 1}_of_{node.node_id}")] = copy.deepcopy(gradients[node.node_id])
                    
                    grad_avg = Aggregators.compute_average(copy_gradients)
                    updated_weights = optimizer_template.fed_optimize(
                        weights=copy.deepcopy(previous_model),
                        delta=grad_avg)
                    model_template.update_weights(updated_weights)
                    appended_score = model_template.evaluate_model()[1]
                    recorded_values[tuple(copy_gradients.keys())] = appended_score 
                    
                    exlsaa_score += (appended_score - baseline_score)
                
                print(f"Evaluated EXLSAA of client {node.node_id}") #TODO
                self.partial_lsaa[iteration][node.node_id] = exlsaa_score / search_length
            
            if return_coalitions == True:
                return recorded_values
    