import copy
import math
from collections import OrderedDict
from _collections_abc import Generator

import numpy as np

from forcha.utils.computations import Aggregators
from forcha.utils.computations import Subsets
from forcha.models.federated_model import FederatedModel
from forcha.utils.optimizers import Optimizers


def compare_for_debug(dict1, dict2):
    for (row1, row2) in zip(dict1.values(), dict2.values()):
        if False in (row1 == row2):
            return False
        else:
            return True


def select_gradients(
    gradients: OrderedDict,
    query: list,
    in_place: bool = False
    ):
    """Helper function for selecting gradients that of nodes that are
    in the query.

    Parameters
    ----------
    sqe: OrderedDict
        An OrderedDict containing all the gradients.
    query: list
        A size containing ids of the searched nodes.
    in_place: bool, default to False
        No description
    
    Returns
    -------
    Generator
        A generator object that can be iterated to obtain chunks of the original iterable.
    """
    selected_gradients = {}
    if in_place == False:
        for node_id, gradient in gradients.items():
            if node_id in query:
                selected_gradients[node_id] = copy.deepcopy(gradient)
    else:
        for node_id, gradient in gradients.items():
            if node_id in query:
                selected_gradients[node_id] = gradient
    return selected_gradients


class Sample_Shapley_Evaluator():
    """Sample evaluator is used to establish the marginal contribution of each sampled
    client to the general value of the global model using Shapley Value as a method of
    assesment. Shapley Sample Evaluator is able to assess the Shapley value for every 
    client included in the sample. It is also able to sum the marginal values to obain 
    a final Shapley values."""
    
    def __init__(
        self,
        nodes: list,
        iterations: int
        ) -> None:
        """Constructor for the Shapley Sample Evaluator Class. Initializes empty
        hash tables for Shapley value for each iteration as well as hash table
        for final Shapley values.
        
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
        self.shapley = {node: np.float64(0) for node in nodes} # Hash map containing all the nodes and their respective marginal contribution values.
        self.partial_shapley = {round:{node: np.float64(0) for node in nodes} for round in range(iterations)} # Hash map containing all the partial psi for each sampled subset.
    

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
        ) -> dict:
        """Method used to track_results after each training round.
        Given the graidnets, ids of the nodes included in sample,
        last version of the optimizer, previous version of the model
        and the updated version of the model, it calculates values of
        all the marginal contributions using Shapley value.
        
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
        dict
        """
        
        # Operations counter to track the progress of the calculations.
        operation_counter = 1
        number_of_operations = 2 ** (len(nodes_in_sample)) - 1

        # Maps every coalition to it's value, implemented to decrease the time complexity.
        recorded_values = {}
        # recorded_values['zero_vector'] = 1 / len(list(gradients[0].values())[-1]) # value of a random guess, 1/ len(networks_last_layer)
        
        # Converting list of FederatedNode objects to the int representing their identiity.
        nodes_in_sample = [node.node_id for node in nodes_in_sample] 
        # Forming superset of all the possible coalitions.
        superset = Subsets.form_superset(nodes_in_sample, return_dict=True)

        for node in nodes_in_sample:
            shap = 0.0
            # Select subsets that do not contain agent i
            coalitions = Subsets.select_subsets(
                coalitions = superset, 
                searched_node = node
                )
            
            for coalition in coalitions.keys():
                coalition_without_client = tuple(sorted(coalition))
                coalition_with_client = tuple(sorted(coalition + (node, )))
                # Evaluating the performance of the model trained without client i
                # Check if already calculated in another pass
                if recorded_values.get(coalition_without_client):
                    coalition_without_client_score = recorded_values[coalition_without_client]
                # Else calculate the score and add to the registered values.
                else:
                    print(f"{operation_counter} of {number_of_operations}: forming and evaluating subset {coalition_without_client}")
                    coalitions_gradients = select_gradients(
                        gradients = gradients,
                        query = coalition_without_client
                        )
                    optimizer_template.set_weights(
                        previous_delta=copy.deepcopy(optimizer[0]),
                        previous_momentum=copy.deepcopy(optimizer[1]),
                        learning_rate=copy.deepcopy(optimizer[2])
                        )
                    grad_avg = Aggregators.compute_average(coalitions_gradients)
                    weights = optimizer_template.fed_optimize(
                        weights=copy.deepcopy(previous_model),
                        delta = grad_avg
                        )
                    model_template.update_weights(weights)
                    coalition_without_client_score = model_template.evaluate_model()[1]
                    recorded_values[coalition_without_client] = coalition_without_client_score
                    operation_counter += 1
                # Evaluating the performance of the model trained with client i
                # Check if already calculated in another pass
                if recorded_values.get(coalition_with_client):
                    coalition_with_client_score = recorded_values[coalition_with_client]
                # Else calculate the score and add to the registered values.
                else:
                    print(f"{operation_counter} of {number_of_operations}: forming and evaluating subset {coalition_with_client}")
                    coalitions_gradients = select_gradients(
                        gradients = gradients, 
                        query = coalition_with_client
                        )
                    optimizer_template.set_weights(
                        previous_delta=copy.deepcopy(optimizer[0]),
                        previous_momentum=copy.deepcopy(optimizer[1]),
                        learning_rate=copy.deepcopy(optimizer[2])
                        )
                    grad_avg = Aggregators.compute_average(coalitions_gradients)
                    weights = optimizer_template.fed_optimize(
                        weights=copy.deepcopy(previous_model),
                        delta = grad_avg
                        )
                    model_template.update_weights(weights)
                    coalition_with_client_score = model_template.evaluate_model()[1]
                    recorded_values[coalition_with_client] = coalition_with_client_score
                    operation_counter += 1
                possible_combinations = math.comb((len(nodes_in_sample) - 1), len(coalition_without_client)) # Find the total number of possibilities to choose k things from n items:
                divisor = 1 / possible_combinations
                shap += divisor * (coalition_with_client_score - coalition_without_client_score)
            
            # # Calculating zero coalition - random guess
            # coalition_without_client_score = recorded_values['zero_vector']
            # coalition_with_client = (node, )
            # if recorded_values.get(coalition_with_client):
            #     coalition_with_client_score = recorded_values[coalition_with_client]
            # else:
            #     print(f"{operation_counter} of {number_of_operations}: forming and evaluating subset {coalition_with_client}")
            #     coalitions_gradients = select_gradients(
            #         gradients = gradients, 
            #         query = coalition_with_client
            #         )
            #     optimizer_template.set_weights(
            #         previous_delta=copy.deepcopy(optimizer[0]),
            #         previous_momentum=copy.deepcopy(optimizer[1]),
            #         learning_rate=copy.deepcopy(optimizer[2])
            #         )
            #     grad_avg = Aggregators.compute_average(coalitions_gradients)
            #     weights = optimizer_template.fed_optimize(
            #         weights=copy.deepcopy(previous_model),
            #         delta = grad_avg
            #         )
            #     model_template.update_weights(weights)
            #     coalition_with_client_score = model_template.evaluate_model()[1]
            #     recorded_values[coalition_with_client] = coalition_with_client_score
            #     operation_counter += 1
            #     shap += coalition_with_client_score - coalition_without_client_score
            
            self.partial_shapley[iteration][node] =  shap / (len(nodes_in_sample))

        if return_coalitions == True:
            return recorded_values
        

    # def update_shap_multip(self,
    #                        gradients: OrderedDict,
    #                        nodes_in_sample: list,
    #                        optimizer: Optimizers,
    #                        iteration: int,
    #                        previous_model: FederatedModel,
    #                        number_of_workers: int = 30,
    #                        return_coalitions: bool = True):
    #     """Method used to track_results after each training round.
    #     Update_shap_multip is a default method used to calculate
    #     Shapley round, as it uses a number of workers to complete a task.
    
    #     Given the graidnets, ids of the nodes included in sample,
    #     last version of the optimizer, previous version of the model
    #     and the updated version of the model, it calculates values of
    #     all the marginal contributions using Shapley value.
        
    #     Parameters
    #     ----------
    #     gradients: OrderedDict
    #         An OrderedDict containing gradients of the sampled nodes.
    #     nodes_in_sample: list
    #         A list containing id's of the nodes that were sampled.
    #     previous_optimizer: Optimizers
    #         An instance of the forcha.Optimizers class.
    #     iteration: int
    #         The current iteration.
    #     previous_model: FederatedModel
    #         An instance of the FederatedModel object.
    #     updated_model: FederatedModel
    #         An instance of the FederatedModel object.
    #     number_of_workers: int, default to 50
    #         A number of workers that will simultaneously work on a task.
    #     return_coalitions: bool, default to True
    #         If set to True, method will return value-mapping for every coalition.
    #     Returns
    #     -------
    #     None
    #     """
    #     coalition_results = {}
    #     nodes_in_sample = [node.node_id for node in nodes_in_sample] 
    #     superset = Subsets.form_superset(nodes_in_sample, return_dict=False)
    #     # Operations counter to track the progress of the calculations.
    #     operation_counter = 0
    #     number_of_operations = 2 ** (len(nodes_in_sample)) - 1
    #     if len(superset) < number_of_workers:
    #         number_of_workers = len(superset)
    #     chunked = chunker(seq = superset, size = number_of_workers)
    #     with Pool(number_of_workers) as pool:
    #         for chunk in chunked:
    #             results = [pool.apply_async(self.establish_value, (coalition, 
    #                                                                 copy.deepcopy(gradients),
    #                                                                 copy.deepcopy(optimizer),
    #                                                                 copy.deepcopy(previous_model))) for coalition in chunk]
    #             for result in results:
    #                 coalition, score = result.get()
    #                 coalition_results[tuple(sorted(coalition))] = score
    #             operation_counter += len(chunk)
    #             print(f"Completed {operation_counter} out of {number_of_operations} operations")
    #     print("Finished evaluating all of the coalitions. Commencing calculation of individual Shapley values.")
    #     for node in nodes_in_sample:
    #         shap = 0.0
    #         S = Subsets.select_subsets(coalitions = superset, searched_node = node)
    #         for s in S:
    #             s_wo_i = tuple(sorted(s)) # Subset s without the agent i
    #             s_copy = copy.deepcopy(s)
    #             s_copy.append(node)
    #             s_w_i = tuple(sorted(s_copy)) # Subset s with the agent i
    #             sample_pos = math.comb((len(nodes_in_sample) - 1), len(s_wo_i))
    #             divisor = float(1 / sample_pos)
    #             shap += float(divisor * (coalition_results[s_w_i] - coalition_results[s_wo_i]))
            
    #         self.partial_shapley[iteration][node] =  float(shap / (len(nodes_in_sample)))
        
    #     if return_coalitions == True:
    #         return coalition_results


    # def establish_value(self,
    #                     coalition: list,
    #                     gradients: OrderedDict,
    #                     optimizer: Optimizers,
    #                     model: FederatedModel) -> tuple[list, float]:
    #     """Helper method used to establish a value of a particular coalition.
    #     Called asynchronously in multiprocessed version of the self.update_shap_multip()
    #     method.
        
    #     Parameters
    #     ----------
    #     None
    #     gradients: OrderedDict
    #         An OrderedDict containing gradients of the sampled nodes.
    #     coalition: list
    #         A list containing id's of the nodes that were sampled.
    #     optimizer: Optimizers
    #         An instance of the forcha.Optimizers class.
    #     model: FederatedModel
    #         An instance of the FederatedModel object.
    #     Returns
    #     -------
    #     tuple[list, float]
    #     """
    #     gradients = select_gradients(gradients = gradients,
    #                                  query = coalition)
    #     grad_avg = Aggregators.compute_average(gradients)
    #     weights = optimizer.fed_optimize(weights = model.get_weights(), delta = grad_avg)
    #     model.update_weights(weights)
    #     score = model.quick_evaluate()[1]
    #     return (coalition, score)


    def calculate_final_result(self):
        """Method used to sum up all the partial Shapley values to obtain
        a final Shapley score for each client.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        tuple[dict[int: dict], dict[int: float]]
        """
        for iteration_results in self.partial_shapley.values():
            for node, value in iteration_results.items():
                self.shapley[node] += np.float64(value)
        return (self.partial_shapley, self.shapley)