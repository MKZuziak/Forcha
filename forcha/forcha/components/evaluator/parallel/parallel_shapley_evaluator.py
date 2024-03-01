import copy
from multiprocessing import Pool
from collections import OrderedDict
from _collections_abc import Generator
import math

import numpy as np

from forcha.components.evaluator.shapley_evaluator import Sample_Shapley_Evaluator
from forcha.components.evaluator.shapley_evaluator import select_gradients
from forcha.models.federated_model import FederatedModel
from forcha.utils.optimizers import Optimizers
from forcha.utils.computations import Subsets
from forcha.utils.computations import Aggregators


def calculate_coalition_value(
    coalition: list,
    gradients: OrderedDict,
    optimizer: Optimizers,
    previous_model: OrderedDict,
    model_template: FederatedModel,
    optimizer_template: Optimizers,
    ) -> tuple[int, dict, float]:
        
        result = {}
        coalitions_gradients = select_gradients(
            gradients = gradients,
            query = coalition
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
        score = model_template.evaluate_model()[1]
        result[tuple(sorted(coalition))] = score
        
        return result


def chunker(
    seq: iter, 
    size: int
    ) -> Generator:
    """Helper function for splitting an iterable into a number
    of chunks each of size n. If the iterable can not be splitted
    into an equal number of chunks of size n, chunker will return the
    left-overs as the last chunk.
        
    Parameters
    ----------
    sqe: iter
        An iterable object that needs to be splitted into n chunks.
    size: int
        A size of individual chunk
    
    Returns
    -------
    Generator
        A generator object that can be iterated to obtain chunks of the original iterable.
    """
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))



class Parallel_Sample_Shapley_Evaluator(Sample_Shapley_Evaluator):
    def __init__(
        self, 
        nodes: list, 
        iterations: int,
        number_of_workers: int = 20
        ) -> None:
        """Constructor for the Parallel Sample Shapley Evaluator. Initializes empty
        hash tables for Shapley values for each iteration as well as hash table
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
        super().__init__(nodes, iterations)
        self.number_of_workers = number_of_workers

    
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
        None"""
        print("Calculating Shapley score in parallel")
        #Operations counter to track the progress of the calculations.
        operation_counter = 0
        number_of_operations = 2 ** (len(nodes_in_sample)) - 1
        
        # Maps every coalition to it's value, implemented to decrease the time complexity.
        recorded_values = {}        
        
        # Converting list of FederatedNode objects to the int representing their identiity.
        nodes_in_sample = [node.node_id for node in nodes_in_sample] 
        # Forming superset of all the possible coalitions.
        superset = Subsets.form_superset(nodes_in_sample, return_dict=False)
        if len(superset) < self.number_of_workers:
            self.number_of_workers = len(superset)
        chunked = chunker(
            seq = superset,
            size = self.number_of_workers
        )
        
        for chunk in chunked:
            with Pool(self.number_of_workers) as pool:
                results = [pool.apply_async(
                    calculate_coalition_value,
                        (coalition,
                        copy.deepcopy(gradients),
                        copy.deepcopy(optimizer),
                        copy.deepcopy(previous_model),
                        model_template,
                        optimizer_template))
                        for coalition in chunk]
                for result in results:
                    coalition = result.get()
                    recorded_values.update(coalition)
            operation_counter += len(chunk)
            print(f"Completed {operation_counter} out of {number_of_operations} operations")
        print("Finished evaluating all of the coalitions. Commencing calculation of individual Shapley values.")
        for node in nodes_in_sample:
            shap = 0.0
            coalitions_of_interest = Subsets.select_subsets(
                coalitions = superset,
                searched_node = node
            )
            for coalition in coalitions_of_interest:
                coalition_without_client = tuple(sorted(coalition))
                coalition_with_client = tuple(sorted(tuple((coalition)) + (node,))) #TODO: Make a more elegant solution...
                coalition_without_client_score = recorded_values[coalition_without_client]
                coalition_with_client_score = recorded_values[coalition_with_client]
                possible_combinations = math.comb((len(nodes_in_sample) - 1), len(coalition_without_client))
                divisor = 1 / possible_combinations
                shap += divisor * (coalition_with_client_score - coalition_without_client_score)
            self.partial_shapley[iteration][node] = shap / len(nodes_in_sample)
        
        if return_coalitions == True:
            return recorded_values