from forcha.components.orchestrator.generic_orchestrator import Orchestrator
from forcha.utils.orchestrations import create_nodes, sample_nodes, train_nodes
from forcha.utils.computations import Aggregators
from forcha.utils.orchestrations import create_nodes, sample_nodes, train_nodes
from forcha.utils.optimizers import Optimizers
from forcha.components.evaluator.parallel.parallel_manager import Parallel_Manager
from forcha.components.evaluator.evaluation_manager import Evaluation_Manager
from forcha.components.archiver.archive_manager import Archive_Manager
from forcha.components.settings.settings import Settings
from forcha.utils.helpers import Helpers
import torch
import datasets
import copy
from multiprocessing import Pool
import numpy as np


from multiprocessing import set_start_method
set_start_method("spawn", force=True)

def compare_for_debug(dict1, dict2):
    for (row1, row2) in zip(dict1.values(), dict2.values()):
        if False in (row1 == row2):
            return False
        else:
            return True

class Evaluator_Orchestrator(Orchestrator):
    """Orchestrator is a central object necessary for performing the simulation.
        It connects the nodes, maintain the knowledge about their state and manages the
        multithread pool. Evaluator orchestrator is a child class of the Generic Orchestrator.
        Unlike its parent, Evaluator performs a training using Federated Optimization
        - pseudo-gradients from the models and momentum. Additionally, Evaluator Orchestrator
        is able to assess clients marginal contribution with the help of Evaluation Manager."""
    
    
    def __init__(self, settings: Settings, **kwargs) -> None:
        """Orchestrator is initialized by passing an instance
        of the Settings object. Settings object contains all the relevant configurational
        settings that an instance of the Orchestrator object may need to complete the simulation.
        Evaluator Orchestrator additionaly requires a configurations passed to the Optimizer 
        and Evaluator Manager upon its initialization.
        
        Parameters
        ----------
        settings: Settings 
            An instance of the Settings object cotaining all the settings of the orchestrator.
            The Evaluator Orchestrator additionaly requires the passed object to contain a 
            configuration for the Optimizer and the Evaluation Manager.
       
       Returns
       -------
       None
        """
        super().__init__(settings, **kwargs)
    

    def train_protocol(self,
                nodes_data: list[datasets.arrow_dataset.Dataset, 
                datasets.arrow_dataset.Dataset]) -> None:
        """"Performs a full federated training according to the initialized
        settings. The train_protocol of the orchestrator.evaluator_orchestrator
        follows a popular FedAvg generalisation, FedOpt. Instead of weights from each
        clients, it aggregates gradients (understood as a difference between the weights
        of a model after all t epochs of the local training) and aggregates according to 
        provided rule. The evaluation process is menaged by the instance of the Evaluation
        Manager object, which is called upon each iteration.

        Parameters
        ----------
        nodes_data: list[datasets.arrow_dataset.Dataset, datasets.arrow_dataset.Dataset]: 
            A list containing train set and test set wrapped 
            in a hugging face arrow_dataset.Dataset containers
        
        Returns
        -------
        int
            Returns 0 on the successful completion of the training.
            """
        
        # Initializing all the attributes using an instance of the Settings object.
        iterations = self.settings.iterations # Int, number of iterations
        nodes_number = self.settings.number_of_nodes # Int, number of nodes
        nodes = [node for node in range(nodes_number)] # List of ints, list of nodes ids
        sample_size = self.settings.sample_size # Int, size of the sample
        # Initialization of the generator object    
        self.Generator = np.random.default_rng(self.settings.seed)
        # Initializing an instance of the Optimizer class object.
        optimizer_settings = self.settings.optimizer_settings # Dict containing instructions for the optimizer, dict.
        self.Optimizer = Optimizers(weights = self.central_model.get_weights(),
                                    settings=optimizer_settings)
        # Initializing an instance of the Archiver class if enabled in the settings.
        if self.settings.enable_archiver == True:
            Archive_manager = Archive_Manager(archive_manager = self.settings.archiver_settings,
                logger = self.orchestrator_logger)
        # Creating (empty) federated nodes.
        nodes_green = create_nodes(nodes, 
                                   self.settings.nodes_settings)
         # Creating a list of models for the nodes.
        model_list = self.model_initialization(nodes_number=nodes_number,
                                               model=self.central_net)
        # Initializing nodes -> loading the data and models onto empty nodes.
        nodes_green = self.nodes_initialization(nodes_list=nodes_green,
                                                model_list=model_list,
                                                data_list=nodes_data)
        
        # Initializing the Evaluation Manager
        if self.parallelization:
            Evaluation_manager = Parallel_Manager(settings = self.settings.evaluator_settings,
                                                  model_template = copy.deepcopy(self.central_model),
                                                  optimizer_template = copy.deepcopy(self.Optimizer),
                                                  nodes = nodes,
                                                  iterations = iterations)
        else:
            Evaluation_manager = Evaluation_Manager(settings = self.settings.evaluator_settings,
                                                 model_template = copy.deepcopy(self.central_model),
                                                 optimizer_template = copy.deepcopy(self.Optimizer),
                                                 nodes = nodes,
                                                 iterations = iterations)
        
        for iteration in range(iterations):
            self.orchestrator_logger.info(f"Iteration {iteration}")
            gradients = {}
            # Evaluation step: preserving the last version of the model and optimizer
            Evaluation_manager.preserve_previous_model(previous_model = copy.deepcopy(self.central_model.get_weights()))
            Evaluation_manager.preserve_previous_optimizer(previous_optimizer = copy.deepcopy(self.Optimizer.get_weights()))
            
            # Sampling nodes and asynchronously apply the function
            sampled_nodes = sample_nodes(nodes_green, 
                                         sample_size=sample_size,
                                         generator=self.Generator) # SAMPLING FUNCTION -> CHANGE IF NEEDED
            if self.batch_job:
                for batch in Helpers.chunker(sampled_nodes, size=self.batch):
                    with Pool(sample_size) as pool:
                        results = [pool.apply_async(train_nodes, (node, 'gradients')) for node in batch]
                        # consume the results
                        for result in results:
                            node_id, model_gradients = result.get()
                            gradients[node_id] = copy.deepcopy(model_gradients)
            else:
                with Pool(sample_size) as pool:
                    results = [pool.apply_async(train_nodes, (node, 'gradients')) for node in sampled_nodes]
                    # consume the results
                    for result in results:
                        node_id, model_gradients = result.get()
                        gradients[node_id] = copy.deepcopy(model_gradients)
            
            grad_copy = copy.deepcopy(gradients) #TODO Copy for the evaluation, since Agg.compute_average changes the weights
            # Computing the average
            grad_avg = Aggregators.compute_average(gradients) # AGGREGATING FUNCTION -> CHANGE IF NEEDED
            # Upadting the weights using gradients and momentum
            updated_weights = self.Optimizer.fed_optimize(weights=self.central_model.get_weights(),
                                                          delta=grad_avg)
            # Updating the orchestrator
            self.central_model.update_weights(updated_weights)
            
            # Evaluation step: preserving the updated central model
            Evaluation_manager.preserve_updated_model(updated_model = copy.deepcopy(self.central_model.get_weights()))
            # Evaluation step: calculating all the marginal contributions
            Evaluation_manager.track_results(gradients = grad_copy,
                                             nodes_in_sample = sampled_nodes,
                                             iteration = iteration)
            # Updating the nodes
            for node in nodes_green:
                node.model.update_weights(updated_weights)         
                   
            # Passing results to the archiver -> only if so enabled in the settings.
            if self.settings.enable_archiver == True:
                Archive_manager.archive_training_results(iteration = iteration,
                                                        central_model=self.central_model,
                                                        nodes=nodes_green)
            #torch.cuda.empty_cache()
            print("DEBUG CONFIRMED")
        # Evaluation step: Calling evaluation manager to preserve all steps
        results = Evaluation_manager.finalize_tracking(path = Archive_manager.metrics_savepath)
        print(results)
        self.orchestrator_logger.critical("Training complete")
        return 0