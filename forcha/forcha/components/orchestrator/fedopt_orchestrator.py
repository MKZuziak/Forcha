import datasets
import copy
from forcha.components.orchestrator.generic_orchestrator import Orchestrator
from forcha.utils.computations import Aggregators
from forcha.utils.loggers import Loggers
from forcha.utils.orchestrations import create_nodes, sample_nodes, train_nodes
from forcha.utils.optimizers import Optimizers
from forcha.components.archiver.archive_manager import Archive_Manager
from multiprocessing import Pool
from forcha.components.settings.settings import Settings
from forcha.utils.helpers import Helpers
from forcha.utils.debugger import log_gpu_memory
import numpy as np

# set_start_method set to 'spawn' to ensure compatibility across platforms.
from multiprocessing import set_start_method
set_start_method("spawn", force=True)


class Fedopt_Orchestrator(Orchestrator):
    """Orchestrator is a central object necessary for performing the simulation.
        It connects the nodes, maintain the knowledge about their state and manages the
        multithread pool. Fedopt orchestrator is a child class of the Generic Orchestrator.
        Unlike its parent, FedOpt Orchestrator performs a training using Federated Optimization
        - pseudo-gradients from the models and momentum."""
    

    def __init__(self, 
                 settings: Settings,
                 **kwargs) -> None:
        """Orchestrator is initialized by passing an instance
        of the Settings object. Settings object contains all the relevant configurational
        settings that an instance of the Orchestrator object may need to complete the simulation.
        FedOpt orchestrator additionaly requires a configuration passed to the Optimizer upon
        its initialization.
        
        Parameters
        ----------
        settings : Settings 
            An instance of the Settings object cotaining all the settings of the orchestrator.
            The FedOpt orchestrator additionaly requires the passed object to contain a 
            configuration for the Optimizer.
       
       Returns
       -------
       None
       """
        super().__init__(settings, **kwargs)
    

    def train_protocol(self) -> None:
        """"Performs a full federated training according to the initialized
        settings. The train_protocol of the fedopt.orchestrator.Fedopt_Orchestrator
        follows a popular FedAvg generalisation, FedOpt. Instead of weights from each
        clients, it aggregates gradients (understood as a difference between the weights
        of a model after all t epochs of the local training) and aggregates according to 
        provided rule.
        SOURCE: Adaptive Federated Optimization, S.J. Reddi et al.

        Parameters
        ----------
        nodes_data: list[datasets.arrow_dataset.Dataset, datasets.arrow_dataset.Dataset]: 
            A list containing train set and test set wrapped 
            in a hugging face arrow_dataset.Dataset containers.
        
        Returns
        -------
        int
            Returns 0 on the successful completion of the training.
        """
        # OPTIMIZER CLASS OBJECT
        optimizer_settings = self.settings.optimizer_settings
        self.Optimizer = Optimizers(weights = self.central_model.get_weights(),
                                    settings=optimizer_settings)

        # TRAINING PHASE ----- FEDOPT
        # FEDOPT - CREATE POOL OF WORKERS
        for iteration in range(self.iterations):
            self.orchestrator_logger.info(f"Iteration {iteration}")
            gradients = {}
            # Sampling nodes and asynchronously apply the function
            sampled_nodes = sample_nodes(self.nodes_green, 
                                         sample_size=self.sample_size,
                                         generator=self.generator) # SAMPLING FUNCTION
            # FEDAVG - TRAINING PHASE
            # OPTION: BATCH TRAINING
            if self.batch_job:
                self.orchestrator_logger.info(f"Entering batched job, size of the batch {self.batch}")
                for batch in Helpers.chunker(sampled_nodes, size=self.batch):
                    with Pool(self.sample_size) as pool:
                        results = [pool.apply_async(train_nodes, (node, 'gradients')) for node in batch]
                        # consume the results
                        for result in results:
                            node_id, model_weights = result.get()
                            gradients[node_id] = copy.deepcopy(model_weights)
            # OPTION: NON-BATCH TRAINING
            else:
                with Pool(self.sample_size) as pool:
                    results = [pool.apply_async(train_nodes, (node, 'gradients')) for node in sampled_nodes]
                    for result in results:
                        node_id, model_gradients = result.get()
                        gradients[node_id] = copy.deepcopy(model_gradients)
            # FEDOPT: AGGREGATING FUNCTION
            grad_avg = Aggregators.compute_average(gradients) # AGGREGATING FUNCTION
            updated_weights = self.Optimizer.fed_optimize(weights=self.central_model.get_weights(),
                                                          delta=grad_avg)
            # FEDOPT: UPDATING THE CENTRAL MODEL 
            self.central_model.update_weights(updated_weights)     
            # FEDOPT: UPDATING THE NODES
            for node in self.nodes_green:
                node.model.update_weights(updated_weights)    
                   
            # ARCHIVER: PRESERVING RESULTS
            if self.enable_archiver:
                self.archive_manager.archive_training_results(iteration = iteration,
                                                              central_model=self.central_model,
                                                              nodes=self.nodes_green)
            
            if self.full_debug == True:
                log_gpu_memory(iteration=iteration)

        self.orchestrator_logger.critical("Training complete")
        return 0