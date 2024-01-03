import copy
from forcha.components.evaluator.parallel.parallel_manager import Parallel_Manager
from forcha.components.evaluator.evaluation_manager import Evaluation_Manager
from forcha.components.orchestrator.generic_orchestrator import Orchestrator
from forcha.utils.optimizers import Optimizers
from forcha.utils.computations import Aggregators
from forcha.utils.orchestrations import sample_nodes, train_nodes
from forcha.components.settings.settings import Settings
from forcha.utils.debugger import log_gpu_memory
from forcha.utils.helpers import Helpers
from multiprocessing import Pool

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
    
    
    def __init__(self, 
                 settings: Settings, 
                 **kwargs
                 ) -> None:
        """Orchestrator is initialized by passing an instance
        of the Settings object. Settings object contains all the relevant configurational
        settings that an instance of the Orchestrator object may need to complete the simulation.
        Evaluator Orchestrator additionaly requires a configurations passed to the Optimizer 
        and Evaluator Manager upon its initialization.
        
        Parameters
        ----------
        settings : Settings
            An instance of the settings object cotaining all the settings 
            of the orchestrator.
        **kwargs : dict, optional
            Extra arguments to enable selected features of the Orchestrator.
            passing full_debug to **kwargs, allow to enter a full debug mode.

        Returns
        -------
        None
        """
        super().__init__(settings, **kwargs)
    

    def train_protocol(self) -> None:
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
        # OPTIMIZER CLASS OBJECT
        optimizer_settings = self.settings.optimizer_settings # Dict containing instructions for the optimizer, dict.
        self.Optimizer = Optimizers(weights = self.central_model.get_weights(),
                                    settings=optimizer_settings)
        # EVALUATION MANAGER: INITIALIZAITON
        if self.parallelization:
            Evaluation_manager = Parallel_Manager(settings = self.settings.evaluator_settings,
                                                  model_template = copy.deepcopy(self.central_model),
                                                  optimizer_template = copy.deepcopy(self.Optimizer),
                                                  nodes = self.nodes,
                                                  iterations = self.iterations)
        else:
            Evaluation_manager = Evaluation_Manager(settings = self.settings.evaluator_settings,
                                                 model_template = copy.deepcopy(self.central_model),
                                                 optimizer_template = copy.deepcopy(self.Optimizer),
                                                 nodes = self.nodes,
                                                 iterations = self.iterations)
        
        # TRAINING PHASE ----- FEDOPT WITH EVALUATOR
        # FEDOPT - CREATE POOL OF WORKERS
        for iteration in range(self.iterations):
            self.orchestrator_logger.info(f"Iteration {iteration}")
            gradients = {}
            
            # Checking for connectivity
            connected_nodes = self.update_connectivity()
            if len(connected_nodes) < self.sample_size:
                self.orchestrator_logger.warning(f"Not enough connected nodes to draw a full sample! Skipping an iteration {iteration}")
                continue
            else:
                self.orchestrator_logger.info(f"Nodes connected at round {iteration}: {[node.node_id for node in connected_nodes]}")
            
            # Weights dispatched before the training (if activated)
            if self.settings.dispatch_model:
                for node in connected_nodes:
                    node.model.update_weights(copy.deepcopy(self.central_model.get_weights()))
                self.orchestrator_logger.info(f"Iteration {iteration}, dispatching nodes to connected clients.")
            
            
            # EVALUATION MANAGER: preserving the last version of the model and optimizer
            Evaluation_manager.preserve_previous_model(previous_model = copy.deepcopy(self.central_model.get_weights()))
            Evaluation_manager.preserve_previous_optimizer(previous_optimizer = copy.deepcopy(self.Optimizer.get_weights()))
            # Sampling nodes and asynchronously apply the function
            sampled_nodes = sample_nodes(
                connected_nodes, 
                sample_size=self.sample_size,
                generator=self.generator
                ) # SAMPLING FUNCTION
            # FEDOPT - TRAINING PHASE
            # OPTION: BATCH TRAINING
            if self.batch_job:
                self.orchestrator_logger.info(f"Entering batched job, size of the batch {self.batch}")
                for batch in Helpers.chunker(sampled_nodes, size=self.batch):
                    with Pool(len(list(batch))) as pool:
                        results = [pool.apply_async(train_nodes, (node, 'gradients')) for node in batch]
                        for result in results:
                            node_id, model_gradients = result.get()
                            gradients[node_id] = copy.deepcopy(model_gradients)
            # OPTION: NON-BATCH TRAINING
            else:
                with Pool(self.sample_size) as pool:
                    results = [pool.apply_async(train_nodes, (node, iteration, 'gradients')) for node in sampled_nodes]
                    for result in results:
                        node_id, model_gradients = result.get()
                        gradients[node_id] = copy.deepcopy(model_gradients)
            # EVALUATOR: MAKE COPIES OF THE GRADIENTS
            grad_copy = copy.deepcopy(gradients) #TODO Copy for the evaluation, since Agg.compute_average changes the weights
            # FEDOPT: AGGREGATING FUNCTION
            grad_avg = Aggregators.compute_average(gradients) # AGGREGATING FUNCTION -> CHANGE IF NEEDED
            updated_weights = self.Optimizer.fed_optimize(weights=self.central_model.get_weights(),
                                                          delta=grad_avg)
            # FEDOPT: UPDATING THE CENTRAL MODEL 
            self.central_model.update_weights(copy.deepcopy(updated_weights))
            # EVALUATOR: PRESERVE UPDATED MODEL
            Evaluation_manager.preserve_updated_model(
                updated_model = copy.deepcopy(self.central_model.get_weights()))
            # EVALUATOR: TRACK RESULTS
            Evaluation_manager.track_results(gradients = grad_copy,
                                             nodes_in_sample = sampled_nodes,
                                             iteration = iteration)
            # FEDOPT: UPDATING THE NODES
            for node in self.nodes_green:
                node.model.update_weights(copy.deepcopy(updated_weights))         
                   
            # ARCHIVER: PRESERVING RESULTS
            if self.enable_archiver == True:
                self.archive_manager.archive_training_results(iteration = iteration,
                                                              central_model=self.central_model,
                                                              nodes=self.nodes_green)
        # EVALUATOR: PRESERVE RESULTS
        results = Evaluation_manager.finalize_tracking(path = self.archive_manager.metrics_savepath)
        self.orchestrator_logger.critical("Training complete")
        return 0