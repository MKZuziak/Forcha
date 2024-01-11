import copy
from forcha.components.orchestrator.generic_orchestrator import Orchestrator
from forcha.utils.optimizers import Optimizers
from forcha.utils.computations import Aggregators
from forcha.utils.orchestrations import sample_nodes, train_nodes
from forcha.components.settings.settings import Settings
from forcha.utils.debugger import log_gpu_memory
from forcha.utils.helpers import Helpers
from multiprocessing import Pool


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
                 **kwargs
                 ) -> None:
        """Orchestrator is initialized by passing an instance
        of the Settings object. Settings object contains all the relevant configurational
        settings that an instance of the Orchestrator object may need to complete the simulation.

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
            training_results = {}
        
            # Checking for connectivity
            connected_nodes = [node for node in self.network]
            # Weights dispatched before the training (if activated)
            if self.settings.dispatch_model:
                for node in connected_nodes:
                    node.model.update_weights(copy.deepcopy(self.central_model.get_weights()))
                self.orchestrator_logger.info(f"Iteration {iteration}, dispatching nodes to connected clients.")
            
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
                        results = [pool.apply_async(train_nodes, (node, iteration, 'gradients')) for node in batch]
                        for result in results:
                            node_id, model_weights, loss_list, accuracy_list = result.get()
                            gradients[node_id] = copy.deepcopy(model_weights)
                            training_results[node_id] = {
                                "iteration": iteration,
                                "node_id": node_id,
                                "loss": loss_list[-1], 
                                "accuracy": accuracy_list[-1]}
            # OPTION: NON-BATCH TRAINING
            else:
                with Pool(self.sample_size) as pool:
                    results = [pool.apply_async(train_nodes, (node, iteration, 'gradients')) for node in sampled_nodes]
                    for result in results:
                            node_id, model_weights, loss_list, accuracy_list = result.get()
                            gradients[node_id] = copy.deepcopy(model_weights)
                            training_results[node_id] = {
                                "iteration": iteration,
                                "node_id": node_id,
                                "loss": loss_list[-1], 
                                "accuracy": accuracy_list[-1]}
            # FEDOPT: AGGREGATING FUNCTION
            grad_avg = Aggregators.compute_average(copy.deepcopy(gradients)) # AGGREGATING FUNCTION
            # ARCHIVER: PRESERVING TRAINING ON NODES RESULTS
            if self.enable_archiver == True:
                self.archive_manager.archive_training_results(
                    iteration = iteration,
                    results=training_results
                )
                self.archive_manager.archive_local_test_results(
                    iteration = iteration,
                    nodes = sampled_nodes
                )
                        
            updated_weights = self.Optimizer.fed_optimize(weights=copy.deepcopy(self.central_model.get_weights()),
                                                          delta=copy.deepcopy(grad_avg))
            # FEDOPT: UPDATING THE NODES
            for node in connected_nodes:
                node.model.update_weights(copy.deepcopy(updated_weights))
            # FEDOPT: UPDATING THE CENTRAL MODEL 
            self.central_model.update_weights(copy.deepcopy(updated_weights))
                   
            # ARCHIVER: PRESERVING RESULTS
            if self.enable_archiver == True:
                self.archive_manager.archive_testing_results(
                    iteration = iteration,
                    central_model=self.central_model,
                    nodes=connected_nodes)
            
            if self.full_debug == True:
                log_gpu_memory(iteration=iteration)

        self.orchestrator_logger.critical("Training complete")
        return 0