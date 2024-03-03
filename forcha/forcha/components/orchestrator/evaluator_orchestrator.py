import copy
from multiprocessing import Pool, set_start_method

# from forcha.components.evaluator.parallel.parallel_manager import Parallel_Manager
from forcha.components.evaluator.evaluation_manager import Evaluation_Manager
from forcha.components.evaluator.parallel.parallel_manager import Parallel_Manager
from forcha.components.orchestrator.generic_orchestrator import Orchestrator
from forcha.utils.optimizers import Optimizers
from forcha.utils.computations import Aggregators
from forcha.utils.orchestrations import sample_nodes, train_nodes
from forcha.components.settings.settings import Settings
from forcha.utils.debugger import log_gpu_memory
from forcha.utils.helpers import Helpers
from forcha.utils.handlers import save_csv_file, save_model_metrics, save_training_metrics

# Set start method set to spawn to ensure cross-platform compatibility.
set_start_method("spawn", force=True)


class Evaluator_Orchestrator(Orchestrator):
    """Orchestrator is a central object necessary for performing the simulation.
        It connects the nodes, maintain the knowledge about their state and manages the
        multithread pool. Evaluator orchestrator is a child class of the Generic Orchestrator.
        Unlike its parent, Evaluator performs a training using Federated Optimization
        - pseudo-gradients from the models and momentum. Additionally, Evaluator Orchestrator
        is able to assess clients marginal contribution with the help of Evaluation Manager."""
    
    
    def __init__(
        self, 
        settings: Settings,
        number_of_workers: int = 20,
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
        super().__init__(
            settings, 
            number_of_workers,
            **kwargs)
    

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
        # BEGINING OF TRAINING
        ########################################################
        ########################################################
        ########################################################
        
        ########################################################
        # FEDOPT - CREATE OPTIMIZER INSTANCE
        self.optimizer = Optimizers(
            weights = self.central_model.get_weights(),
            settings=self.settings
            )
        ########################################################
        
        ########################################################
        # FEDOPT EVALUATOR - CREATE EVALUATION MANAGER INSTANCE
        if self.parallelization:
            self.evaluation_manager = Parallel_Manager(
                settings = self.settings,
                model_template = self.central_model,
                optimizer_template = self.optimizer,
                nodes = [node.node_id for node in self.network],
                iterations = self.iterations,
                full_debug = self.full_debug,
                number_of_workers = self.number_of_workers)
        else:
            self.evaluation_manager = Evaluation_Manager(
                settings = self.settings,
                model_template = self.central_model,
                optimizer_template = self.optimizer,
                nodes = [node.node_id for node in self.network],
                iterations = self.iterations,
                full_debug = self.full_debug
                )
        ########################################################
        
        # TRAINING PHASE ----- FEDOPT WITH EVALUATOR
        for iteration in range(self.iterations):
            # BEGINING OF ITERATION
            ########################################################
            ########################################################
            self.orchestrator_logger.info(f"Iteration {iteration}")
           
            ########################################################
            # FEDOPT - INIT PHASE
            gradients = {}
            training_results = {}       
            # Checking for connectivity
            connected_nodes = [node for node in self.network]
            # Weights dispatched before the training (if activated)
            self.orchestrator_logger.info(f"Iteration {iteration}, dispatching nodes to connected clients.")
            for node in connected_nodes:
                node.model.update_weights(self.central_model.get_weights())
            ########################################################
            
            ########################################################
            # FEDOPT EVALUATOR - PRESERVING PHASE
            self.evaluation_manager.preserve_previous_model(previous_model = self.central_model.get_weights())
            self.evaluation_manager.preserve_previous_optimizer(previous_optimizer = self.optimizer.get_weights())
            ########################################################
            
            ########################################################
            # FEDOPT - SAMPLING PHASE
            sampled_nodes = sample_nodes(
                connected_nodes, 
                sample_size=self.sample_size,
                generator=self.generator
                )
            ########################################################
            
            ########################################################
            # FEDOPT - TRAINING PHASE
            # FEDOPT OPTION I: BATCH TRAINING
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
                                "accuracy": accuracy_list[-1]
                                }
            # FEDOPT OPTION II: BATCH TRAINING
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
                            "accuracy": accuracy_list[-1]
                            }
            ########################################################
           
            ########################################################
            # FEDOPT EVALUATOR - PRESERVING PHASE
            grad_copy = copy.deepcopy(gradients)
            ########################################################
            
            ########################################################
            # FEDOPT - TESTING RESULTS BEFORE THE MODEL UPDATE PHASE
            if self.settings.save_training_metrics:
                save_training_metrics(
                    file = training_results,
                    saving_path = self.settings.results_path,
                    file_name = "training_metrics.csv"
                    )
            # METRICS: TEST RESULTS ON NODES (TRAINED MODEL)
                for node in sampled_nodes:
                    save_model_metrics(
                        iteration = iteration,
                        model = node.model,
                        logger = self.orchestrator_logger,
                        saving_path = self.settings.results_path,
                        file_name = 'local_model_on_nodes.csv'
                        )
            ########################################################
            
            ########################################################
            # FEDOPT - AGGREGATION AND CENTRAL UPDATE PHASE
            grad_avg = Aggregators.compute_average(gradients) # AGGREGATING FUNCTION -> CHANGE IF NEEDED
            updated_weights = self.optimizer.fed_optimize(
                weights=copy.deepcopy(self.central_model.get_weights()),
                delta=copy.deepcopy(grad_avg)) 
            self.central_model.update_weights(copy.deepcopy(updated_weights))
            ########################################################
            
            ########################################################
            # FEDOPT EVALUATOR - PRESERVE UPDATED MODEL AND TRACK RESULTS
            self.evaluation_manager.preserve_updated_model(
                updated_model = copy.deepcopy(self.central_model.get_weights()))
            # EVALUATOR: TRACK RESULTS
            self.evaluation_manager.track_results(
                gradients = grad_copy,
                nodes_in_sample = sampled_nodes,
                iteration = iteration)
            ########################################################
            
            ########################################################
            # FEDOPT - UPDATING THE NODES AND SAVE RESULTS
            for node in connected_nodes:
                node.model.update_weights(copy.deepcopy(updated_weights))       
            if self.settings.save_training_metrics:
                save_model_metrics(
                    iteration = iteration,
                    model = self.central_model,
                    logger = self.orchestrator_logger,
                    saving_path = self.settings.results_path,
                    file_name = "global_model_on_orchestrator.csv"
                )
                for node in connected_nodes:
                    save_model_metrics(
                        iteration = iteration,
                        model = node.model,
                        logger = self.orchestrator_logger,
                        saving_path = self.settings.results_path,
                        file_name = "global_model_on_nodes.csv")
            if self.settings.save_central_model:
                self.central_model.store_model_on_disk(
                    iteration=iteration,
                    path=self.settings.orchestrator_model_path
                )
            ########################################################
            
            ########################################################
            ########################################################
            # END OF ITERATION
        
        ########################################################
        # FEDOPT EVALUATOR - PRESERVE FINAL RESULTS
        self.evaluation_manager.finalize_tracking(path=self.settings.results_path)
        ########################################################
        
        if self.full_debug == True:
            log_gpu_memory(iteration=iteration)    
        # EVALUATOR: PRESERVE RESULTS
        self.orchestrator_logger.critical("Training complete")
        return 0