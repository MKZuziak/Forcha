import datasets 
import copy
from forcha.components.nodes.federated_node import FederatedNode
from forcha.models.federated_model import FederatedModel
from forcha.utils.computations import Aggregators
from forcha.utils.loggers import Loggers
from forcha.utils.orchestrations import create_nodes, check_health, sample_nodes, train_nodes
from forcha.components.archiver.archive_manager import Archive_Manager
from forcha.components.settings.settings import Settings
from forcha.utils.debugger import log_gpu_memory
from forcha.utils.helpers import Helpers
import numpy as np
from multiprocessing import Pool
from torch import nn
from typing import Union

# set_start_method set to 'spawn' to ensure compatibility across platforms.
from multiprocessing import set_start_method
set_start_method("spawn", force=True)


class Orchestrator():
    """Orchestrator is a central object necessary for performing the simulation.
        It connects the nodes, maintain the knowledge about their state and manages the
        multithread pool. generic_orchestrator.orchestrator is a parent to all more
        specific orchestrators.
        
        Attributes
        ----------
        settings: Forcha.settings.Settings
            A settings object containing all the settings used during the simulation.
        net: nn.Module
            A template of the neural network architecture that is loaded into FederatedModel.
        central_model: forcha.models.federated_model.FederatedModel
            A FederatedModel object attached to the orchestrator.
        orchestrator_logger: forcha.utils.loggers.Loggers
            A pre-configured logger attached to the Orchestrator.
        validation_data: datasets.arrow_dataset.Dataset
            A datasets.arrow_dataset.Dataset with validation data for the Orchestrator.
        full_debug: Bool
            A boolean flag enabling full debug mode of the orchestrator (default to False).
        batch_job: Bool
            A boolean flag diasbling simultaneous training of the clients (default to False).
        (optional) batch: int
            If batch_job is set to True, this will be a number of clients allowed
            to train simultaneously.
        parallelization: Bool
            A boolean flag enabling parallelization of certain operations (default to False)
        generator: np.random.default_rng
            A random number generator attached to the Orchestrator.
        """
    
    
    def __init__(self, 
                 settings: Settings,
                 **kwargs) -> None:
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
        self.settings = settings
        # Special option to enter a full debug mode.
        if kwargs.get("full_debug"):
            self.full_debug = True
        else:
            self.full_debug = False
        # Batch job enabled or disabled
        if kwargs.get("batch_job"):
            self.batch_job = True
            self.batch = kwargs["batch"]
        else:
            self.batch_job = False
        # Parallelization enabled or disabled
        if kwargs.get("parallelization"):
            self.parallelization = True
        else:
            self.parallelization = False
        self.orchestrator_logger = Loggers.orchestrator_logger()
        
        # Initialization of the generator object    
        self.generator = np.random.default_rng(self.settings.seed)
    
    
    def prepare_orchestrator(self, 
                             model: nn,
                             validation_data: datasets.arrow_dataset.Dataset,
                             ) -> None:
        """Loads the orchestrator's test data and creates an instance
        of the Federated Model object that will be used throughout the training.
        
        Parameters
        ----------
        validation_data : datasets.arrow_dataset.Dataset:
            Validation dataset that will be used by the Orchestrator.
        model : torch.nn
            Model architecture that will be used throughout the training.
        Returns
        -------
        None"""
        self.validation_data = [validation_data]
        self.central_net = model
        self.central_model = FederatedModel(settings = self.settings.model_settings,
                                        net=model,
                                        local_dataset=self.validation_data,
                                        node_name='orchestrator')
    

    def model_initialization(self,
                             nodes_number: int,
                             model: Union[nn.Module, list[nn.Module]],
                             local_warm_start: bool = False,
                             ) -> list[nn.Module]:
        """Creates a list of neural nets (not FederatedModels) that will be
        passed onto the nodes and converted into FederatedModels. If local_warm_start
        is set to True, the method call should be passed a list of models which
        length is equall to the number of nodes.
        
        Parameters
        ----------
        nodes_number: int 
            number of nodes that will participate in the training.
        model: Union[nn.Module, list[nn.Module]] 
            a neural net schematic (if warm start is set to False) or 
            a number of different neural net schematics
            (if warm start is set to True) that 
            are prepared for the nodes to be loaded as FederatedModels.
        local_warm_start: bool, default False
            boolean value for switching on/off the warm start utility.
        
        Returns
        -------
        list[nn.Module]
            returns a list containing an instances of torch.nn.Module class.
        
        Raises
        ------
        NotImplemenetedError
            If local_warm_start is set to True.
        """
        if local_warm_start == True:
            raise NotImplementedError("Local warm start is not implemented yet.")
        else:
            # Deep copy is nec. because the models will have different (non-shared) parameters
            model_list = [copy.deepcopy(model) for _ in range(nodes_number)]
        return model_list


    def nodes_initialization(self,
                             nodes_list: list[FederatedNode],
                             model_list: list[nn.Module],
                             data_list: list[datasets.arrow_dataset.Dataset, 
                                    datasets.arrow_dataset.Dataset]
                                    ) -> list[FederatedNode]:
        """Prepare instances of a FederatedNode object for a participation in 
        the Federated Training.  Contrary to the 'create nodes' function, 
        it accepts only already initialized instances of the FederatedNode
        object.
        
        Parameters
        ----------
            nodess_list: list[FederatedNode] 
                The list containing all the initialized FederatedNode instances.
            model_list: list[nn.Module] 
                The list containing all the initialized nn.Module objects. 
                Note that conversion from nn.Module into the FederatedModel will occur 
                at the local node level.
            data_list (list[..., ....]): 
                The list containing train set and test set 
                wrapped in a hugging facr arrow_dataset.Dataset containers.
        
        Raises
        ------
            list[FederatedNode]"""
        
        results = []
        for node, model, dataset in zip(nodes_list, model_list, data_list):
            node.prepare_node(model, dataset)
            results.append(node)
        nodes_green = []
        for result in results:
            if check_health(result,
                            orchestrator_logger=self.orchestrator_logger):
                nodes_green.append(result)
        return nodes_green # Returning initialized nodes
    
    
    def prepare_training(self,
                         nodes_data: list[datasets.arrow_dataset.Dataset,
                                          datasets.arrow_dataset.Dataset]) -> None:
        """Prepares all the necessary elements of the training, including nodes and helpers.
        Must be run before the train_protocol method is invoked.
        
        Parameters
        ----------
        nodes_data: list[datasets.arrow_dataset.Dataset, datasets.arrow_dataset.Dataset] 
            A list containing train set and test set
            wrapped in a hugging face arrow_dataset.Dataset containers.
        -------------
        Returns:
        Int
            Returns 0 on the successful completion of the training."""
        
        self.iterations = self.settings.iterations
        self.nodes_number = self.settings.number_of_nodes
        self.local_warm_start = self.settings.local_warm_start
        self.sample_size = self.settings.sample_size
        self.nodes = [node for node in range(self.nodes_number)]
        self.enable_archiver = self.settings.enable_archiver
        
        # Initializing an instance of the Archiver class if enabled in the settings.
        if self.enable_archiver:
            self.archive_manager = Archive_Manager(
                archive_manager = self.settings.archiver_settings,
                logger = self.orchestrator_logger
            )
        
        # Creating nodes
        # Creating (empty) federated nodes
        self.nodes_green = create_nodes(self.nodes,
                                        self.settings.nodes_settings)
        
        self.model_list = self.model_initialization(nodes_number=self.nodes_number,
                                                    model=self.central_net)
        
        self.nodes_green = self.nodes_initialization(nodes_list=self.nodes_green,
                                                model_list=self.model_list,
                                                data_list=nodes_data)      

    def train_protocol(self) -> None:
        """Performs a full federated training according to the initialized
        settings. The train_protocol of the generic_orchestrator.Orchestrator
        follows a classic FedAvg algorithm - it averages the local weights
        and aggregates them taking a weighted average.
        SOURCE: Communication-Efficient Learning of
        Deep Networks from Decentralized Data, H.B. McMahan et al.

        Parameters
        ----------
        -------------
        Returns:
        Int
            Returns 0 on the successful completion of the training."""
    # TRAINING PHASE ----- FEDAVG
        # FEDAVG - CREATE POOL OF WORKERS
        for iteration in range(self.iterations):
            self.orchestrator_logger.info(f"Iteration {iteration}")
            weights = {}
            # Sampling nodes and asynchronously apply the function
            sampled_nodes = sample_nodes(self.nodes_green, 
                                         sample_size=self.sample_size,
                                         generator=self.generator) # SAMPLING FUNCTION
            # FEDAVG - TRAINING PHASE
            # OPTION: BATCH TRAINING
            if self.batch_job:
                for batch in Helpers.chunker(sampled_nodes, size=self.batch):
                    with Pool(self.sample_size) as pool:
                        results = [pool.apply_async(train_nodes, (node,)) for node in batch]
                        for result in results:
                            node_id, model_weights = result.get()
                            weights[node_id] = model_weights
            # OPTION: NON-BATCH TRAINING
            else:
                with Pool(self.sample_size) as pool:
                    results = [pool.apply_async(train_nodes, (node,)) for node in sampled_nodes]
                    for result in results:
                        node_id, model_weights = result.get()
                        weights[node_id] = model_weights
            # FEDAVG: AGGREGATING FUNCTION
            avg = Aggregators.compute_average(weights) # AGGREGATING FUNCTION
            # FEDAVG: UPDATING THE NODES
            for node in self.nodes_green:
                node.model.update_weights(avg)
            # FEDAVG: UPDATING THE CENTRAL MODEL 
            self.central_model.update_weights(avg)

            # ARCHIVER: PRESERVING RESULTS
            if self.settings.enable_archiver == True:
                self.archive_manager.archive_training_results(iteration = iteration,
                                                              central_model=self.central_model,
                                                              nodes=self.nodes_green)
            if self.full_debug == True:
                log_gpu_memory(iteration=iteration)

        self.orchestrator_logger.critical("Training complete")
        return 0
                        
