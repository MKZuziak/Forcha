import torch
from numpy import array
from numpy.random import default_rng
from datasets import arrow_dataset
from forcha.models.federated_model import FederatedModel
from forcha.utils.loggers import Loggers
from forcha.utils.helpers import find_nearest

node_logger = Loggers.node_logger()

class FederatedNode:
    def __init__(self, 
                 node_id: int,
                 settings: dict,
                 group: int = None,
                 seed: float | int = 42
                 ) -> None:
        """An abstract object representing a single node in the federated training.
        
        Parameters
        ----------
        node_id: int 
            An int identifier of a node
        settings: dict
            A dictionary containing settings for the node
        group: int, default to None
            A group assigned to a device. Important for performing MCFC simulations, default to None.
        seed: float or int, default to 42
            The seed assigned to a particular Random Generator. Important for performing MCFC simulations, default to 42.
        Returns
        -------
        None
        """
        self.state = None # Attribute controlling the state of the object.
        self.node_id = node_id
        self.settings = settings
        self.model = None
        self.train_data = None
        self.test_data = None
        
        self.transition_matrix = None # Transition matrix, important for performing MCFL.
        self.group = group
        self.generator = default_rng(seed=(seed + node_id))
            
            
    def load_transition_matrix(self,
                               transition_matrix: array) -> None:
        """Loads the transition matrix for performing
        MCFL simulation. 
        
        Parameters
        ----------
        transition_matrix: array
            Transition matrix of the states used during performing MCFL simulation.
        Returns
        ------------
        None
        """
        self.transition_matrix = transition_matrix
        self.states = [i for i in range(self.transition_matrix.shape[0])]
    
    
    def update_state(self,
                     iteration: int) -> None:
        """Updates state of the node
        ------------
        Arguments:
        iteration: int
            Current Iteration of the training.
        ------------
        Returns:
            None"""
        if self.transition_matrix.any():
            self.state = self.generator.choice(a=self.states, p=self.transition_matrix[self.state, :])
            node_logger.info(f"[ITERATION {iteration} | NODE {self.node_id}] transitioned to state {self.state}")
        else:
            self.state = 1
        # TODO: Update the state accordingly


    def prepare_node(self, 
                     model: torch.nn.Module, 
                     data: arrow_dataset.Dataset,
                     save_model: bool = False,
                     save_path: str = None
                     ) -> None:
        """Prepares node for the training, given the passed model 
        and dataset.
        
        Parameters
        ----------
        model: nn.Module 
            The Neural Network architecture that we want to use.
        dsata: arrow_dataset.Dataset 
            The local dataset that will be used with this set.
        save_model: bool (default to False)
            A boolean flag enabling to save model
        save_path: str (default to None)
            A path to the directory in which model should be saved.
        
        Returns
        -------
        None
        """
       
        self.train_data = data[0]
        self.test_data = data[1]
        self.save_model = save_model
        self.save_path = save_path
        self.model = FederatedModel(
            settings=self.settings["model_settings"],
            net = model,
            local_dataset = data,
            node_name=self.node_id
        )
        self.state = 0


    def train_local_model(self,
                          iteration: int,
                          mode: str) -> tuple[list[float], list[float], list[float]]:
        """This function starts the server phase of the federated learning.
        In particular, it trains the model locally and then sends the weights.
        Then the updated weights are received and used to update
        the local model.
        
        Parameters
        ----------
        node: FederatedNode 
            Node that we want to train.
        mode: str 
            Mode of the training. 
            Mode = 'weights': Node will return model's weights.
            Mode = 'gradients': Node will return model's gradients.
        
        Returns
        -------
            Tuple[List[float], List[float], List[float]]: _description_
        """
        node_logger.info(f"[ITERATION {iteration} | NODE {self.node_id}] Starting training on node {self.node_id}")
        loss_list: list[float] = []
        accuracy_list: list[float] = []

        local_epochs = self.settings['local_epochs']

        if mode == 'gradients':
            self.model.preserve_initial_model()
        for epoch in range(local_epochs):
            metrics = self.local_training(
                iteration=iteration, 
                epoch=epoch
                )
            loss_list.append(metrics["loss"])
            accuracy_list.append(metrics["accuracy"])
        if self.save_model:
            self.model.store_model_on_disk(
                iteration=iteration, 
                path=self.save_path
                )
        
        node_logger.info(f"[ITERATION {iteration} | NODE {self.node_id}] Results of training on node {self.node_id}: {accuracy_list}")
        if mode == 'weights:':
            return (
                self.node_id,
                self.model.get_weights()
                )
        elif mode == 'gradients':
            return (
                self.node_id,
                self.model.get_gradients()
                )
        else:
            node_logger.info(f"[ITERATION {iteration} | NODE {self.node_id}] No mode was provided, returning only model's weights")
            return (
                self.node_id,
                self.model.get_weights()
                )


    def local_training(self,
                       iteration: int,
                       epoch: int
                       ) -> dict[int, int]:
        """Helper method for performing one epoch of local training.
        Performs one round of Federated Training and pack the
        results (metrics) into the appropiate data structure.
        
        Parameters
        ----------
    
        Returns
        -------
            dict[int, int]: metrics from the training.
        """
        loss, accuracy = self.model.train(iteration=iteration, epoch=epoch)
        return {"loss": loss, "accuracy": accuracy}