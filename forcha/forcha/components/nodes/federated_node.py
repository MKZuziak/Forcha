import torch
from datasets import arrow_dataset
from forcha.models.federated_model import FederatedModel
from forcha.utils.loggers import Loggers

node_logger = Loggers.node_logger()

class FederatedNode:
    def __init__(self, 
                 node_id: int,
                 settings: dict
                 ) -> None:
        """An abstract object representing a single node in the federated training.
        
        Parameters
        ----------
        node_id: int 
            An int identifier of a node
        settings: dict
            A dictionary containing settings for the node
        
        Returns
        -------
        None
        """
        self.state = 1 # Attribute controlling the state of the object.
                         # 0 - initialized, resting
                         # 1 - initialized, in run-time
        
        self.node_id = node_id
        self.settings = settings
        self.state = 0
        self.model = None
        self.train_data = None
        self.test_data = None


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
       
        self.state = 1
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

        if self.model != None and self.train_data != None \
        and self.test_data != None:
            self.state = 0
        else:
            # TODO: LOGGING INFO
            pass
    

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