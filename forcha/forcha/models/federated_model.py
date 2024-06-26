# Libraries imports
import torch, copy
import numpy as np
from datasets import arrow_dataset
from collections import OrderedDict
# Modules imports
from collections import Counter
from typing import Any, Generic, Mapping, TypeVar, Union
#from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch import nn, optim
from torchvision import transforms
from sklearn.metrics import f1_score, recall_score, confusion_matrix, precision_score
import os
from forcha.exceptions.modelexception import ModelException
from forcha.utils.loggers import Loggers
from forcha.components.settings.settings import Settings

model_logger = Loggers.model_logger()

class FederatedModel:
    """This class is used to encapsulate the (PyTorch) federated model that
    we will train. It accepts only the PyTorch models and 
    provides a utility functions to initialize the model, 
    retrieve the weights or perform an indicated number of traning
    epochs.
    """
    def __init__(
        self,
        settings: Settings,
        net: nn.Module,
        local_dataset: list[arrow_dataset.Dataset, arrow_dataset.Dataset] | list[arrow_dataset.Dataset],
        node_name: int
        ) -> None:
        """Initialize the Federated Model. This model will be attached to a 
        specific client and will wait for further instructions.
        
        Parameters
        ----------
        settings: dict 
            The settings of the local node.
        net: nn.Module 
            The Neural Network architecture that we want to use.
        local_dataset: list 
            The local dataset that will be used with this set.
        node_name: int 
            An identifier for the node that uses this container.
        
        Returns
        -------
        None
        """
        # FORCE CPU IF ENABLED
        if hasattr(settings, 'force_cpu'):
            self.device = torch.device('cpu')
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu = torch.device("cpu")
        self.initial_model = None
        self.optimizer = None  
        self.net = copy.deepcopy(net) # Do we need to create a deepcopy?
        self.settings = settings
        self.node_name = node_name
        
        # If both, train and test data were provided
        if len(local_dataset) == 2:
            self.trainloader, self.testloader = self.prepare_data(local_dataset)
        # If only a test dataset was provided.
        elif len(local_dataset) == 1:
            self.testloader = self.prepare_data(local_dataset, only_test=True)
        else:
            raise ModelException("The provided dataset object seem to be wrong. Please provide list[train_set, test_set] or list[test_set]")

        # List containing all the parameters to update
        params_to_update = []
        for _, param in self.net.named_parameters():
            if param.requires_grad is True:
                params_to_update.append(param)


        # Choosing an optimizer based on settings
        # ADAM
        if self.settings.optimizer == "Adam":
            if hasattr(self.settings, 'adam_betas'):
                betas = settings.adam_betas
            else:
                betas = (0.9, 0.999)
            
            if hasattr(self.settings, 'weight_decay'):
                weight_decay = settings.weight_decay
            else:
                weight_decay = 0

            if hasattr(self.settings, 'amsgrad'):
                amsgrad = True
            else:
                amsgrad = False
            
            self.optimizer = optim.Adam(
                params_to_update,
                lr = self.settings.learning_rate,
                betas=betas,
                weight_decay=weight_decay,
                amsgrad=amsgrad
            )
        
        
        # SGD
        elif self.settings.optimizer == "SGD":
            if hasattr(self.settings, 'momentum'):
                momentum = settings.momentum
            else:
                momentum = 0
            
            if hasattr(self.settings, 'weight_decay'):
                weight_decay = settings.weight_decay
            else:
                weight_decay = 0

            if hasattr(self.settings, 'dampening'):
                dampening = settings.dampening
            else:
                dampening = 0

            if hasattr(self.settings, 'nesterov'):
                nesterov = settings.nesterov
            else:
                nesterov = False
            
            self.optimizer = optim.SGD(
                params_to_update,
                lr = self.settings.learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                dampening=dampening,
                nesterov=nesterov
            )
        
        
        # RMS
        elif self.settings.optimizer == "RMS":
            if hasattr(self.settings, 'momentum'):
                momentum = settings.momentum
            else:
                momentum = 0
            if hasattr(self.settings, 'alpha'):
                alpha = settings.alpha
            else:
                alpha = 0.99
            if hasattr(self.settings, 'weight_decay'):
                weight_decay = settings.weight_decay
            else:
                weight_decay = 0

            self.optimizer = optim.RMSprop(
                params_to_update,
                lr=self.settings.learning_rate,
                alpha=alpha,
                weight_decay=weight_decay)
        else:
            raise ModelException("The provided optimizer name may be incorrect or not implemented.\
            Please provide list[train_set, test_set] or list[test_set]")
        

    def prepare_data(
        self,
        local_dataset: list[arrow_dataset.Dataset, arrow_dataset.Dataset] | list[arrow_dataset.Dataset],
        only_test: bool = False
        ) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Convert training and test data stored on the local client into
        torch.utils.data.DataLoader.
        
        Parameters
        ----------
        local_dataset: list[...] 
            A local dataset that should be loaded into DataLoader
        only_test: bool [default to False]: 
            If true, only a test set will be returned
        
        Returns
        -------------
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: training and test set or
        Tuple[torch.utils.data.DataLoader]: test set, if only_test == True.
        """
        if only_test == False:
            local_dataset[0] = local_dataset[0].with_transform(self.transform_func)
            local_dataset[1] = local_dataset[1].with_transform(self.transform_func)
            batch_size = self.settings.batch_size
            trainloader = torch.utils.data.DataLoader(
                local_dataset[0],
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
            )

            testloader = torch.utils.data.DataLoader(
                local_dataset[1],
                batch_size=16,
                shuffle=False,
                num_workers=0,
            )
            #self.print_data_stats(trainloader) #TODO
            return trainloader, testloader
        else:
            local_dataset[0] = local_dataset[0].with_transform(self.transform_func)
            testloader = torch.utils.data.DataLoader(
                local_dataset[0],
                batch_size=16,
                shuffle=False,
                num_workers=0,
            )
            return testloader


    def print_model_footprint(self) -> None:
        """Prints all the information about the model..
        Args:
        """
        unique_hash = hash(next(iter(self.trainloader))['image'] + self.node_name)
        
        string = f"""
        model id: {self.node_name}
        device: {self.device},
        optimizer: {self.optimizer},
        unique hash: {unique_hash}
        """
        return (self.node_name, self.device, self.optimizer, unique_hash)
        
        # num_examples = {
        #     "trainset": len(self.training_set),
        #     "testset": len(self.test_set),
        # }
        # targets = []
        # for _, data in enumerate(trainloader, 0):
        #     targets.append(data[1])
        # targets = [item.item() for sublist in targets for item in sublist]
        # model_logger.info(f"{self.node_name}, {Counter(targets)}")
        # model_logger.info(f"{self.node_name}: Training set size: {num_examples['trainset']}")
        # model_logger.info(f"{self.node_name}: Test set size: {num_examples['testset']}")


    def get_weights_list(self) -> list[float]:
        """Get the parameters of the network.
        
        Parameters
        ----------
        
        Returns
        -------
        List[float]: parameters of the network
        """
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]


    def get_weights(self) -> dict:
        """Get the weights of the network.
        
        Parameters
        ----------
        
        Raises
        -------------
            Exception: if the model is not initialized it raises an exception.
        
        Returns
        -------------
            _type_: weights of the network
        """
        self.net.to(self.cpu) # Dupming weights on cpu.
        return self.net.state_dict()
    
    
    def get_gradients(self) -> None:
        """Get the gradients of the network (differences between received and trained model)
        
        Parameters
        ----------
        
        Raises
        -------------
            Exception: if the original model was not preserved.
        
        Returns
        -------------
            Oredered_Dict: Gradients of the network.
        """
        assert self.initial_model != None, "Computing gradients require saving initial model first!"
        self.net.to(self.cpu) # Dupming weights on cpu.
        self.initial_model.to(self.cpu)
        weights_t1 = self.net.state_dict()
        weights_t2 = self.initial_model.state_dict()
        
        self.gradients = OrderedDict.fromkeys(weights_t1.keys(), 0)
        for key in weights_t1:
            self.gradients[key] =  weights_t1[key] - weights_t2[key]

        return self.gradients # Try: to provide original weights, no copies


    def update_weights(
        self, 
        avg_tensors
        ) -> None:
        """Updates the weights of the network stored on client with passed tensors.
        
        Parameters
        ----------
        avg_tensors: Ordered_Dict
            An Ordered Dictionary containing a averaged tensors. Copied for the 
            particular node.
        
        Raises
        ------
        Exception: _description_
       
        Returns
        -------
        None
        """
        self.net.load_state_dict(copy.deepcopy(avg_tensors), strict=True)


    def store_model_on_disk(
        self,
        iteration: int,
        path: str
        ) -> None:
        """Saves local model in a .pt format.
        Parameters
        ----------
        Iteration: int
            Current iteration
        Path: str
            Path to the saved repository
        
        Returns: 
        -------
        None
        
        Raises
        -------
            Exception if the model is not initialized it raises an exception
        """
        name = f"node_{self.node_name}_iteration_{iteration}.pt"
        save_path = os.path.join(path, name)
        torch.save(
            self.net.state_dict(),
            save_path,
        )


    def preserve_initial_model(self) -> None:
        """Preserve the initial model provided at the
        end of the turn (necessary for computing gradients,
        when using aggregating methods such as FedOpt).
        
        Parameters
        ----------
        
        Returns
        -------
            Tuple[float, float]: Loss and accuracy on the training set.
        """
        self.initial_model = copy.deepcopy(self.net)


    def train(
        self,
        iteration: int,
        epoch: int
        ) -> tuple[float, torch.tensor]:
        """Train the network and computes loss and accuracy.
        
        Parameters
        ----------
        iterations: int 
            Current iteration
        epoch: int
            Current (local) epoch
        
        Returns
        -------
        None
        """
        criterion = nn.CrossEntropyLoss()
        train_loss = 0
        correct = 0
        total = 0
        # Try: to place a net on the device during the training stage
        self.net.to(self.device)
        self.net.train()
        for _, dic in enumerate(self.trainloader):
            inputs = dic['image']
            targets = dic['label']
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.net.zero_grad() # Zero grading the network                        
            # forward pass, backward pass and optimization
            outputs = self.net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            predicted = torch.nn.functional.softmax(outputs, dim=1).argmax(dim=1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
                    
            # Emptying the cuda_cache
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()

        loss = train_loss / len(self.trainloader)
        accuracy = correct / total
        model_logger.info(f"[ITERATION {iteration} | EPOCH {epoch} | NODE {self.node_name}] Training on {self.node_name} results: loss: {loss}, accuracy: {accuracy}")
        
        return (loss, 
                accuracy)
    

    def evaluate_model(self):
        """Validate the network on the local test set.
        
        Parameters
        ----------
        
        Returns
        -------
            Tuple[float, float]: loss and accuracy on the test set.
        """
        # Try: to place net on device directly during the evaluation stage.
        self.net.to(self.device)
        self.net.eval()
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        correct = 0
        total = 0
        y_pred = []
        y_true = []
        losses = []
        
        with torch.no_grad():
            for _, dic in enumerate(self.testloader):
                inputs = dic['image']
                targets = dic['label']
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output = self.net(inputs)
                
                total += targets.size(0)
                test_loss = criterion(output, targets).item()
                losses.append(test_loss)
                pred = torch.nn.functional.softmax(output, dim=1).argmax(dim=1, keepdim=True)
                correct += pred.eq(targets.view_as(pred)).sum().item()
                y_pred.append(pred)
                y_true.append(targets)

        test_loss = np.mean(losses)
        accuracy = correct / total

        y_true = [item.item() for sublist in y_true for item in sublist]
        y_pred = [item.item() for sublist in y_pred for item in sublist]

        f1score = f1_score(y_true, y_pred, average="macro")
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")

        cm = confusion_matrix(y_true, y_pred)
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        accuracy_per_class = cm.diagonal()

        true_positives = np.diag(cm)
        num_classes = len(list(set(y_true)))

        false_positives = []
        for i in range(num_classes):
            false_positives.append(sum(cm[:,i]) - cm[i,i])

        false_negatives = []
        for i in range(num_classes):
            false_negatives.append(sum(cm[i,:]) - cm[i,i])

        true_negatives = []
        for i in range(num_classes):
            temp = np.delete(cm, i, 0)   # delete ith row
            temp = np.delete(temp, i, 1)  # delete ith column
            true_negatives.append(sum(sum(temp)))

        denominator = [sum(x) for x in zip(false_positives, true_negatives)]
        false_positive_rate = [num/den for num, den in zip(false_positives, denominator)]

        denominator = [sum(x) for x in zip(true_positives, false_negatives)]
        true_positive_rate = [num/den for num, den in zip(true_positives, denominator)]

        # # Emptying the cuda_cache
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        return (
                test_loss,
                accuracy,
                f1score,
                precision,
                recall,
                accuracy_per_class,
                true_positive_rate,
                false_positive_rate
                )


    def quick_evaluate(self) -> tuple[float, float]:
        """Quicker version of the evaluate_model(function) 
        Validate the network on the local test set returning only the loss and accuracy.
        
        Parameters
        ----------
        
        Returns
        -------
            Tuple[float, float]: loss and accuracy on the test set.
        """
        # Try: to place net on device directly during the evaluation stage.
        self.net.to(self.device)
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for _, dic in enumerate(self.testloader):
                inputs = dic['image']
                targets = dic['label']
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                predicted = torch.nn.functional.softmax(outputs, dim=1).argmax(dim=1, keepdim=True)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss = test_loss / len(self.testloader)
        accuracy = correct / total
                
        # # Emptying the cuda_cache
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        return (
            test_loss,
            accuracy
            )


    def transform_func(
        self,
        data
        ):
        convert_tensor = transforms.ToTensor()
        data['image'] = [convert_tensor(img) for img in data['image']]
        return data