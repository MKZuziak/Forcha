import os

from forcha.components.settings.fedopt_settings import FedoptSettings

class EvaluatorSettings(FedoptSettings):
    def __init__(self,
                 simulation_seed: int = 42,
                 global_epochs: int = 10,
                 local_epochs: int = 2,
                 number_of_nodes: int = 10,
                 sample_size: int = 10,
                 optimizer: str = 'RMS',
                 batch_size: int = 32,
                 global_optimizer: str = 'Simple',
                 global_learning_rate: float = 1.0,
                 learning_rate: float = 0.01,
                 save_nodes_models: bool = False,
                 save_central_model: bool = False,
                 save_training_metrics: bool = True,
                 b1: float = 0,
                 b2: float = 0,
                 tau: float = 0,
                 in_sample_alpha: bool = True,
                 in_sample_loo: bool = True,
                 in_sample_shap: bool = False,
                 line_search_length: int = 1,
                 scheduler = None,
                 root_name : str = os.getcwd(),
                 **kwargs) -> None:
        """Initialization of an instance of the FedoptSettings object. Requires choosing the initialization method.
        Can be initialized either from a dictionary containing all the relevant key-words or from the 
        manual launch. It is highly advised that the Settings object should be initialized from the dicitonary.
        It inherits all the properties and attributes from the Parent class addting an additional Optimizer settings.
        
        Parameters
        ----------
        allow_default: bool
            A logical switch to allow using default values in absence of passed values.
        initialization_method: str, default to 'dict' 
            The method of initialization. Either 'dict' or 'manual'.
        dict_settings: dict, default to None
            A dictionary containing all the relevant settings if the initialization is made from dir. 
            Default to None
        
        Returns
        -------
        None
        """
        super(EvaluatorSettings, self).__init__(
            simulation_seed =simulation_seed,
            global_epochs = global_epochs,
            local_epochs = local_epochs,
            number_of_nodes = number_of_nodes,
            sample_size = sample_size,
            optimizer = optimizer,
            batch_size = batch_size,
            learning_rate = learning_rate,
            save_nodes_models = save_nodes_models,
            save_central_model = save_central_model,
            save_training_metrics = save_training_metrics,
            global_optimizer = global_optimizer,
            global_learning_rate = global_learning_rate,
            b1 = b1,
            b2 = b2,
            tau = tau,
            root_name = root_name,
            **kwargs
        )

        self.in_sample_loo = in_sample_loo
        self.in_sample_shap = in_sample_shap
        self.in_sample_alpha = in_sample_alpha
        self.line_search_length = line_search_length
        self.scheduler = scheduler
        self.print_evaluator_template()


    def print_evaluator_template(self):
        """Prints out the used template for the optimizer.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        """
        string = f"""
        global optimizer: {self.global_optimizer},
        global learning rate: {self.global_learning_rate},
        b1: {self.b1},
        b2: {self.b2},
        tau: {self.tau}
        activate LOO evaluation: {self.in_sample_loo},
        activate ALPHA evaluation: {self.in_sample_shap},
        activate SHAP evaluation: {self.in_sample_shap},
        alpha line search length: {self.line_search_length}
        """
        print(string)
