from forcha.exceptions.settingexception import SettingsObjectException
import os
import time

class Settings():
    def __init__(self,
                simulation_seed: int = 42,
                global_epochs: int = 10,
                local_epochs: int = 2,
                number_of_nodes: int = 10,
                sample_size: int = 10,
                optimizer: str = 'RMS',
                batch_size: int = 32,
                learning_rate: float = 0.01,
                save_nodes_models: bool = False,
                save_central_model: bool = False,
                save_training_metrics: bool = True,
                root_name : str = os.getcwd(),
                **kwargs) -> None:
        """Initialization of an instance of the Settings object. Requires choosing the initialization method.
        Can be initialized either from a dictionary containing all the relevant key-words or from the 
        manual launch. It is highly advised that the Settings object should be initialized from the dicitonary.
        
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
        acceptable_keys_list = ['momentum', 'nesterov', 'force_cpu']
        self.simulation_seed = simulation_seed
        self.global_epochs = global_epochs
        self.local_epochs = local_epochs
        self.number_of_nodes = number_of_nodes
        self.sample_size = sample_size
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_nodes_models = save_nodes_models
        self.save_central_model = save_central_model
        self.save_training_metrics = save_training_metrics

        self.orchestrator_model_path, self.nodes_model_path, self.results_path = self.form_archive(
            root_name = root_name
        )
        for k in kwargs.keys():
            if k in acceptable_keys_list:
                self.__setattr__(k, kwargs[k])

        self.print_orchestrator_template()
        self.print_nodes_template()
    

    def form_archive(
        self,
        root_name: str
        ):
        time_tuple = time.localtime()
        time_string = time.strftime("%m_%d_%Y__%H_%M_%S", time_tuple)
        root_name = os.path.join(root_name, f"archiver_from_{time_string}")
        
        counter = 1
        while os.path.exists(root_name):
            root_name = os.path.join(os.getcwd(), f"archiver_from_{time_string}_{counter}")
            counter += 1
        
        # General Directory
        os.mkdir(root_name)
        # Directory for storing results
        results_path = os.path.join(root_name, 'results')
        os.mkdir(results_path)
        # Directory for storing models
        model_path = os.path.join(root_name, 'models')
        orchestrator_model_path = os.path.join(model_path, 'orchestrator')
        nodes_model_path = os.path.join(model_path, 'nodes')
        os.mkdir(model_path)
        os.mkdir(orchestrator_model_path)
        os.mkdir(nodes_model_path)
        
        return (orchestrator_model_path, nodes_model_path, results_path)


    def print_orchestrator_template(self,
                                    orchestrator_type: str = 'general'):
        """Prints out the used template for the orchestrator.
        
        Parameters
        ----------
        orchestrator_type: str, default to general
            A type of the orchestrator
        
        Returns
        -------
        None
        """
        string = f"""
        simulation seed: {self.simulation_seed},
        global epochs: {self.global_epochs},
        number_of_nodes: {self.number_of_nodes},
        sample_size:    {self.sample_size},
        """
        print(string)


    def print_nodes_template(self):
        """Prints out the used template for the nodes.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        string = f"""
        local_epochs: {self.local_epochs},
        optimizer: {self.optimizer},
        batch_size: {self.batch_size},
        learning_rate: {self.learning_rate}                    
        """
        print(string)