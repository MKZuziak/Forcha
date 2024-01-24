from forcha.components.settings.evaluator_settings import Evaluator_Settings
from forcha.exceptions.settingexception import SettingsObjectException
import numpy as np

# TODO: THIS WAS NOT REFACTORED YET

class Adjustive_Settings(Evaluator_Settings):
    def __init__(self, allow_default: bool, 
                 initialization_method: str = 'dict', 
                 dict_settings: dict = None, 
                 **kwargs) -> None:
        """Initialization of an instance of the Adjustive_orchestrator Settings object. Requires choosing the 
        initialization method.Can be initialized either from a dictionary containing all the relevant key-words or from the 
        manual launch. It is highly advised that the Settings object should be initialized from the dicitonary.
        It inherits all the properties and attributes from the Parent class adding additionally the Evaluator object.
        
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
        super().__init__(allow_default, 
                         initialization_method, 
                         dict_settings, 
                         **kwargs)
        if initialization_method == 'dict':
            self.init_adjustive_properties(dict_settings=self.orchestrator_settings)
        elif initialization_method == 'manual':
            # TODO: Not finished yet!
            raise NotImplemented()
        else:
            raise SettingsObjectException('Initialization method is not supported. '\
                                          'Supported methods: dict, manual')
    
    
    def init_adjustive_properties(self,
                                  dict_settings: dict):
        """Loads the evaluator configuration onto the settings instance. If the self.allow_default 
        flag was set to True during instance creation, a default evaluator tempalte will be created
        in absence of the one provided.
        
        Parameters
        ----------
        dict_settings: dict, default to None
            A dictionary containing all the relevant settings if the initialization is made from dir. 
            Default to None
        
        Returns
        -------
        None
        """        
        try:
            self.sampling_array = dict_settings['sampling_array']
            if self.sampling_array == 'uniform':
                self.sampling_array = np.full(self.number_of_nodes, (1 / self.number_of_nodes))
            else:
                raise SettingsObjectException("Provided type of generating sampling array is not supported. Currently supported: uniform distribution.")
        except KeyError:
            if self.allow_defualt:
                self.sampling_array = np.full(self.number_of_nodes, (1 / self.number_of_nodes))
            else:
                raise SettingsObjectException("Could not find the provided method of sampling array creation.")
        
        try:
            self.action = dict_settings['action']
        except KeyError:
            if self.allow_defualt:
                self.action = 'adjust'
            else:
                raise SettingsObjectException("Provided type of orchestrator requires an action parameter. Supported types: adjust and remove.")
        
        try:
            self.delta = dict_settings['delta']
        except KeyError:
            if self.allow_defualt:
                self.delta = 1.0
            else:
                raise SettingsObjectException("Provided type of orchestrator requires a delta parameter. Supported types: [float].")
        
        try:
            self.evaluator_settings['leading_method']
        except KeyError:
            if self.allow_defualt:
                self.evaluator_settings['leading_method'] = "LSAA"
            else:
                raise SettingsObjectException("Provided type of orchestrator requires a leading method parameter. Supported types: LOO, LSAA, EXLSAA.")
        
        self.print_adjustive_template()


    def print_adjustive_template(self):
        """Prints out the used template for the adjustive propierties.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        dict
        """
        string = f"""
        Sampling array: {self.sampling_array},
        Action: {self.action},
        Delta: {self.delta},
        Leading method: {self.evaluator_settings['leading_method']}
        """
        print(string) #TODO: Switch for logger