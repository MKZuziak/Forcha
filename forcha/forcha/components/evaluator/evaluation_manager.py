import copy
import os
import csv
from collections import OrderedDict

from forcha.components.evaluator.alpha_evaluator import Alpha_Amplified
from forcha.components.evaluator.loo_evaluator import Sample_LOO_Evaluator
from forcha.components.evaluator.shapley_evaluator import Sample_Shapley_Evaluator
from forcha.models.federated_model import FederatedModel
from forcha.exceptions.evaluatorexception import Sample_Evaluator_Init_Exception
from forcha.components.settings.evaluator_settings import EvaluatorSettings
from forcha.utils.optimizers import Optimizers
from forcha.utils.csv_handlers import save_coalitions

# def compare_for_debug(dict1, dict2):
#     for (row1, row2) in zip(dict1.values(), dict2.values()):
#         if False in (row1 == row2):
#             return False
#         else:
#             return True


class Evaluation_Manager():
    """Evaluation Manager encapsulates the whole process of assessing the marginal
    clients' contributions, so the orchestrator code is free of any encumbrance
    connected to it. Evaluation Manager life-cycle is splitted into four different
    stages: initialization, preservation, tracking and finalization. Initialization
    requires a dictionary containing all the relevant settings. Preservation should
    be called each round before the training commences (so the evaluation manager
    has last set of weights and momentum information). Tracking should be called each
    round to compute different metrics. Finalization can be called at the end of the
    life-cycle to preserve the results."""
    

    def __init__(
        self,
        settings: EvaluatorSettings,
        model_template: FederatedModel,
        optimizer_template: Optimizers,
        nodes: list[int] = None,
        iterations: int = None,
        full_debug: bool = False
        ) -> None:
        """Manages the process of evaluation. Creates an instance of Evaluation_Manager 
        object, that controls all the instances that perform evaluation. Evaluation
        Manager operatores on 'flags' that are represented as bolean attributes of the
        instance of the Class. The value of flags is dictated by the corresponding value
        used in the settings. Giving an example, settings [...]['IN_SAMPLE_LOO'] to
        True will trigger the flag of the instance. This means, that each time the method
        Evaluation_Manager().track_results is called, Evaluator will try to calculate
        Leave-One-Out score for each client in the sample.
        
        Parameters
        ----------
        settings: EvaluatorSettings
            A dictionary containing all the relevant settings for the evaluation_manager.
        model_template: FederatedModel
            A initialized instance of the class forcha.models.pytorch.federated_model.FederatedModel
            This is necessary to initialize some contribution-estimation objects. This is a Federated
            Model that is associated with the central model (and its test set).
        nodes: list[int], default to None
            A list containing the id's of all the relevant nodes.
        iterations: int, default to None
            Number of iterations.
        full_debug: bool, default to False
            Boolean flag for enabling a full debug mode.
        
        Returns
        -------
        None
        """
        # Settings processed as new attribute
        self.settings = settings
        # Copies of the previous model, updated model and optimizer
        self.previous_c_model = None
        self.updated_c_model = None
        self.previous_optimizer = None
        # List[int] of all the nodes in the population
        self.nodes = nodes
        # Template of the model and the optimizer
        self.model_template = copy.deepcopy(model_template)
        self.optimizer_template = copy.deepcopy(optimizer_template)
        # Boolean flag for a full debug
        self.full_debug = full_debug
        
        # Sets up a flag for each available method of evaluation.
        self.compiled_flags = []
        # Flag: LOO-InSample Method
        if settings.in_sample_loo:
            self.flag_sample_evaluator = True
            self.compiled_flags.append('in_sample_loo')
        else:
            self.flag_sample_evaluator = False
        # Flag: Shapley-InSample Method
        if settings.in_sample_shap:
            self.flag_samplesh_evaluator = True
            self.compiled_flags.append('in_sample_shap')
        else:
            self.flag_samplesh_evaluator = False
        # Flag: Alpha-Amplification
        if settings.in_sample_alpha:
            self.flag_alpha_evaluator = True
            self.compiled_flags.append('in_sample_alpha')
        else:
            self.flag_alpha_evaluator = False
        

        # Initialization of objects necessary to perform evaluation.
        # Initialization: In-sample Shapley
        if self.flag_samplesh_evaluator:
            self.shapley_evaluator = Sample_Shapley_Evaluator(
                nodes=nodes, 
                iterations=iterations
                )
        # Initialization: LOO-InSample Method
        if self.flag_sample_evaluator:
            self.sample_evaluator = Sample_LOO_Evaluator(
                nodes=nodes, 
                iterations=iterations
                )
        if self.flag_alpha_evaluator:
            self.alpha_evaluator = Alpha_Amplified(
                nodes = nodes, 
                iterations = iterations
                )
            self.search_length = settings.line_search_length

        # Sets up the scheduler
        if settings.scheduler == True:
            self.scheduler = settings.schedule
        else:
            self.scheduler = {flag: [iteration for iteration in range(iterations)] for flag in self.compiled_flags}
    
        
    # def set_leading_method(self,
    #                        name: str):
    #     """Sets the leading method of evaluation.
    #     This method will be returned in subseqeunt
    #     'self.get_last_results' calls. 
        
    #     Parameters
    #     ----------
    #     name (str): name of the method that should be set to main.
    #     Returns
    #     -------
    #     None
    #     """
    #     if name == "LOO":
    #         self.default_method = self.or_evaluator
    #     elif name == "ALPHA":
    #         self.default_method = self.alpha_evaluator
    #     else:
    #         raise NameError # TODO: Add custom error.
    
    
    def preserve_previous_model(
        self,
        previous_model: OrderedDict
        ):
        """Preserves the model from the previous round by copying 
        its structure and using it as an attribute's value. Should
        be called each training round before the proper training
        commences.
        
        Parameters
        ----------
        previous_model: OrderedDict
            Ordered Dict containing weights of the previous (central) model.
        
        Returns
        -------
        None
        """
        self.previous_c_model = copy.deepcopy(previous_model)
    

    def preserve_updated_model(
        self,
        updated_model: OrderedDict
        ):
        """Preserves the updated version of the central model
        by copying its structure and using it as an attribute's value. 
        Should be called each training after updating the weights
        of the central model.
        
        Parameters
        ----------
        updated_model: FederatedModel
            Ordered Dict containing weights of the previous (central) model.
        
        Returns
        -------
        None
        """
        self.updated_c_model = copy.deepcopy(updated_model)
    
    
    def preserve_previous_optimizer(
        self,
        previous_optimizer: tuple[OrderedDict, OrderedDict, float]):
        """Preserves the Optimizer from the previous round by copying 
        its structure and using it as an attribute's value. Should
        be called each training round before the proper training
        commences.
        
        Parameters
        ----------
        previous_optimizer: tuple[OrderedDict, OrderedDict, float]
            Tuple containing data of the optimizer, of the form:
            tuple[previous_delta, previous_momentum, learning_rate]
        
        Returns
        -------
        None
        """
        self.previous_optimizer = copy.deepcopy(previous_optimizer)
    
    
    def get_last_results(
        self,
        iteration: int
        ) -> tuple[int, dict]:
        """Returns the results of the last evaluation 
        round

        Parameters
        iteration (int): curret iteration
        ----------
        
        Returns
        -------
        tuple[int, dict]: tuple containing a last round id and a dict mapping nodes' id to the result.
        """
        return self.default_method.return_last_value(iteration = iteration)

    
    def track_results(
        self,
        gradients: OrderedDict,
        nodes_in_sample: list,
        iteration: int):
        """Method used to track_results after each training round.
        Because the Orchestrator abstraction should be free of any
        unnecessary encumbrance, the Evaluation_Manager.track_results()
        will take care of any result preservation and score calculation that 
        must be done in order to establish the results.
        
        Parameters
        ----------
        gradients: OrderedDict
            An OrderedDict containing gradients of the sampled nodes.
        nodes_in_sample: list
            A list containing id's of the nodes that were sampled.
        iteration: int
            The current iteration.
        
        Returns
        -------
        None
        """
        
        # In-sample LOO
        if self.flag_sample_evaluator:
            if iteration in self.scheduler['in_sample_loo']: # Checks scheduler
                debug_values = self.sample_evaluator.evaluate_round(
                    model_template = self.model_template,
                    optimizer_template = self.optimizer_template,
                    gradients = gradients,
                    nodes_in_sample = nodes_in_sample,
                    iteration = iteration,
                    optimizer = copy.deepcopy(self.previous_optimizer),
                    final_model = copy.deepcopy(self.updated_c_model),
                    previous_model= copy.deepcopy(self.previous_c_model)
                    )
                # Preserving debug values (if enabled)
                if self.full_debug:
                    if iteration  == 0:
                        save_coalitions(
                            values=debug_values,
                            path=self.settings.results_path,
                            name='col_values_loo_debug.csv',
                            iteration=iteration,
                            mode=0
                            )
                    else:
                        save_coalitions(
                            values=debug_values,
                            path=self.settings.results_path,
                            name='col_values_loo_debug.csv',
                            iteration=iteration,
                            mode=1
                            )

        # In-sample Shapley
        if self.flag_samplesh_evaluator:
            if iteration in self.scheduler['in_sample_shap']: # Checks scheduler
                debug_values = self.shapley_evaluator.evaluate_round(
                    model_template = self.model_template,
                    optimizer_template = self.optimizer_template,
                    gradients = gradients,
                    nodes_in_sample = nodes_in_sample,
                    iteration = iteration,
                    optimizer = copy.deepcopy(self.previous_optimizer),
                    final_model = copy.deepcopy(self.updated_c_model),
                    previous_model= copy.deepcopy(self.previous_c_model)
                    )
                # Preserving debug values (if enabled)
                if self.full_debug:
                    if iteration  == 0:
                        save_coalitions(
                            values=debug_values,
                            path=self.settings.results_path,
                            name='col_values_shapley_debug.csv',
                            iteration=iteration,
                            mode=0
                            )
                    else:
                        save_coalitions(
                            values=debug_values,
                            path=self.settings.results_path,
                            name='col_values_shapley_debug.csv',
                            iteration=iteration,
                            mode=1
                            )
    
        # In-sample ALPHA
        if self.flag_alpha_evaluator:
            if iteration in self.scheduler['in_sample_alpha']: # Checks scheduler
                debug_values = self.alpha_evaluator.evaluate_round(
                    model_template = self.model_template,
                    optimizer_template = self.optimizer_template,
                    gradients = gradients,
                    nodes_in_sample = nodes_in_sample,
                    iteration = iteration,
                    search_length = self.search_length,
                    optimizer = copy.deepcopy(self.previous_optimizer),
                    final_model = copy.deepcopy(self.updated_c_model),
                    previous_model = copy.deepcopy(self.previous_c_model))
            
                            # Preserving debug values (if enabled)
                if self.full_debug:
                    if iteration  == 0:
                        save_coalitions(
                            values=debug_values,
                            path=self.settings.results_path,
                            name='col_values_alpha_debug.csv',
                            iteration=iteration,
                            mode=0
                            )
                    else:
                        save_coalitions(
                            values=debug_values,
                            path=self.settings.results_path,
                            name='col_values_alpha_debug.csv',
                            iteration=iteration,
                            mode=1
                            )


    def finalize_tracking(
        self,
        path: str = None
        ):
        """Method used to finalize the tracking at the end of the training.
        Because the Orchestrator abstraction should be free of any
        unnecessary encumbrance, all the options configuring the behaviour
        of the finalize_tracking method, should be pre-configured at the stage
        of the Evaluation_Manager instance initialization.  
        
        Parameters
        ----------
        path: str, default to None
            a string or Path-like object to the directory in which results
            should be saved.
        
        Returns
        -------
        
        None
        """
        results = {'partial': {}, 'full': {}}

        # if self.flag_shap_or:
        #     raise NotImplementedError
        
        # if self.flag_loo_or:
        #     raise NotImplementedError
        
        if self.flag_sample_evaluator:
            partial_psi, psi = self.sample_evaluator.calculate_final_result()
            results['partial']['partial_loo'] = partial_psi
            results['full']['loo'] = psi
        
        if self.flag_samplesh_evaluator:
            partial_shap, shap = self.shapley_evaluator.calculate_final_result()
            results['partial']['partial_shap'] = partial_shap
            results['full']['shap'] = shap
        
        if self.flag_alpha_evaluator:
            partial_alpha, alpha = self.alpha_evaluator.calculate_final_result()
            results['partial']['partial_alpha'] = partial_alpha
            results['full']['alpha'] = alpha
                
        # Preserve partial results
        for metric, values in results['partial'].items():
            s_path = os.path.join(path, (str(metric) + '.csv'))
            field_names = self.nodes
            field_names.append('iteration') # Field names == nodes id's (keys)
            with open(s_path, 'w+', newline='') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
                csv_writer.writeheader()
                for iteration, row in values.items():
                    row['iteration'] = iteration
                    csv_writer.writerow(row)
        
        # Preserve final results
        for metric, values in results['full'].items():
            s_path = os.path.join(path, (str(metric) + '.csv'))
            field_names = values.keys() # Field names == nodes id's (keys)
            with open(s_path, 'w+', newline='') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
                csv_writer.writeheader()
                csv_writer.writerow(values)
        
        return results