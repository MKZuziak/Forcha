import copy
import os
import csv
from collections import OrderedDict

from forcha.components.evaluator.parallel.parallel_alpha_evaluator import Parallel_Alpha_Amplified
from forcha.components.evaluator.parallel.parallel_loo_evaluator import Parallel_Sample_LOO_Evaluator
from forcha.components.evaluator.parallel.parallel_shapley_evaluator import Parallel_Sample_Shapley_Evaluator
from forcha.components.evaluator.evaluation_manager import Evaluation_Manager
from forcha.models.federated_model import FederatedModel
from forcha.components.settings.evaluator_settings import EvaluatorSettings
from forcha.utils.optimizers import Optimizers
from forcha.utils.csv_handlers import save_coalitions


class Parallel_Manager(Evaluation_Manager):
    def __init__(
        self,
        settings: EvaluatorSettings,
        model_template: FederatedModel,
        optimizer_template: Optimizers,
        nodes: list[int] = None,
        iterations: int = None,
        full_debug: bool = False,
        number_of_workers: int = 20
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
        number_of_workers: int, default to 20
            Number of workers (if any) that will distribute the computations.
        
        Returns
        -------
        None
        """
        # Settings processed as new attribute
        self.settings = settings
        self.number_of_workers = number_of_workers
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
            self.shapley_evaluator = Parallel_Sample_Shapley_Evaluator(
                nodes=nodes, 
                iterations=iterations,
                number_of_workers=self.number_of_workers
                )
        # Initialization: LOO-InSample Method
        if self.flag_sample_evaluator:
            self.sample_evaluator = Parallel_Sample_LOO_Evaluator(
                nodes=nodes, 
                iterations=iterations
                )
        if self.flag_alpha_evaluator:
            self.alpha_evaluator = Parallel_Alpha_Amplified(
                nodes = nodes, 
                iterations = iterations
                )
            self.search_length = settings.line_search_length

        # Sets up the scheduler
        if settings.scheduler == True:
            self.scheduler = settings.schedule
        else:
            self.scheduler = {flag: [iteration for iteration in range(iterations)] for flag in self.compiled_flags}