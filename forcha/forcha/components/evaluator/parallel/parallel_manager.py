from forcha.components.evaluator.evaluation_manager import Evaluation_Manager
from forcha.models.federated_model import FederatedModel
from forcha.exceptions.evaluatorexception import Sample_Evaluator_Init_Exception
from forcha.components.evaluator.parallel.parallel_alpha import Parallel_Alpha
from forcha.components.evaluator.parallel.parallel_psi import Parallel_PSI
from forcha.utils.optimizers import Optimizers
from collections import OrderedDict
from forcha.utils.csv_handlers import save_coalitions
import copy


class Parallel_Manager(Evaluation_Manager):
    def __init__(self, 
                 settings: dict, 
                 model_template: FederatedModel,
                 optimizer_template: Optimizers,  
                 nodes: list = None, 
                 iterations: int = None,
                 full_debug: bool = False) -> None:
        super().__init__(
            settings, 
            model_template, 
            optimizer_template, 
            nodes, 
            iterations, 
            full_debug
            )
        # Sets up a flag for each available method of evaluation.
        # Flag: Shapley-OneRound Method
        # Flag: LOO-InSample Method
        if settings.LooSample:
            self.flag_sample_evaluator = True
            self.compiled_flags.append('in_sample_loo')
        else:
            self.flag_sample_evaluator = False
        # Flag: LSAA
        if settings.AlphaSample:
            self.flag_alpha_evaluator = True
            self.compiled_flags.append('ALPHA')
        else:
            self.flag_alpha_evaluator = False


        # Initialization: LOO-InSample Method
        if self.flag_sample_evaluator:
            try:
                self.sample_evaluator = Parallel_PSI(nodes=nodes, iterations=iterations)
            except NameError as e:
                raise Sample_Evaluator_Init_Exception # TODO
            
        if self.flag_alpha_evaluator:
            try:
                self.alpha_evaluator = Parallel_Alpha(nodes = nodes, iterations = iterations)
                self.search_length = settings.line_search_length
            except NameError as e:
                raise #TODO: Custom error
            except KeyError as k:
                raise #TODO: Lacking configuration error
        
        # self.flag_shap_or = False
        # self.flag_loo_or = False
        # self.flag_samplesh_evaluator = False
        
    def track_results(self,
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
        # LOO-InSample Method
        if self.flag_sample_evaluator:
            if iteration in self.scheduler['in_sample_loo']: # Checks scheduler
                debug_values = self.sample_evaluator.update_psi(
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
        #LSAA Method
        if self.flag_alpha_evaluator:
            if iteration in self.scheduler['ALPHA']: # Checks scheduler
                debug_values = self.alpha_evaluator.update_alpha(
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