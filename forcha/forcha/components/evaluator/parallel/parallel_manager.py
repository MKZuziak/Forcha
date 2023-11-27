from forcha.components.evaluator.evaluation_manager import Evaluation_Manager
from forcha.models.federated_model import FederatedModel
from forcha.exceptions.evaluatorexception import Sample_Evaluator_Init_Exception
from forcha.components.evaluator.parallel.parallel_alpha import Parallel_Alpha
from forcha.components.evaluator.parallel.parallel_psi import Parallel_PSI
from collections import OrderedDict
from forcha.utils.csv_handlers import save_coalitions
import os
import csv


class Parallel_Manager(Evaluation_Manager):
    def __init__(self, 
                 settings: dict, 
                 model_template: FederatedModel,
                 orchestrator_template,  
                 nodes: list = None, 
                 iterations: int = None) -> None:
        super().__init__(settings, model_template, orchestrator_template, nodes, iterations)
        # Sets up a flag for each available method of evaluation.
        # Flag: Shapley-OneRound Method
        # Flag: LOO-InSample Method
        if settings.get("IN_SAMPLE_LOO"):
            self.flag_sample_evaluator = True
            self.compiled_flags.append('in_sample_loo')
        else:
            self.flag_sample_evaluator = False
        # Flag: LSAA
        if settings.get("ALPHA"):
            self.flag_alpha_evaluator = True
            self.compiled_flags.append('ALPHA')
        else:
            self.flag_alpha_evaluator = False
        
        # Sets up a flag for each available method of score preservation
        # Flag: Preservation of partial results (for In-Sample Methods)
        if settings['preserve_evaluation'].get("preserve_partial_results"):
            self.preserve_partial_results = True
        else:
            self.preserve_partial_results = False
        # Flag: Preservation of the final result (for In-Sample Methods)
        if settings['preserve_evaluation'].get("preserve_final_results"):
            self.preserve_final_results = True

        # Initialization: LOO-InSample Method
        if self.flag_sample_evaluator == True:
            try:
                self.sample_evaluator = Parallel_PSI(nodes=nodes, iterations=iterations)
            except NameError as e:
                raise Sample_Evaluator_Init_Exception # TODO
            
        if self.flag_alpha_evaluator == True:
            try:
                self.alpha_evaluator = Parallel_Alpha(nodes = nodes, iterations = iterations)
                self.search_length = settings['line_search_length']
            except NameError as e:
                raise #TODO: Custom error
            except KeyError as k:
                raise #TODO: Lacking configuration error
        
        self.flag_shap_or = False
        self.flag_loo_or = False
        self.flag_samplesh_evaluator = False
        
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
        #LSAA Method
        if self.flag_alpha_evaluator:
            if iteration in self.scheduler['ALPHA']: # Checks scheduler
                debug_values = self.alpha_evaluator.update_alpha(
                    gradients = gradients,
                    nodes_in_sample = nodes_in_sample,
                    iteration = iteration,
                    search_length = self.search_length,
                    optimizer = self.previous_optimizer,
                    previous_model = self.previous_c_model,
                    final_model = self.updated_c_model) # Preserving debug values (if enabled)
                if self.full_debug:
                    if iteration  == 0:
                        save_coalitions(values=debug_values,
                                        path=self.full_debug_path,
                                        name='col_values_alpha.csv',
                                        iteration=iteration,
                                        mode=0)
                    else:
                        save_coalitions(values=debug_values,
                                        path=self.full_debug_path,
                                        name='col_values_alpha.csv',
                                        iteration=iteration,
                                        mode=1)
        #PSI Method
        if self.flag_sample_evaluator:
            if iteration in self.scheduler['in_sample_loo']: # Checks scheduler
                debug_values = self.sample_evaluator.update_psi(gradients = gradients,
                                    nodes_in_sample = nodes_in_sample,
                                    iteration = iteration,
                                    optimizer = self.previous_optimizer,
                                    final_model= self.updated_c_model,
                                    previous_model = self.previous_c_model)# Preserving debug values (if enabled)
                if self.full_debug:
                    if iteration  == 0:
                        save_coalitions(values=debug_values,
                                        path=self.full_debug_path,
                                        name='col_values_loo.csv',
                                        iteration=iteration,
                                        mode=0)
                    else:
                        save_coalitions(values=debug_values,
                                        path=self.full_debug_path,
                                        name='col_values_loo.csv',
                                        iteration=iteration,
                                        mode=1)