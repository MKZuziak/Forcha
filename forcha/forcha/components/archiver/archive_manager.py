from forcha.exceptions.settingexception import ArchiverSettingsException
from forcha.models.federated_model import FederatedModel
from forcha.utils.handlers import Handler
from forcha.components.nodes.federated_node import FederatedNode
import os


class Archive_Manager():
    def __init__(self,
                 archive_manager: dict,
                 logger = None) -> None:
        try:
            # Modes
            self.orchestrator_metrics = archive_manager["orchestrator"]
            self.clients_on_central = archive_manager["clients_on_central"]
            self.central_on_local = archive_manager["central_on_local"]
            self.save_results = archive_manager["save_results"]
            if self.save_results:
                self.only_log = False
            else:
                self.only_log = archive_manager["log_results"]
            
            self.save_orchestrator_model = archive_manager["save_orchestrator_model"]
            self.save_nodes_model = archive_manager['save_nodes_model']

            
            # Paths and filenames for preserving metrics
            self.metrics_savepath = archive_manager["metrics_savepath"]
            self.orchestrator_metrics_file = archive_manager["orchestrator_filename"]
            self.clients_on_central_file = archive_manager["clients_on_central_filename"]
            self.central_on_local_file = archive_manager["central_on_local_filename"]

            # Paths for preserving models
            self.orchestrator_save_path = archive_manager["orchestrator_model_savepath"]
            self.nodes_save_path = archive_manager["nodes_model_savepath"]

        except ArchiverSettingsException:
            raise ArchiverSettingsException('The dictionary passed to the Archiver does not contain all the necessary key-words '/
                                            "The Dictionary should contain following key-items pairs: {orchestrator: bool," /
                                            "clients_on_central: bool, central_on_local: bool, save_results: bool, log_results: bool}")
        
        if logger != None:
            self.logger = logger
    
    
    def archive_training_results(
        self,
        iteration: int,
        results: dict
    ):
        if self.orchestrator_metrics:
            Handler.save_training_metrics(file=results,
                                  saving_path = self.metrics_savepath,
                                  file_name = 'training_metrics.csv')
    
    
    def archive_local_test_results(
        self,
        nodes:list,
        iteration: int
        ):
        if self.central_on_local:
            if self.save_results:
                for node in nodes:
                    Handler.save_model_metrics(iteration = iteration,
                                                model = node.model,
                                                logger = self.logger,
                                                saving_path = self.metrics_savepath,
                                                log_to_screen = True,
                                                file_name='testing_metrics.csv') # PRESERVING METRICS FUNCTION -> CHANGE IF NEEDED
            elif self.only_log:
                for node in nodes:
                    Handler.log_model_metrics(iteration = iteration,
                                                model = node.model,
                                                logger = self.logger)
                
    
    
    def archive_testing_results(
        self,
        iteration: int,
        central_model: FederatedModel,
        nodes: list[FederatedNode]
    ):
        if self.orchestrator_metrics:
            if self.save_results:
                Handler.save_model_metrics(iteration=iteration,
                                           model = central_model,
                                           logger = self.logger,
                                           saving_path = self.metrics_savepath,
                                           log_to_screen = True,
                                           file_name=self.orchestrator_metrics_file)
            elif self.only_log:
                Handler.log_model_metrics(iteration=iteration,
                                          model=central_model,
                                          logger = self.logger)
            if self.save_orchestrator_model:
                central_model.store_model_on_disk(iteration = iteration,
                                                path = self.orchestrator_save_path)
        
        if self.central_on_local:
            if self.save_results:
                for node in nodes:
                    Handler.save_model_metrics(iteration = iteration,
                                               model = node.model,
                                               logger = self.logger,
                                               saving_path = self.metrics_savepath,
                                               log_to_screen = True,
                                               file_name=self.central_on_local_file) # PRESERVING METRICS FUNCTION -> CHANGE IF NEEDED
            elif self.only_log:
                for node in nodes:
                    Handler.log_model_metrics(iteration = iteration,
                                              model = node.model,
                                              logger = self.logger)
            