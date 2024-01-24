from torch.nn import Module
import csv
from typing import Any
import os


def log_model_metrics(iteration: int, 
                        model: Module,
                        logger,
                        ) -> dict[float, float, float, float, float, list, list, list, int]:
    """Used to log the model's metrics (on the orchestrator level).
    
    Parameters
    ----------
    iteration: int
        Current iteration of the training.
    model: torch.nn.Module
        Network deposited on the client
    loger: Logger
        Logger used to handle the entries.
    
    Returns
    -------
        dict[float, float, float, float, float, list, list, list, int]
    """
    try:
        (
            loss,
            accuracy,
            fscore,
            precision,
            recall,
            test_accuracy_per_class,
            true_positive_rate,
            false_positive_rate
        ) = model.evaluate_model()
        metrics = {"loss":loss, 
                    "accuracy": accuracy, 
                    "fscore": fscore, 
                    "precision": precision,
                    "recall": recall, 
                    "test_accuracy_per_class": test_accuracy_per_class, 
                    "true_positive_rate": true_positive_rate,
                    "false_positive_rate": false_positive_rate,
                    "epoch": iteration}
        logger.info(f"[ITERATION {iteration} | NODE {model.node_name}] Evaluating central model on node {model.node_name}. Results: {metrics}")
    except Exception as e:
        logger.warning(f"Unable to compute metrics. {e}")


def save_model_metrics(iteration: int, 
                        model: Module,
                        logger = None,
                        saving_path: str = None,
                        file_name: str = 'metrics.csv',
                        log_to_screen: bool = False) -> None:
    """Used to save the model metrics.
    
    Parameters
    ----------
    iteration: int 
        Current iteration of the training.
    model: torch.nn.Module 
        Network deposited on the client
    logger: Logger (default to None)
        Logger object that we want to use to handle the logs.
    saving_path: str (default to None)
        The saving path of the csv file - if none, the file will be saved in the current working directory.
    file_name: str 
        A desired file name for the metrics, default to 'metrics.csv'.
    log_to_screen: bool (default to False)
        Boolean flag whether we want to log the results to the screen.
    
    Returns
    -------
        None"""
    try:
        (
            loss,
            accuracy,
            fscore,
            precision,
            recall,
            test_accuracy_per_class,
            true_positive_rate,
            false_positive_rate
        ) = model.evaluate_model()
        metrics = {"epoch": iteration,
                    "node": model.node_name,
                    "loss":loss, 
                    "accuracy": accuracy, 
                    "fscore": fscore, 
                    "precision": precision,
                    "recall": recall, 
                    "test_accuracy_per_class": test_accuracy_per_class, 
                    "true_positive_rate": true_positive_rate,
                    "false_positive_rate": false_positive_rate,
                    "epoch": iteration}
        if log_to_screen == True:
            logger.info(f"Evaluating model after iteration {iteration} on node {model.node_name}. Results: {metrics}")
    except Exception as e:
        logger.warning(f"Unable to compute metrics. {e}")
    path = os.path.join(saving_path, file_name)
    with open(path, 'a+', newline='') as saved_file:
            writer = csv.DictWriter(saved_file, list(metrics.keys()))
            if os.path.getsize(path) == 0:
                writer.writeheader()
            writer.writerow(metrics)


def save_csv_file(
    file,
    saving_path: str = None,
    file_name: str = 'custom_metrics.csv'
    ) -> None:
    """Used to preserve the content of a csv file."""
    path = os.path.join(saving_path, file_name)
    with open(path, 'a+', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, list(file[0].keys()))
        writer.writeheader()
        for row in file:
            writer.writerow(row)


def save_training_metrics(
    file,
    saving_path: str = None,
    file_name: str = 'metrics.csv'
    ) -> None:
    """Used to preserve the content of a csv file."""
    path = os.path.join(saving_path, file_name)
    with open(path, 'a+', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, next(iter(file.values())).keys())
        if os.path.getsize(path) == 0:
            writer.writeheader()
        for row in file.values():
            writer.writerow(row)


def execute_local_test(
    file,
    saving_path: str = None,
    file_name: str = "local_tests.csv"
    ) -> None:
    pass

        