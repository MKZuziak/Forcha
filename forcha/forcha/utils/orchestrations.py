from typing import Any, Tuple, List, Dict, AnyStr, Union
from forcha.components.nodes.federated_node import FederatedNode
import numpy as np
import datasets
from logging import Logger
import random


def prepare_nodes(node: FederatedNode, 
                 model: Any,
                 dataset: list[datasets.arrow_dataset.Dataset, 
                               datasets.arrow_dataset.Dataset],
                query) -> AnyStr:
    """Used to connect the node and prepare it for training.
    Updates instances of a FederatedNode object and
    puts it into communication_queue.
    
    -------------
    Args:
        node (int): ID of the node that we want to connect.
        model (Any): Compiled or pre-compiled model to be trained.
        dataset (list[datasets.arrow_dataset.Dataset, 
                datasets.arrow_dataset.Dataset]): A dataset in the
                format ["train_data", "test_data"] that will be used 
                by the selected node.
        comunication_queue (multiprocess.Manager.Queue): Communication queue.
    -------------
    Returns:
        message(str): "OK" """
    node.prepare_node(model=model, data=dataset)
    query.put(node)
    return "OK"


def create_nodes(node_id: int, 
                nodes_settings) -> list[FederatedNode]: 
    """Creates a list of nodes that will be connected to the 
    orchestrator and contained in a list[FederatedNode] container.
    -------------
    Args:
        node (int): ID of the node that we want to connect.
    -------------
    Returns:
        list[FederatedNode]: List of nodes that were created.
    """
    nodes = [FederatedNode(id, nodes_settings) for id in node_id]
    return nodes


def check_health(node: FederatedNode,
                 orchestrator_logger: Logger) -> bool:
    """Checks whether node has successfully conducted the transaction
    and can be moved to the next phase of the training. According to the
    adopted standard - if node.state == 0, node is ready for the next
    transaction. On the contrary, if node.state == 1, then node must be 
    excluded from the simulation (internal error).
    -------------
    Args:
        node (FederatedNode): FederatedNode object
    -------------
    Returns:
        bool(): True if node is healthy, False otherwise."""
    if node.state == 0:
        orchestrator_logger.warning(f"Node {node.node_id} was updated successfully.")
        return True
    else:
        orchestrator_logger.warning(f"Node {node.node_id} failed during the update.")
        return False


def sample_nodes(nodes: list[FederatedNode], 
                 sample_size: int,
                 generator: np.random.Generator,
                 return_aux: bool = False) -> list[FederatedNode]:
    """Sample the nodes given the provided sample size. If sample_size is bigger
    or equal to the number of av. nodes, the sampler will return the original list.
     -------------
    Args:
        nodes (list[FederatedNode]): original list of nodes to be sampled from,
        sample_size (int): size of the sample.
        generator (np.random.Generator): a numpy generator initialized on the server side.
        return_aux (bool = auxiliary): if set to True, will return a list containing id's of the sampled nodes.
    -------------
    Returns:
        list[FederatedNode]: List of sampled nodes."""
    sample = generator.choice(nodes, size=sample_size, replace=False)
    if return_aux == True:
        sampled_ids = [node.node_id for node in sample]
        return (sample, sampled_ids)
    else:
        return sample


def sample_weighted_nodes(nodes: list[FederatedNode], 
                          sample_size: int,
                          sampling_array: np.array,
                          generator: np.random.Generator,
                          return_aux: bool = False) -> list[FederatedNode]:
    """Sample the nodes given the provided sample size. It requires passing a sampling array
    containing list of weights associated with each node.
     -------------
    Args:
        nodes (list[FederatedNode]): original list of nodes to be sampled from,
        sample_size (int): size of the sample.
        sampling_array (np.array[float]): numpy array of weights associated with each agent.
        generator (np.random.Generator): a numpy generator initialized on the server side.
        return_aux (bool = auxiliary): if set to True, will return a list containing id's of the sampled nodes.
    -------------
    Returns:
        list[FederatedNode]: List of sampled nodes."""
    sample = generator.choice(nodes, size=sample_size, p = sampling_array, replace=False)
    if return_aux == True:
        sampled_ids = [node.node_id for node in sample]
        return (sample, sampled_ids)
    else:
        return sample


def train_nodes(
    node: FederatedNode,
    iteration: int,
    mode: str = 'weights') -> tuple[int, List[float]]:
    """Used to command the node to start the local training.
    Invokes .train_local_model method and returns the results.
    Parameters
    ----------
    node: FederatedNode 
        Node that we want to train.
    mode: str
        Mode of the training. 
        Mode = 'weights': Node will return model's weights.
        Mode = 'gradients': Node will return model's gradients.
    Returns
    -------
    tuple(node_id: str, weights)"""
    node_id, weights, loss_list, accuracy_list = node.train_local_model(
        mode = mode,
        iteration=iteration)
    return (node_id, weights, loss_list, accuracy_list)