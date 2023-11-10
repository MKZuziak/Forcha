def basic_fedavg(iterations: int,
                    number_of_nodes: int,
                    sample_size: int,
                    root_path: str,
                    local_lr: float = 0.01,
                    local_epochs: int = 2,
                    batch_size: int = 32):
    return {
        "orchestrator": {
            "iterations": iterations,
            "number_of_nodes": number_of_nodes,
            "sample_size": sample_size,
            'enable_archiver': True,
            "archiver":{
                "root_path": root_path,
                "orchestrator": True,
                "clients_on_central": True,
                "central_on_local": True,
                "log_results": True,
                "save_results": True,
                "save_orchestrator_model": True,
                "save_nodes_model": False,
                "form_archive": True
                }},
        "nodes":{
            "local_epochs": local_epochs,
            "model_settings": {
                "optimizer": "SGD",
                "batch_size": batch_size,
                "learning_rate": local_lr,
                "FORCE_CPU": False}}}

def basic_fedopt(iterations: int,
                    number_of_nodes: int,
                    sample_size: int,
                    root_path: str,
                    local_lr: float = 0.01,
                    central_lr: float = 0.5,
                    local_epochs: int = 2,
                    batch_size: int = 32):
    return {
        "orchestrator": {
            "iterations": iterations,
            "number_of_nodes": number_of_nodes,
            "sample_size": sample_size,
            'enable_archiver': True,
            "archiver":{
                "root_path": root_path,
                "orchestrator": True,
                "clients_on_central": True,
                "central_on_local": True,
                "log_results": True,
                "save_results": True,
                "save_orchestrator_model": True,
                "save_nodes_model": False,
                "form_archive": True
                },
            "optimizer": {
                "name": "Simple",
                "learning_rate": central_lr}
            },
        "nodes":{
            "local_epochs": local_epochs,
            "model_settings": {
                "optimizer": "SGD",
                "batch_size": batch_size,
                "learning_rate": local_lr,
                "FORCE_CPU": False}}}