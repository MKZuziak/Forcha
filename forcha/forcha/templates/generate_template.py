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

def basic_evaluator(iterations: int,
                    number_of_nodes: int,
                    sample_size: int,
                    root_path: str,
                    local_lr: float = 0.01,
                    central_lr: float = 0.5,
                    local_epochs: int = 2,
                    batch_size: int = 32,
                    LOO: bool = True,
                    LSAA: bool = True,
                    EXLSAA: bool = False,
                    search_length: int = 1):
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
                "learning_rate": central_lr},
            "evaluator" : {
            "LOO_OR": False,
            "Shapley_OR": False,
            "IN_SAMPLE_LOO": LOO,
            "IN_SAMPLE_SHAP": False,
            "LSAA": LSAA,
            "EXTENDED_LSAA": EXLSAA,
            "line_search_length": search_length,
            "preserve_evaluation": {
                "preserve_partial_results": True,
                "preserve_final_results": True},
            "full_debug": False,
            "number_of_workers": 50}
            },
        "nodes":{
            "local_epochs": local_epochs,
            "model_settings": {
                "optimizer": "SGD",
                "batch_size": batch_size,
                "learning_rate": local_lr,
                "FORCE_CPU": False}}}