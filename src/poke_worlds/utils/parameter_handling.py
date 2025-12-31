import os
import yaml
from poke_worlds.utils.fundamental import get_logger, check_optional_installs


def load_yaml(yaml_path: str) -> dict:
    """
    Loads a yaml file and returns the contents as a dictionary.
    Args:
        yaml_path (str): Path to the yaml file.
    Returns:
        dict: Contents of the yaml file.
    """
    with open(yaml_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def compute_secondary_parameters(params: dict):
    """
    Computes secondary parameters based on the primary parameters.
    This sets up the directory structure for the project, including data, model, tmp, sync, and log directories.
    It also initializes the logger and adds it to the parameters dictionary.

    Args:
        params (dict): Primary parameters dictionary.
    """
    params["rom_data_dir"] = os.path.join(params["storage_dir"], "rom_data")
    params["log_dir"] = os.path.join(params["storage_dir"], "logs")
    for dirname in ["rom_data_dir", "log_dir"]:
        if not os.path.exists(params[dirname]):
            os.makedirs(params[dirname])
    if "log_file" not in params:
        log_file = os.path.join(params["log_dir"], "log.txt")
        params["log_file"] = log_file
    else:
        # check if log_file is a child of log_dir, but handle silly // vs / cases
        log_dir_str = params["log_dir"].replace("//", "/")
        log_file_str = params["log_file"].replace("//", "/")
        if not log_file_str.startswith(log_dir_str):
            log_file = os.path.join(params["log_dir"], params["log_file"])
            params["log_file"] = log_file
    logger = get_logger(filename=params["log_file"])
    params["logger"] = logger
    configs = check_optional_installs()
    for config in configs:
        params[f"{config}_importable"] = configs[config]
    # convert all rom_data_paths to absolute paths
    for key in params:
        if key.endswith("_rom_data_path"):
                params[key] = os.path.abspath(os.path.join(params["rom_data_dir"], params[key]))
    if params["debug_skip_lm"]:
        if not params["debug_mode"]:
            logger.error("Can only set `debug_skip_lm` to True in configs if you are in debug mode. Set `debug_mode` to True in configs")




def load_parameters(parameters: dict = None) -> dict:
    """
    Loads the parameters for the project from configs/private_vars.yaml and any other yaml files in the configs directory.

    That is, unless a non None parameters dictionary is passed through, in which case we assume all is good and just return it.

    Args:
        parameters (dict, optional): If provided, this dictionary is returned as the parameters.
            If None, parameters are loaded from the config files. Defaults to None.

    Returns:
        dict: Parameters dictionary.
    """
    if parameters is not None:
        if "logger" not in parameters: # this is a flag that secondary parameters need to be computed
            compute_secondary_parameters(parameters)
        return parameters
    essential_keys = ["storage_dir"]
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    params = {"project_root": project_root}
    logger = get_logger()
    config_files = os.listdir(os.path.join(project_root, "configs"))
    def error(msg):
        logger.error(msg)
        raise ValueError(msg)

    if "private_vars.yaml" not in config_files:
        error("Please create private_vars.yaml in the configs directory")
    for file in config_files:
        if file.endswith(".yaml"):
            configs = load_yaml(os.path.join(project_root, "configs", file))
            for key in configs:
                if key in params:
                    error(f"{key} is present in multiple config files. At least one of which is {file}. Please remove the duplicate")
            params.update(configs)
        else:
            pass

    for key in params:
        if params[key] == "PLACEHOLDER":
            error(f"{key} is currently the placeholder value in private_vars.yaml. Please set it")
    for essential_key in essential_keys:
        if essential_key not in params:
            error(f"Please set {essential_key} in one of the config yamls")
    # check if there are any .py files in storage_dir, if so, log error
    if os.path.exists(params['storage_dir']):
        if any([f.endswith(".py") for f in os.listdir(params["storage_dir"])]):
            logger.warning(f"There are .py files in the storage_dir {params['storage_dir']}. It is recommended to set a path which has nothing else inside it to avoid issues.")
    else:
        full_path = os.path.abspath(params["storage_dir"])        
        if params["storage_dir"] == "storage":
            logger.warning(f"Using default storage directory '{full_path}'. This may cause issues if your project root directory has limited space. To change the storage directory, modify the 'storage_dir' parameter in your config files and run this method again.")
        os.makedirs(full_path)
        logger.info(f"Created storage directory {full_path}. You will find a {full_path}/rom_data/ directory inside it, which is where you must place your downloaded ROM (.gb or .gbc) files.")
    # For every path, see if it looks relative, and if so, make it absolute based on project_root
    for key in params:
        if isinstance(params[key], str):
            try_path = os.path.join(project_root, params[key])
            if os.path.exists(try_path):
                params[key] = os.path.abspath(try_path)
    compute_secondary_parameters(params)
    return params
