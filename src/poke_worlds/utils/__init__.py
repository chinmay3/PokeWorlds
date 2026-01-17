# These are all the utils functions or classes that you may want to import in your project
from poke_worlds.utils.parameter_handling import load_parameters
from poke_worlds.utils.log_handling import log_error, log_info, log_warn, log_dict
from poke_worlds.utils.fundamental import file_makedir, check_optional_installs
from pandas import isna
from typing import Type, List
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

def is_none_str(s) -> bool:
    """
    Checks if a string is None or represents a null value.

    Args:
        s (str or None): The string to check.

    Returns:
        bool: True if the string is None or represents a null value, False otherwise.
    """
    if s is None:
        return True
    if isinstance(s, str):
        options = ["none", "null", "nan", ""]
        for option in options:
            if s.lower() == option:
                return True
    return isna(s)


def nested_dict_to_str(nested_dict: dict, *, indent: int=0, indent_char: str="  ") -> str:
    """
    Converts a nested dictionary to a formatted string representation.
    Example Usage: 
    ```python
    nested_dict={2: 4, 3: {4: 5, 6: {7: 8}}}
    print(nested_dict_to_str(nested_dict))
    2: 4
    3: Dict: 
      4: 5
      6: Dict: 
        7: 8
    ```

    Args:
        nested_dict (dict): The nested dictionary to convert.
        indent (int): The current indentation level.
        indent_char (str): The character(s) used for indentation.
    Returns:
        str: A formatted string representation of the nested dictionary.

    """
    result = ""
    for key, value in nested_dict.items():
        result += indent_char * indent + str(key) + ": "
        if isinstance(value, dict):
            result += "Dict: \n" + nested_dict_to_str(value, indent + 1, indent_char=indent_char)
        else:
            result += str(value) + "\n"
    return result

def verify_parameters(parameters: dict):
    """
    Does a basic sanity check to ensure parameters is a non-empty dictionary.
    """
    if parameters is None:
        raise ValueError("Parameters cannot be None.")
    if not isinstance(parameters, dict):
        raise ValueError("Parameters must be a dictionary.")
    if len(parameters) == 0:
        raise ValueError("Parameters dictionary cannot be empty.")
    

def get_lowest_level_subclass(class_list: List[Type]) -> Type:
    """
    Given a list of classes, returns the class that is the lowest level subclass in the inheritance hierarchy.
    """
    lowest_level_tracker = None
    for cls in class_list:
        if lowest_level_tracker is None:
            lowest_level_tracker = cls
        elif issubclass(cls, lowest_level_tracker):
            lowest_level_tracker = cls
    return lowest_level_tracker


def show_frames(frames: np.ndarray, titles: List[str]=None, save=False, parameters: dict=None):
    """
    Plots each frame as an image in matplotlib. If save is true, will save each frame as title.png in the frame_saves/ directory.
    titles length must be equal to frame length if specified.
    """
    parameters = load_parameters(parameters)
    if isinstance(titles, str):
        titles = [titles]
    if save:
        if titles is None:
            log_error(f"Cannot save frames without titles specified.", parameters)    
    if titles is not None:
        if len(titles) == 1 and len(frames) > 1:
            titles = [titles[0] + f"_{i}" for i in range(len(frames))]
    if titles is not None and len(titles) != len(frames):
        log_error(f"Length of titles {len(titles)} does not match number of frames {len(frames)}", parameters)
    save_dir = "frame_saves/"
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(frames)):
        plt.imshow(frames[i])
        if titles is not None:
            plt.title(titles[i])
        if save:
            filename = os.path.join(save_dir, titles[i].replace(" ", "_").replace("/", "_") + ".png")
            plt.imsave(filename, frames[i][:, :, 0], cmap='gray')
        else:
            plt.show()