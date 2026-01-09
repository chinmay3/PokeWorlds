from poke_worlds.utils import load_parameters, log_warn, log_error, log_info
from poke_worlds.utils.vlm import convert_numpy_greyscale_to_pillow, HuggingFaceVLM
from typing import List, Union
import numpy as np
from PIL import Image
import torch

project_parameters = load_parameters()
if project_parameters["full_importable"]:
    # Import anything related to full here. 
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
else:
    pass

project_parameters["warned_debug_lm"] = False
class ExecutorVLM:
    """A class that holds the Executor VLM that is potentially the same as HuggingFaceVLM"""
    _MODEL = None
    _PROCESSOR = None

    @staticmethod
    def start():
        """
        Initializes the Executor VLM model and processor if not already done.
        """
        if ExecutorVLM._MODEL is not None:
            return
        if not project_parameters["full_importable"] or project_parameters["debug_skip_lm"]:
            if not project_parameters["debug_mode"]:
                log_error(f"Tried to instantiate an Executor VLM, but the required packages are not installed. Run `uv pip install -e \".[full]\"` to install required packages.", project_parameters)
            else:
                if not project_parameters["warned_debug_lm"]:
                    log_warn(f"Tried to instantiate an Executor VLM, but the required packages are not installed. Running in dev mode, so all LM calls will return a placeholder string.", project_parameters)
                    project_parameters["warned_debug_lm"] = True
        else:
            model_name = project_parameters["executor_vlm_model"]
            backbone_name = project_parameters["backbone_vlm_model"]
            if model_name == backbone_name:
                log_info(f"ExecutorVLM using shared model with BackboneVLM: {model_name}", project_parameters)
                HuggingFaceVLM.start()
                ExecutorVLM._MODEL = HuggingFaceVLM._MODEL
                ExecutorVLM._PROCESSOR = HuggingFaceVLM._PROCESSOR
            else:
                log_info(f"Starting ExecutorVLM with model: {model_name}", project_parameters)
                ExecutorVLM._PROCESSOR = AutoProcessor.from_pretrained(model_name, padding_side="left")
                ExecutorVLM._MODEL = AutoModelForImageTextToText.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto")

    @staticmethod
    def infer(text: str, image: np.ndarray, max_new_tokens: int) -> str:
        """
        Performs inference with the given text and image  

        :param text: The input text prompt.
        :param image: The input image as a numpy array.
        :param max_new_tokens: The maximum number of new tokens to generate.
        :return: The generated text output.
        :rtype: str      
        """
        if ExecutorVLM._MODEL is None:
            ExecutorVLM.start()
        if ExecutorVLM._MODEL is None: # it is only still None in debug mode
            return "LM Output"
        if max_new_tokens is None:
            log_error(f"Can't set max_new_tokens to None", project_parameters)
        return HuggingFaceVLM.do_infer(ExecutorVLM._MODEL, ExecutorVLM._PROCESSOR, [text], [image], max_new_tokens, 1)[0]
    
    @staticmethod
    def multi_infer(text: str, images: List[np.ndarray], max_new_tokens: int) -> str:
        """
        Performs inference with the given text and multiple images        

        :param text: The input text prompt.
        :param images: The list of input images as numpy arrays.
        :param max_new_tokens: The maximum number of new tokens to generate.
        :return: The generated text output.
        :rtype: str
        """
        if ExecutorVLM._MODEL is None:
            ExecutorVLM.start()
        if ExecutorVLM._MODEL is None: # it is only still None in debug mode
            return "LM Output"
        if max_new_tokens is None:
            log_error(f"Can't set max_new_tokens to None", project_parameters)
        return HuggingFaceVLM.do_multi_infer(ExecutorVLM._MODEL, ExecutorVLM._PROCESSOR, [text], [images], max_new_tokens, 1)[0]