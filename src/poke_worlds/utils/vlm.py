"""
Provides query access to a single VLM that is shared across the project
"""
from poke_worlds.utils.fundamental import check_optional_installs
from poke_worlds.utils.parameter_handling import load_parameters
from poke_worlds.utils.log_handling import log_warn, log_error, log_info
from typing import List
import numpy as np
from PIL import Image

def convert_numpy_greyscale_to_pillow(arr: np.ndarray) -> Image:
    """
    Converts a numpy image with shape: H x W x 1 into a Pillow Image

    Args:
        arr: the numpy array

    Returns:
        image: PIL Image
    """
    rgb = np.stack([arr[:, :, 0], arr[:, :, 0], arr[:, :, 0]], axis=2)
    return Image.fromarray(rgb)

project_parameters = load_parameters()
if project_parameters["full_importable"]:
    # Import anything related to full here. 
    import torch
    from transformers import pipeline
else:
    pass

if project_parameters["use_vllm"]:
    if project_parameters["vllm_importable"]:
        from vllm import LLM, SamplingParams
    elif project_parameters["full_importable"]:
        log_warn("Project parameters has `use_vllm` set to True, but vllm is not installed. Run `uv pip install -e \".[full, vllm]\"` to install required packages.", project_parameters)
        project_parameters["use_vllm"] = False
    else:
        pass # Do not warn if neither is installed. 


class HuggingFaceVLM:
    """A class that holds the HuggingFace VLM that is shared across the project"""
    _BATCH_SIZE=8
    _MODEL = None

    @staticmethod
    def start():
        if HuggingFaceVLM._MODEL is not None:
            return
        if not project_parameters[f"full_importable"]:
            log_error(f"Tried to instantiate a HuggingFace VLM, but the required packages are not installed. Run `uv pip install -e \".[full]\"` to install required packages.", project_parameters)
        else:
            HuggingFaceVLM._MODEL = pipeline("image-text-to-text", model=project_parameters["backbone_vlm_model"], device_map="auto", dtype=torch.bfloat16)

    @staticmethod
    def infer(texts: List[str], images: List[np.ndarray], max_new_tokens: int, batch_size: int = None, stop_strings="[STOP]") -> List[str]:
        """
        Performs inference with the given texts and images        
        """
        if HuggingFaceVLM._MODEL is None:
            HuggingFaceVLM.start()
        if max_new_tokens is None:
            log_error(f"Can't set max_new_tokens to None", project_parameters)
        all_images = [convert_numpy_greyscale_to_pillow(img) for img in images]
        all_texts = [f"<|im_start|>user\n<vision_start|><|image_pad|><|vision_end|>\n{text}\n<|im_end|><|im_start|>assistant\n" for text in texts]
        if batch_size is None:
            batch_size = HuggingFaceVLM._BATCH_SIZE
        all_outputs = []
        for i in range(0, len(all_images), batch_size):
            images = all_images[i:i+batch_size]
            texts = all_texts[i:i+batch_size]
            outputs = HuggingFaceVLM._MODEL(images=images, text=texts, max_new_tokens=max_new_tokens, stop_strings=stop_strings)
            all_outputs.extend(outputs)
        output_only = []
        for out in all_outputs:
            output_only.append(out["generated_text"].split("assistant")[-1].strip())
        return output_only

class vLLMVLM:
    """ A class that holds the vLLM VLM shared across the project """
    _BATCH_SIZE = 8
    _MODEL = None

    @staticmethod
    def start():
        _MODEL = LLM(model=project_parameters["backbone_vlm_model"])
        pass # Start the VLM


def perform_vlm_inference(texts: List[str], images: List[np.array], max_new_tokens: int, batch_size: int = None, stop_strings="[STOP]"):
    """
    Routes to the correct VLM class and performs inference
    """
    parameters = project_parameters
    if parameters["use_vllm"]:
        return vLLMVLM.infer(texts=texts, images=images, max_new_tokens=max_new_tokens, batch_size=batch_size, stop_strings=stop_strings)    
    else:
        return HuggingFaceVLM.infer(texts=texts, images=images, max_new_tokens=max_new_tokens, batch_size=batch_size, stop_strings=stop_strings)


def _ocr_merge(texts: List[str]) -> List[str]:
    """
    Merges OCR texts by removing duplicates and combining them into a single string.
    Args:
        texts: List of OCR text strings.
    Returns:
        List containing merged text strings.
    """
    final_strings = []
    for text in texts:
        if len(final_strings) == 0:
            final_strings.append(text)
        else:
            reference = final_strings[-1]
            if text in reference:
                continue
            elif reference in text:
                final_strings[-1] = text
            else:
                final_strings.append(text)
    return final_strings

def ocr(images: List[np.array], do_merge: bool=True) -> List[str]:
    """
    Performs OCR on the given images using the VLM.

    Args:
        images: List of images in numpy array format (H x W x C)
        do_merge: Whether to merge similar OCR results. Use this if images are sequential frames from a game.
    Returns:
        List of extracted text strings. May contain duplicates if images have frames containing the same text.
    """
    parameters = project_parameters
    text_prompt = "Perform OCR and state the text in this image:"
    texts = [text_prompt] * len(images)
    batch_size = parameters["ocr_batch_size"]
    max_new_tokens = parameters["ocr_max_new_tokens"]
    ocred = perform_vlm_inference(texts=texts, images=images, max_new_tokens=max_new_tokens, batch_size=batch_size)
    if do_merge:
        ocred = _ocr_merge(ocred)
    return ocred