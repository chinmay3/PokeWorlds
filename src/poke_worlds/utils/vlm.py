"""
Provides query access to a single VLM that is shared across the project
"""
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
    from transformers import AutoModelForImageTextToText, AutoProcessor
else:
    pass

if project_parameters["use_vllm"]:
    if project_parameters["vllm_importable"]:
        from vllm import LLM, SamplingParams, EngineArgs

        
    elif project_parameters["full_importable"]:
        log_warn("Project parameters has `use_vllm` set to True, but vllm is not installed. Run `uv pip install -e \".[full, vllm]\"` to install required packages.", project_parameters)
        project_parameters["use_vllm"] = False
    else:
        pass # Do not warn if neither is installed. 


class HuggingFaceVLM:
    """A class that holds the HuggingFace VLM that is shared across the project"""
    _BATCH_SIZE=8
    _MODEL = None
    _PROCESSOR = None

    @staticmethod
    def start():
        if HuggingFaceVLM._MODEL is not None:
            return
        if not project_parameters[f"full_importable"]:
            log_error(f"Tried to instantiate a HuggingFace VLM, but the required packages are not installed. Run `uv pip install -e \".[full]\"` to install required packages.", project_parameters)
        else:
            HuggingFaceVLM._MODEL = AutoModelForImageTextToText.from_pretrained(project_parameters["backbone_vlm_model"], dtype=torch.bfloat16, device_map="auto")
            HuggingFaceVLM._PROCESSOR = AutoProcessor.from_pretrained(project_parameters["backbone_vlm_model"], padding_side="left")

    @staticmethod
    def infer(texts: List[str], images: List[np.ndarray], max_new_tokens: int, batch_size: int = None) -> List[str]:
        """
        Performs inference with the given texts and images        
        """
        if HuggingFaceVLM._MODEL is None:
            HuggingFaceVLM.start()
        if max_new_tokens is None:
            log_error(f"Can't set max_new_tokens to None", project_parameters)
        all_images = [convert_numpy_greyscale_to_pillow(img) for img in images]
        all_texts = [f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{text}\n<|im_end|><|im_start|>assistant\n" for text in texts]
        if batch_size is None:
            batch_size = HuggingFaceVLM._BATCH_SIZE
        all_outputs = []
        for i in range(0, len(all_images), batch_size):
            images = all_images[i:i+batch_size]
            texts = all_texts[i:i+batch_size]
            inputs = HuggingFaceVLM._PROCESSOR(text=texts, images=images, padding=True, truncation=True, return_tensors="pt").to(HuggingFaceVLM._MODEL.device)
            input_length = inputs["input_ids"].shape[1]
            outputs = HuggingFaceVLM._MODEL.generate(**inputs, max_new_tokens=max_new_tokens, repetition_penalty=1.2, stop_strings=["[STOP]"], tokenizer=HuggingFaceVLM._PROCESSOR.tokenizer)
            output_only = outputs[:, input_length:]
            decoded_outputs = HuggingFaceVLM._PROCESSOR.batch_decode(output_only, skip_special_tokens=True)
            all_outputs.extend(decoded_outputs)
        return all_outputs

class vLLMVLM:
    """ A class that holds the vLLM VLM shared across the project """
    _BATCH_SIZE = 8
    _MODEL = None

    @staticmethod
    def start():
        if "qwen3" not in project_parameters["backbone_vlm_model"]:
            log_warn(f"You are using a non Qwen3 model with the vLLM VLM class. This has not been tested.", project_parameters)
        engine_args = {
            "model": project_parameters["backbone_vlm_model"],
            "max_model_len": 4096,
            "limit_mm_per_prompt": {"image": 1},
        }        
        vLLMVLM._MODEL = LLM(**engine_args)

    @staticmethod
    def infer(texts: List[str], images: List[np.ndarray], max_new_tokens: int, batch_size: int = None) -> List[str]:
        if vLLMVLM._MODEL is None:
            vLLMVLM.start()
        images = [convert_numpy_greyscale_to_pillow(image) for image in images]
        
        params = SamplingParams(
            max_tokens=max_new_tokens
        )

        placeholder = "<|image_pad|>"

        prompts = [
            (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
                f"{text}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            for text in texts
        ]
        inputs = []
        for i in range(len(prompts)):
            inputs.append({
                "prompt": prompts[i],
                "multi_modal_data": {"image": images[i]},
                "multi_modal_uuids": {"image": f"uuid_{i}"}
            })

        outputs = vLLMVLM._MODEL.generate(
            inputs,
            sampling_params=params,
        )
        final_texts = []
        for output in outputs:
            final_texts.append(output.outputs[0].text)
        return final_texts


def perform_vlm_inference(texts: List[str], images: List[np.array], max_new_tokens: int, batch_size: int = None):
    """
    Routes to the correct VLM class and performs inference
    """
    parameters = project_parameters
    if parameters["use_vllm"]:
        return vLLMVLM.infer(texts=texts, images=images, max_new_tokens=max_new_tokens, batch_size=batch_size)    
    else:
        return HuggingFaceVLM.infer(texts=texts, images=images, max_new_tokens=max_new_tokens, batch_size=batch_size)


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