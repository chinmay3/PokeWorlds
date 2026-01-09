"""
Provides query access to a single VLM that is shared across the project
"""
from poke_worlds.utils.parameter_handling import load_parameters
from poke_worlds.utils.log_handling import log_warn, log_error, log_info
from typing import List, Union
import numpy as np
from PIL import Image
import torch


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

_project_parameters = load_parameters()
if _project_parameters["full_importable"]:
    # Import anything related to full here. 
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
else:
    pass

if _project_parameters["use_vllm"]:
    raise NotImplementedError(f"vLLM just doesn't work. Need to sort out installation issues.")
    if _project_parameters["vllm_importable"]:
        from vllm import LLM, SamplingParams, EngineArgs
        
    elif _project_parameters["full_importable"]:
        log_warn("Project parameters has `use_vllm` set to True, but vllm is not installed. Run `uv pip install -e \".[full, vllm]\"` to install required packages.", _project_parameters)
        _project_parameters["use_vllm"] = False
    else:
        pass # Do not warn if neither is installed. 

_project_parameters["warned_debug_llm"] = False
class HuggingFaceVLM:
    """A class that holds the HuggingFace VLM that is shared across the project"""
    _BATCH_SIZE=8
    _MODEL = None
    _PROCESSOR = None

    @staticmethod
    def start():
        if HuggingFaceVLM._MODEL is not None:
            return
        if not _project_parameters[f"full_importable"] or _project_parameters["debug_skip_lm"]:
            if not _project_parameters["debug_mode"]:
                log_error(f"Tried to instantiate a HuggingFace VLM, but the required packages are not installed. Run `uv pip install -e \".[full]\"` to install required packages.", _project_parameters)
            else:
                if not _project_parameters["warned_debug_llm"]:
                    log_warn(f"Tried to instantiate a HuggingFace VLM, but the required packages are not installed. Running in dev mode, so all LM calls will return a placeholder string.", _project_parameters)
                    _project_parameters["warned_debug_llm"] = True
        else:
            log_info(f"Loading Backbone HuggingFace VLM model: {_project_parameters['backbone_vlm_model']}", _project_parameters)
            HuggingFaceVLM._MODEL = AutoModelForImageTextToText.from_pretrained(_project_parameters["backbone_vlm_model"], dtype=torch.bfloat16, device_map="auto")
            HuggingFaceVLM._PROCESSOR = AutoProcessor.from_pretrained(_project_parameters["backbone_vlm_model"], padding_side="left")

    
    @staticmethod
    def do_infer(model, processor, texts, images, max_new_tokens, batch_size):
        all_images = [convert_numpy_greyscale_to_pillow(img) for img in images]
        all_texts = [f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{text}\n<|im_end|><|im_start|>assistant\n" for text in texts]
        if batch_size is None:
            batch_size = HuggingFaceVLM._BATCH_SIZE
        all_outputs = []
        for i in range(0, len(all_images), batch_size):
            images = all_images[i:i+batch_size]
            texts = all_texts[i:i+batch_size]
            inputs = processor(text=texts, images=images, padding=True, truncation=True, return_tensors="pt").to(model.device)
            input_length = inputs["input_ids"].shape[1]
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, repetition_penalty=1.2, stop_strings=["[STOP]"], tokenizer=processor.tokenizer)
            output_only = outputs[:, input_length:]
            decoded_outputs = processor.batch_decode(output_only, skip_special_tokens=True)
            all_outputs.extend(decoded_outputs)
        return all_outputs        

    
    @staticmethod
    def infer(texts: List[str], images: List[np.ndarray], max_new_tokens: int, batch_size: int = None) -> List[str]:
        """
        Performs inference with the given texts and images        
        """
        if HuggingFaceVLM._MODEL is None:
            HuggingFaceVLM.start()
        if HuggingFaceVLM._MODEL is None: # it is only still None in debug mode
            return ["LM Output" for text in texts]
        if max_new_tokens is None:
            log_error(f"Can't set max_new_tokens to None", _project_parameters)
        return HuggingFaceVLM.do_infer(HuggingFaceVLM._MODEL, HuggingFaceVLM._PROCESSOR, texts, images, max_new_tokens, batch_size)

    @staticmethod
    def do_multi_infer(model, processor, texts, images, max_new_tokens, batch_size):
        # Might need to branch this function out based on the VLM kind later. 
        all_outputs = []
        all_texts = []
        for i, text in enumerate(texts):
            full_text = f"<|im_start|>user\n"
            for j in range(len(images[i])):
                full_text += f"Picture: {i+1}<|vision_start|><|image_pad|><|vision_end|>"
            full_text += f"\n{text}\n<|im_end|><|im_start|>assistant\n"
            all_texts.append(full_text)
        for i in range(0, len(all_texts), batch_size):
            batch_images = images[i:i+batch_size]
            flat_images = []
            for img_list in batch_images:
                for img in img_list:
                    if isinstance(img, np.ndarray):
                        flat_images.append(convert_numpy_greyscale_to_pillow(img))
                    else:
                        flat_images.append(img)                    
            batch_texts = all_texts[i:i+batch_size]
            inputs = processor(text=batch_texts, images=flat_images, padding=True, truncation=True, return_tensors="pt").to(model.device)
            input_length = inputs["input_ids"].shape[1]
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, repetition_penalty=1.2, stop_strings=["[STOP]"], tokenizer=processor.tokenizer)
            output_only = outputs[:, input_length:]
            decoded_outputs = processor.batch_decode(output_only, skip_special_tokens=True)
            all_outputs.extend(decoded_outputs)
        return all_outputs
        

    @staticmethod
    def multi_infer(texts: List[str], images: List[List[Union[np.ndarray, Image.Image]]], max_new_tokens: int, batch_size: int = None) -> List[str]:
        """
        Performs inference with the a single text and multiple images
        """
        if len(texts) != len(images):
            log_error(f"Texts and images must have the same length. Got {len(texts)} texts and {len(images)} image lists.", _project_parameters)
        if HuggingFaceVLM._MODEL is None:
            HuggingFaceVLM.start()
        if HuggingFaceVLM._MODEL is None: # it is only still None in debug mode
            return ["LM Output" for _ in images]
        if max_new_tokens is None:
            log_error(f"Can't set max_new_tokens to None", _project_parameters)
        if batch_size is None:
            batch_size = HuggingFaceVLM._BATCH_SIZE
        return HuggingFaceVLM.do_multi_infer(HuggingFaceVLM._MODEL, HuggingFaceVLM._PROCESSOR, texts, images, max_new_tokens, batch_size)


class vLLMVLM:
    """ A class that holds the vLLM VLM shared across the project """
    _BATCH_SIZE = 8
    _MODEL = None

    @staticmethod
    def start():
        if "qwen3" not in _project_parameters["backbone_vlm_model"]:
            log_warn(f"You are using a non Qwen3 model with the vLLM VLM class. This has not been tested.", _project_parameters)
        engine_args = EngineArgs(
            model=_project_parameters["backbone_vlm_model"],
            max_model_len=4096,
            max_num_seqs=5,
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
                "fps": 1,
            }, # copied from: https://docs.vllm.ai/en/latest/examples/offline_inference/vision_language/?h=vision+language. Unsure if I need it
            limit_mm_per_prompt={"image": 1},
        )
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

    
def perform_object_detection(images: List[np.ndarray], texts: List[List[str]]) -> List[bool]:
    """
    Performs object detection on the given images with the given texts.
    
    :param images: List of images that may contain the object described in texts
    :type images: List[np.ndarray]
    :param texts: Prompts that not only describe the object to be detected, but also instruct the VLM to answer Yes or No if the object is present in the corresponding image.
    :type texts: List[List[str]]
    :return: List of booleans indicating whether the object was detected in each image.
    :rtype: List[bool]
    """
    outputs = perform_vlm_inference(texts=texts, images=images, max_new_tokens=60)
    founds = []
    for i, output in enumerate(outputs):
        if "yes" in output.lower():
            founds.append(True)
        else:
            founds.append(False)
    return founds




def perform_vlm_inference(texts: List[str], images: List[np.array], max_new_tokens: int, batch_size: int = None) -> List[str]:
    """
    Routes to the correct VLM class and performs inference
    
    :param texts: Input prompts
    :type texts: List[str]
    :param images: Input images
    :type images: List[np.array]
    :param max_new_tokens: maximum number of new tokens to generate
    :type max_new_tokens: int
    :param batch_size: Batch size for inference
    :type batch_size: int
    :return: List of output strings from the VLM
    :rtype: List[str]
    """
    parameters = _project_parameters
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

def ocr(images: List[np.ndarray], *, text_prompt=None, do_merge: bool=True) -> List[str]:
    """
    Performs OCR on the given images using the VLM.

    Args:
        images: List of images in numpy array format (H x W x C)
        text_prompt: The prompt to use for the OCR model.
        do_merge: Whether to merge similar OCR results. Use this if images are sequential frames from a game.
    Returns:
        List of extracted text strings. May contain duplicates if images have frames containing the same text.
    """
    if text_prompt is None:
        text_prompt = "If there is no text in the image, just say NONE. Otherwise, perform OCR and state the text in this image:"
    parameters = _project_parameters
    batch_size = parameters["ocr_batch_size"]
    max_new_tokens = parameters["ocr_max_new_tokens"]
    use_images = []
    for image in images:
        if image.mean() < 234:
            use_images.append(image)
    if len(use_images) == 0:
        return []
    texts = [text_prompt] * len(use_images)    
    ocred = perform_vlm_inference(texts=texts, images=use_images, max_new_tokens=max_new_tokens, batch_size=batch_size)
    for i, res in enumerate(ocred):
        if res.strip().lower() == "none":
            log_warn(f"Got NONE as output from OCR. Could this have been avoided?\nimages statistics: {images[i].max(), images[i].min(), images[i].mean(), (images[i] > 0).mean()}", _project_parameters)
    ocred = [text.strip() for text in ocred if text.strip().lower() != "none"]
    if do_merge:
        ocred = _ocr_merge(ocred)
    return ocred

def identify_matches(description: str, screens: List[np.ndarray], reference: Image.Image) -> List[bool]:
    """
    Identifies which screens match the given reference image based on the description.
    Args:
        description: A textual description of the target object.
        screens: A list of screen images in numpy array format (H x W x C).
        reference: A PIL Image of the reference object.

    Returns:
        A list of booleans indicating whether each screen contains the target object.
    """
    texts = [f"The target, described as {description} is shown as reference in Picture 1. Does Picture 2 contain the object from Picture 1 in it? Answer in the following format: \nExplanation: <briefly describe what is in Picture 2, with reference to the image in Picture 1>\nAnswer: <Yes or No>[STOP]" for _ in screens]
    images = []
    for screen in screens:
        images.append([reference, screen])
    outputs = HuggingFaceVLM.multi_infer(texts=texts, images=images, max_new_tokens=120)
    results = []
    for output in outputs:
        if "yes" in output.lower():
            results.append(True)
        else:
            results.append(False)
    return results