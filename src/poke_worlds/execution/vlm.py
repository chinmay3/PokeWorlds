from poke_worlds.utils import load_parameters, log_warn, log_error, log_info
from poke_worlds.utils.fundamental import check_optional_installs
from typing import List, Union
import numpy as np
from PIL import Image
import torch

_project_parameters = load_parameters()
configs = check_optional_installs(warn=True)
for config in configs:
    _project_parameters[f"{config}_importable"] = configs[config]

if _project_parameters["transformers_importable"]:
    # Import anything related to transformers here. 
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
else:
    pass

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


class ExecutorVLM:
    """
    A class that holds the VLM for the executor

    TODO: This function assumes that the model follows a Qwen3-VL style format. May need to branch based on model kind later.
    """
    _BATCH_SIZE=8
    _MODEL = None
    _PROCESSOR = None
    _MODEL_KIND = None

    @staticmethod
    def start():
        if ExecutorVLM._MODEL is not None:
            return
        if _project_parameters["debug_skip_lm"]:
            if not _project_parameters["warned_debug_llm"]:
                log_warn(f"Skipping VLM initialization as per debug_skip_lm=True", _project_parameters)
                _project_parameters["warned_debug_llm"] = True
            return
        elif not _project_parameters["transformers_importable"]:
            if not _project_parameters["warned_debug_llm"]:
                log_warn(f"Tried to instantiate a Executor VLM, but the required packages are not installed. Running in dev mode, so all LM calls will return a placeholder string.", _project_parameters)
                _project_parameters["warned_debug_llm"] = True
            return
        else:
            log_info(f"Loading Backbone Executor VLM model: {_project_parameters['executor_vlm_model']}", _project_parameters)
            ExecutorVLM._MODEL_KIND = _project_parameters["executor_vlm_kind"]
            if ExecutorVLM._MODEL_KIND not in ["qwen3vl"]:
                log_error(f"Unsupported executor_vlm_kind: {ExecutorVLM._MODEL_KIND}", _project_parameters)
            if ExecutorVLM._MODEL_KIND in ["qwen3vl"]:
                ExecutorVLM._MODEL = AutoModelForImageTextToText.from_pretrained(_project_parameters["executor_vlm_model"], dtype=torch.bfloat16, device_map="auto")
                ExecutorVLM._PROCESSOR = AutoProcessor.from_pretrained(_project_parameters["executor_vlm_model"], padding_side="left")
            # this way, we can add more model kinds w different engines (e.g. OpenAI API) later


    
    @staticmethod
    def do_infer(vlm_kind: str, model, processor, texts: List[str], images: List[np.ndarray], max_new_tokens: int, batch_size: int) -> List[str]:
        """
        Performs inference with the given texts and images. 
        """
        if vlm_kind == "qwen3vl":
            all_images = [convert_numpy_greyscale_to_pillow(img) for img in images]
            all_texts = [f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{text}\n<|im_end|><|im_start|>assistant\n" for text in texts]
            if batch_size is None:
                batch_size = ExecutorVLM._BATCH_SIZE
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
        else:
            log_error(f"Unsupported HuggingFace VLM kind: {ExecutorVLM._MODEL_KIND}", _project_parameters)

    
    @staticmethod
    def infer(texts: Union[List[str], str], images: Union[np.ndarray, List[np.ndarray]], max_new_tokens: int, batch_size: int = None) -> List[str]:
        """
        Performs inference with the given texts and images        

        :param texts: List of text prompts or a single text prompt
        :type texts: Union[List[str], str]
        :param images: List of images in numpy array format (H x W x C) or a single image in numpy array format
        :type images: Union[np.ndarray, List[np.ndarray]]
        :param max_new_tokens: Maximum number of new tokens to generate
        :type max_new_tokens: int
        :param batch_size: Batch size for inference
        :type batch_size: int, optional
        :return: List of generated text outputs
        :rtype: List[str]
        """
        if ExecutorVLM._MODEL is None:
            ExecutorVLM.start()
        if ExecutorVLM._MODEL is None: # it is only still None in debug mode
            return ["LM Output" for text in texts]
        if max_new_tokens is None:
            log_error(f"Can't set max_new_tokens to None", _project_parameters)
        if isinstance(texts, str):
            texts = [texts]
            # then images must either be a single image in a list, or an array of shape (H, W, C), or a stack of shape (1, H, W, C)
            if isinstance(images, list):
                if len(images) != 1:
                    log_error(f"When passing a single text string, images must be a single image in a list. Got {len(images)} images.", _project_parameters)
            elif isinstance(images, np.ndarray):
                if images.ndim == 3:
                    images = [images]
                elif images.ndim == 4 and images.shape[0] == 1:
                    images = [images[0]]
                else:
                    log_error(f"When passing a single text string, images must be a single image in a list or an array of shape (H, W, C) or (1, H, W, C). Got array of shape {images.shape}.", _project_parameters)
            else:
                log_error(f"When passing a single text string, images must be a single image in a list or an array of shape (H, W, C) or (1, H, W, C). Got type {type(images)}.", _project_parameters)
        return ExecutorVLM.do_infer(ExecutorVLM._MODEL_KIND, ExecutorVLM._MODEL, ExecutorVLM._PROCESSOR, texts, images, max_new_tokens, batch_size)

    @staticmethod
    def do_multi_infer(vlm_kind: str, model, processor, texts, images, max_new_tokens, batch_size):
        if vlm_kind == "qwen3vl":
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
        else:
            log_error(f"Unsupported HuggingFace VLM kind: {ExecutorVLM._MODEL_KIND}", _project_parameters)
        

    @staticmethod
    def multi_infer(texts: Union[List[str], str], images: Union[List[List[Union[np.ndarray, Image.Image]]], List[Union[np.ndarray, Image.Image]]], max_new_tokens: int, batch_size: int = None) -> List[str]:
        """
        Performs inference with the a single text and multiple images

        :param texts: List of text prompts or a single text prompt
        :type texts: Union[List[str], str]
        :param images: List of lists of images in numpy array format (H x W x C) or a single list of images in numpy array or Pillow Image format
        :type images: Union[List[List[Union[np.ndarray, Image.Image]]], List[Union[np.ndarray, Image.Image]]]
        :param max_new_tokens: Maximum number of new tokens to generate
        :type max_new_tokens: int
        :param batch_size: Batch size for inference
        :type batch_size: int, optional
        :return: List of generated text outputs
        :rtype: List[str]
        """
        if isinstance(texts, str):
            texts = [texts]
            # then images must be a list whose element is NOT a list
            if not isinstance(images, list) or len(images) == 0:
                log_error(f"When passing a single text string, images must be a nonempty list of images. Got type {type(images)}.", _project_parameters)
            if isinstance(images[0], list):
                log_error(f"When passing a single text string, images must be a single list of images, not a list of lists. Got list of lists with length {len(images)}.", _project_parameters)
            images = [images]  # wrap in another list to make it a list of lists
            
        if len(texts) != len(images):
            log_error(f"Texts and images must have the same length. Got {len(texts)} texts and {len(images)} image lists.", _project_parameters)
        if ExecutorVLM._MODEL is None:
            ExecutorVLM.start()
        if ExecutorVLM._MODEL is None: # it is only still None in debug mode
            return ["LM Output" for _ in images]
        if max_new_tokens is None:
            log_error(f"Can't set max_new_tokens to None", _project_parameters)
        if batch_size is None:
            batch_size = ExecutorVLM._BATCH_SIZE
        return ExecutorVLM.do_multi_infer(ExecutorVLM._MODEL_KIND, ExecutorVLM._MODEL, ExecutorVLM._PROCESSOR, texts, images, max_new_tokens, batch_size)
    
def merge_ocr_strings(strings, min_overlap=3):
    """
    Merges a list of strings by removing subsets and combining overlapping fragments.
    
    Written by Gemini3 Pro, and I have not tested it yet. 
    
    Args:
        strings (list): List of strings from OCR.
        min_overlap (int): Minimum characters required to consider two strings an overlap.
    """
    # 1. Clean up: Remove exact duplicates and empty strings
    current_strings = list(set(s.strip() for s in strings if s.strip()))

    # 2. Remove subsets (if "Hello" is in "Hello World", remove "Hello")
    # Sorting by length descending ensures we check smaller strings against larger ones
    current_strings.sort(key=len, reverse=True)
    final_set = []
    for s in current_strings:
        if not any(s in other for other in final_set):
            final_set.append(s)
    
    # 3. Iterative Overlap Merging
    # We use a while loop because merging two strings might create a new 
    # string that can then be merged with a third string.
    merged_list = final_set[:]
    changed = True
    
    while changed:
        changed = False
        i = 0
        while i < len(merged_list):
            j = 0
            while j < len(merged_list):
                if i == j:
                    j += 1
                    continue
                
                s1, s2 = merged_list[i], merged_list[j]
                
                # Check if suffix of s1 matches prefix of s2
                overlap_len = 0
                max_possible_overlap = min(len(s1), len(s2))
                
                for length in range(max_possible_overlap, min_overlap - 1, -1):
                    if s1.endswith(s2[:length]):
                        overlap_len = length
                        break
                
                if overlap_len > 0:
                    # Create the merged string
                    new_string = s1 + s2[overlap_len:]
                    
                    # Remove the two old strings and add the new one
                    # We use indices carefully or rebuild the list
                    val_i = merged_list[i]
                    val_j = merged_list[j]
                    merged_list.remove(val_i)
                    merged_list.remove(val_j)
                    merged_list.append(new_string)
                    
                    changed = True
                    # Reset indices to restart search with the new combined string
                    i = -1 
                    break 
                j += 1
            if changed: break
            i += 1
            
    return merged_list

def _ocr_merge(texts: List[str]) -> List[str]:
    """
    Merges OCR texts by removing duplicates, subsets and combining them into a single string.
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
    texts = [text_prompt] * len(images)    
    ocred = ExecutorVLM.infer(texts=texts, images=images, max_new_tokens=max_new_tokens, batch_size=batch_size)
    for i, res in enumerate(ocred):
        if res.strip().lower() == "none":
            log_warn(f"Got NONE as output from OCR. Could this have been avoided?\nimages statistics: {images[i].max(), images[i].min(), images[i].mean(), (images[i] > 0).mean()}", _project_parameters)
    ocred = [text.strip() for text in ocred if text.strip().lower() != "none"]
    if do_merge:
        ocred = merge_ocr_strings(ocred)
    return ocred
