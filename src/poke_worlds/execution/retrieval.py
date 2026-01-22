from poke_worlds.utils import load_parameters, log_error, log_info, log_warn
from poke_worlds.execution.vlm import ExecutorVLM
import torch
import os
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple, Any, Dict
import numpy as np

project_parameters = load_parameters()


class TextEmbeddingModel:
    _MODEL = None
    _PROCESSOR = None
    _has_warned = False
    _random_embed_size = 2

    def start():
        if TextEmbeddingModel._MODEL is not None:
            return
        if project_parameters["debug_skip_lm"]:
            if TextEmbeddingModel._has_warned:
                return
            log_warn(
                f"Tried to instantiate TextEmbeddingModel while debug_skip_lm is True. Skipping LM load and will return random values ...",
                project_parameters,
            )
            TextEmbeddingModel._has_warned = True
            return
        model_name = project_parameters["text_embedding_model"]
        model_kind = project_parameters["text_embedding_model_kind"]
        if model_kind not in ["qwen3"]:
            log_error(
                f"Unknown text embedding model kind: {model_kind}", project_parameters
            )
        if model_kind == "qwen3":
            TextEmbeddingModel._PROCESSOR = AutoTokenizer.from_pretrained(
                model_name, padding_side="left"
            )
            TextEmbeddingModel._MODEL = AutoModel.from_pretrained(
                model_name, device_map="auto", dtype=torch.bfloat16
            )
        else:
            log_error(
                f"Text embedding model kind {model_kind} not implemented.",
                project_parameters,
            )

    def embed(texts: List[str]) -> torch.Tensor:
        """
        Generate text embeddings for a list of texts.

        :param texts: List of input texts to embed
        :type texts: List[str]
        :return: Tensor containing the embeddings
        :rtype: torch.Tensor
        """
        if isinstance(texts, str):
            texts = [texts]
        if TextEmbeddingModel._MODEL is None:
            TextEmbeddingModel.start()
        if TextEmbeddingModel._MODEL is None:  # only if debug_skip_lm is True
            return torch.randn(len(texts), TextEmbeddingModel._random_embed_size)
        max_length = project_parameters["text_embedding_model_max_length"]
        inputs = TextEmbeddingModel._PROCESSOR(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(TextEmbeddingModel._MODEL.device)
        outputs = TextEmbeddingModel._MODEL(**inputs)
        model_kind = project_parameters["text_embedding_model_kind"]
        if model_kind == "qwen3":
            embeddings = outputs.last_hidden_state[:, -1]
        else:
            log_error(
                f"Text embedding model kind {model_kind} not implemented.",
                project_parameters,
            )
        return embeddings.detach().cpu()

    def compare(emeddings_a: torch.Tensor, embeddings_b: torch.Tensor) -> torch.Tensor:
        """
        Compare two sets of embeddings and return similarity scores.

        :param emeddings_a: First set of embeddings
        :type emeddings_a: torch.Tensor
        :param embeddings_b: Second set of embeddings
        :type embeddings_b: torch.Tensor
        :return: Similarity scores between the two sets of embeddings. Is of shape (len(embeddings_a), len(embeddings_b))
        :rtype: torch.Tensor
        """
        return torch.matmul(
            emeddings_a / emeddings_a.norm(dim=1, keepdim=True),
            (embeddings_b / embeddings_b.norm(dim=1, keepdim=True)).T,
        )  # Why does this work? TODO: Check this.

    def embed_compare(
        text: str, existing_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a text and an existing index of embeddings, embed the text and compare it to the existing index.

        :param text: The input text to embed and compare
        :type text: str
        :param existing_index: The existing index of embeddings to compare against
        :type existing_index: torch.Tensor
        :return: A tuple containing

            - The embedding of the input text
            - The similarity scores between the input text embedding and the existing index
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        text_embedding = TextEmbeddingModel.embed(text)
        similarity_scores = TextEmbeddingModel.compare(text_embedding, existing_index)
        return text_embedding, similarity_scores


def _verify_save_path(save_path: str, parameters: dict):
    if save_path is None:
        return
    storage_dir = parameters["storage_dir"]
    abs_save_path = os.path.abspath(save_path)
    abs_storage_dir = os.path.abspath(storage_dir)
    if not abs_save_path.startswith(abs_storage_dir):
        log_error(
            f"Database save path {abs_save_path} is not inside the storage directory {abs_storage_dir}.",
            parameters,
        )


class DictDatabase:
    def __init__(self, save_path: str = None, parameters: dict = None):
        self._parameters = load_parameters(parameters)
        """ The mode of the database. Can be 'dense' or 'keyword'. """
        self._save_path = save_path
        """ The path to save the database to. If None, the database is not saved to disk. """
        _verify_save_path(save_path, self._parameters)
        self.data: Dict[Any, str] = {}
        """ The dictionary store of the database. Values are text values. """

    def add_entry(self, key: Any, value: str):
        """
        Add a new entry to the database.

        :param key: The key for the new entry
        :type key: Any
        :param value: The text value for the new entry
        :type value: str
        """
        if key in self.data:
            self.data[key] = self.data[key] + "| \n" + value
        else:
            self.data[key] = value
        return

    def get_entry(self, key: Any) -> str:
        """
        Get an entry from the database.

        :param key: The key for the entry to retrieve
        :type key: Any
        :return: The text value for the entry
        :rtype: str
        """
        if key not in self.data:
            return ""
        return self.data[key]

    def modify_entry(self, key: Any, new_value: str):
        """
        Modify an existing entry in the database.

        :param key: The key for the entry to modify
        :type key: Any
        :param new_value: The new text value for the entry
        :type new_value: str
        """
        if key not in self.data:
            log_error(
                f"Key {key} not found in database. Cannot modify non-existent entry.",
                self._parameters,
            )
            return
        self.data[key] = new_value
        return


class DenseTextDatabase:
    _gate_prompt = """
    You have encountered a NOVEL SITUATION of [KEY_DESCRIPTION]: '[NEW_KEY]'. 
    You may have seen a similar situation before, OLD EXPERIENCE: [OLD_KEY]
    Your task is to decide whether this old experience is relevant to the novel situation. It is relevant if [MATCH_DESCRIPTION]
    Give your output in the following format:
    Reasoning: <A very brief reasoning comparing the two situations, and reasoning whether it fits the stated criteria for relevance>
    Decision: YES or NO
    [STOP]
    Now give your answer:
    Reasoning:"""

    _gate_image = np.random.randint(
        low=0, high=255, size=(40, 40)
    )  # Placeholder image, TODO: check that this doesn't mess up the VLM.

    def __init__(
        self,
        save_path: str = None,
        parameters: dict = None,
    ):
        self._parameters = load_parameters(parameters)
        """ The mode of the database. Can be 'dense' or 'keyword'. """
        self._save_path = save_path
        """ The path to save the database to. If None, the database is not saved to disk. """
        _verify_save_path(save_path, self._parameters)
        self.keys: List[str] = []
        """ The list of text keys in the database. Is of length num_entries """
        self.values: List[Any] = []
        """ The list of values in the database. Is of length num_entries. """
        self.key_embeds: torch.Tensor = None
        """ The tensor index of embeddings for the database keys. Shape: (num_entries, embed_size) """

    def add_entries(self, entries: List[Tuple[str, Any]]):
        """
        Add new text entries to the database.

        :param entries: The list of new text entries to add. Each entry is a tuple of (key_str, value)
        :type entries: List[Tuple[str, Any]]
        """
        if len(entries) == 0:
            return
        if isinstance(entries[0], str):
            entries = [entries]  # wrap single entries
        entries_keys = [entry[0] for entry in entries]
        entries_values = [entry[1] for entry in entries]
        self.keys.extend(entries_keys)
        self.values.extend(entries_values)
        new_embeddings = TextEmbeddingModel.embed(entries)
        if self.key_embeds is None:
            self.key_embeds = new_embeddings
        else:
            self.key_embeds = torch.cat(
                [self.key_embeds, new_embeddings], dim=0
            )  # TODO: Check dim

    def get_embed_top_k(
        self, text: str, k=3
    ) -> Tuple[torch.Tensor, List[Tuple[float, str, Any]]]:
        """
        Embeds the text, and gets the top k most similar entries from the index.

        :param text: The input text to embed and compare
        :type text: str
        :param k: The number of top similar entries to retrieve, defaults to 3
        :type k: int, optional
        :return: A tuple containing

            - The embedding of the input text
            - A list of tuples for the top k entries, each tuple containing:

                - The similarity score
                - The key string
                - The value
        :rtype: Tuple[torch.Tensor, List[Tuple[float, str, Any]]]

        """
        text_embedding, similarity_scores = TextEmbeddingModel.embed_compare(
            text, self.key_embeds
        )
        similarity_scores = similarity_scores[0]  # only one row.
        # get the argsort index
        top_k_indices = torch.topk(similarity_scores, k=k).indices.tolist()
        top_k_scores = similarity_scores[top_k_indices].tolist()
        # select the top k key strings
        top_k_keys = [self.keys[i] for i in top_k_indices]
        top_k_values = [self.values[i] for i in top_k_indices]
        top_k = list(
            zip(top_k_scores, top_k_keys, top_k_values)
        )  # List of tuples (score, key, value)
        return text_embedding, top_k

    def gate_relevance(
        self,
        new_key: str,
        top_k: List[Tuple[float, str, Any]],
        match_description: str,
        key_description: str,
        vlm_class=ExecutorVLM,
    ) -> List[Tuple[float, str, Any]]:
        """
        Given a new key and the top k similar entries, use the VLM to gate which entries are relevant.

        :param new_key: The new key string
        :type new_key: str
        :param top_k: The list of top k entries, each tuple containing (similarity score, key string, value)
        :type top_k: List[Tuple[float, str, Any]]
        :param match_description: The description of what constitutes a match (e.g. 'the old experience describes a similar visual interface as the new')
        :type match_description: str
        :param key_description: The description of the key (e.g. 'the visual interface of a game screen')
        :type key_description: str
        :param vlm_class: The VLM class to use for gating, defaults to ExecutorVLM
        :type vlm_class: class, optional
        :return: The filtered list of relevant entries from top_k
        :rtype: List[Tuple[float, str, Any]]
        """
        relevant_entries = []
        for score, old_key, old_value in top_k:
            prompt = (
                self._gate_prompt.replace("[NEW_KEY]", new_key)
                .replace("[OLD_KEY]", old_key)
                .replace("[MATCH_DESCRIPTION]", match_description)
                .replace("[KEY_DESCRIPTION]", key_description)
            )
            vlm_response = vlm_class.infer(
                texts=[prompt],
                images=[self._gate_image],
                max_new_tokens=200,
            )[0].lower()
            if "decision:" in vlm_response:
                decision_declaration = vlm_response.split("decision:")[1]
                if "no" in decision_declaration:
                    continue
            relevant_entries.append((score, old_key, old_value))
        return relevant_entries

    def modify_entry(self, key: str, new_key: str, new_value: Any):
        """
        Modify an existing entry in the database. Also changes the key and recomputes embeddings.

        :param key: The key for the entry to modify
        :type key: str
        :param new_key: The new key for the entry. Repeat the old key if not changing.
        :type new_key: str
        :param new_value: The new value for the entry
        :type new_value: Any
        """
        if key not in self.keys:
            log_error(
                f"Key {key} not found in database. Cannot modify non-existent entry.",
                self._parameters,
            )
            return
        index = self.keys.index(key)
        self.values[index] = new_value
        if key != new_key:
            self.keys[index] = new_key
            new_embedding = TextEmbeddingModel.embed([new_key])
            self.key_embeds[index : index + 1] = new_embedding
        return

    def remove_entry(self, key: str):
        """
        Remove an existing entry from the database.

        :param key: The key for the entry to remove
        :type key: str
        """
        if key not in self.keys:
            log_error(
                f"Key {key} not found in database. Cannot remove non-existent entry.",
                self._parameters,
            )
            return
        index = self.keys.index(key)
        self.keys.pop(index)
        self.values.pop(index)
        self.key_embeds = torch.cat(
            [self.key_embeds[:index], self.key_embeds[index + 1 :]], dim=0
        )  # TODO: Check dim
        return
