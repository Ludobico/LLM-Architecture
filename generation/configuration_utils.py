import copy
import json
import os
import warnings
from dataclasses import dataclass, is_dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from ..Utils.hub import PushToHubMixin
from ..Utils.import_utils import is_torch_available

NEEDS_CACHE_CONFIG = {}



class WatermarkingConfig:
    def __init__(
        self,
        greenlist_ratio: Optional[float] = 0.25,
        bias: Optional[float] = 2.0,
        hashing_key: Optional[int] = 15485863,
        seeding_scheme: Optional[str] = "lefthash",
        context_width: Optional[int] = 1,
    ):
        self.greenlist_ratio = greenlist_ratio
        self.bias = bias
        self.hashing_key = hashing_key
        self.seeding_scheme = seeding_scheme
        self.context_width = context_width
    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        Constructs a WatermarkingConfig instance from a dictionary of parameters.

        Args:
            config_dict (Dict[str, Any]): Dictionary containing configuration parameters.
            **kwargs: Additional keyword arguments to override dictionary values.

        Returns:
            WatermarkingConfig: Instance of WatermarkingConfig constructed from the dictionary.
        """
        config = cls(**config_dict)
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)
        return config

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (Union[str, os.PathLike]): Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            config_dict = self.to_dict()
            json_string = json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

            writer.write(json_string)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            Dict[str, Any]: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        return output

    def __iter__(self):
        for attr, value in copy.deepcopy(self.__dict__).items():
            yield attr, value

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_json_string(self):
        """
        Serializes this instance to a JSON formatted string.

        Returns:
            str: JSON formatted string representing the configuration instance.
        """
        return json.dumps(self.__dict__, indent=2) + "\n"

    def update(self, **kwargs):
        """
        Update the configuration attributes with new values.

        Args:
            **kwargs: Keyword arguments representing configuration attributes and their new values.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def validate(self):
        watermark_missing_arg_msg = (
            "Some of the keys in `watermarking_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )
        if self.seeding_scheme not in ["selfhash", "lefthash"]:
            raise ValueError(
                watermark_missing_arg_msg.format(
                    key="seeding_scheme",
                    correct_value="[`selfhash`, `lefthash`]",
                    found_value=self.seeding_scheme,
                ),
            )
        if not 0.0 <= self.greenlist_ratio <= 1.0:
            raise ValueError(
                watermark_missing_arg_msg.format(
                    key="greenlist_ratio",
                    correct_value="in range between 0.0 and 1.0",
                    found_value=self.seeding_scheme,
                ),
            )
        if not self.context_width >= 1:
            raise ValueError(
                watermark_missing_arg_msg.format(
                    key="context_width",
                    correct_value="a positive integer",
                    found_value=self.context_width,
                ),
            )


class GenerationConfig(PushToHubMixin):
    def __init__(self, **kwargs):
            # Parameters that control the length of the output
            self.max_length = kwargs.pop("max_length", 20)
            self.max_new_tokens = kwargs.pop("max_new_tokens", None)
            self.min_length = kwargs.pop("min_length", 0)
            self.min_new_tokens = kwargs.pop("min_new_tokens", None)
            self.early_stopping = kwargs.pop("early_stopping", False)
            self.max_time = kwargs.pop("max_time", None)
            self.stop_strings = kwargs.pop("stop_strings", None)

            # Parameters that control the generation strategy used
            self.do_sample = kwargs.pop("do_sample", False)
            self.num_beams = kwargs.pop("num_beams", 1)
            self.num_beam_groups = kwargs.pop("num_beam_groups", 1)
            self.penalty_alpha = kwargs.pop("penalty_alpha", None)
            self.use_cache = kwargs.pop("use_cache", True)

            # Parameters for manipulation of the model output logits
            self.temperature = kwargs.pop("temperature", 1.0)
            self.top_k = kwargs.pop("top_k", 50)
            self.top_p = kwargs.pop("top_p", 1.0)
            self.min_p = kwargs.pop("min_p", None)
            self.typical_p = kwargs.pop("typical_p", 1.0)
            self.epsilon_cutoff = kwargs.pop("epsilon_cutoff", 0.0)
            self.eta_cutoff = kwargs.pop("eta_cutoff", 0.0)
            self.diversity_penalty = kwargs.pop("diversity_penalty", 0.0)
            self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
            self.encoder_repetition_penalty = kwargs.pop("encoder_repetition_penalty", 1.0)
            self.length_penalty = kwargs.pop("length_penalty", 1.0)
            self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
            self.bad_words_ids = kwargs.pop("bad_words_ids", None)
            self.force_words_ids = kwargs.pop("force_words_ids", None)
            self.renormalize_logits = kwargs.pop("renormalize_logits", False)
            self.constraints = kwargs.pop("constraints", None)
            self.forced_bos_token_id = kwargs.pop("forced_bos_token_id", None)
            self.forced_eos_token_id = kwargs.pop("forced_eos_token_id", None)
            self.remove_invalid_values = kwargs.pop("remove_invalid_values", False)
            self.exponential_decay_length_penalty = kwargs.pop("exponential_decay_length_penalty", None)
            self.suppress_tokens = kwargs.pop("suppress_tokens", None)
            self.begin_suppress_tokens = kwargs.pop("begin_suppress_tokens", None)
            self.forced_decoder_ids = kwargs.pop("forced_decoder_ids", None)
            self.sequence_bias = kwargs.pop("sequence_bias", None)
            self.token_healing = kwargs.pop("token_healing", False)
            self.guidance_scale = kwargs.pop("guidance_scale", None)
            self.low_memory = kwargs.pop("low_memory", None)
            watermarking_config = kwargs.pop("watermarking_config", None)
            if watermarking_config is None:
                self.watermarking_config = None
            elif isinstance(watermarking_config, WatermarkingConfig):
                self.watermarking_config = watermarking_config
            else:
                self.watermarking_config = WatermarkingConfig.from_dict(watermarking_config)

            # Parameters that define the output variables of `generate`
            self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
            self.output_attentions = kwargs.pop("output_attentions", False)
            self.output_hidden_states = kwargs.pop("output_hidden_states", False)
            self.output_scores = kwargs.pop("output_scores", False)
            self.output_logits = kwargs.pop("output_logits", None)
            self.return_dict_in_generate = kwargs.pop("return_dict_in_generate", False)

            # Special tokens that can be used at generation time
            self.pad_token_id = kwargs.pop("pad_token_id", None)
            self.bos_token_id = kwargs.pop("bos_token_id", None)
            self.eos_token_id = kwargs.pop("eos_token_id", None)

            # Generation parameters exclusive to encoder-decoder models
            self.encoder_no_repeat_ngram_size = kwargs.pop("encoder_no_repeat_ngram_size", 0)
            self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

            # Assistant generation
            self.num_assistant_tokens = kwargs.pop("num_assistant_tokens", 5)
            self.num_assistant_tokens_schedule = kwargs.pop("num_assistant_tokens_schedule", "heuristic")

            # DoLa generation
            self.dola_layers = kwargs.pop("dola_layers", None)

            # Cache implementation
            self.cache_implementation = kwargs.pop("cache_implementation", None)
            self.cache_config = kwargs.pop("cache_config", None)
            if self.cache_implementation is not None and self.cache_implementation in NEEDS_CACHE_CONFIG:
                cache_config_class = NEEDS_CACHE_CONFIG[self.cache_implementation]
                if self.cache_config is None:
                    self.cache_config = cache_config_class()
                elif isinstance(self.cache_config, dict):
                    self.cache_config = cache_config_class.from_dict(self.cache_config)
            self.return_legacy_cache = kwargs.pop("return_legacy_cache", True)

            # Prompt lookup decoding
            self.prompt_lookup_num_tokens = kwargs.pop("prompt_lookup_num_tokens", None)
            self.max_matching_ngram_size = kwargs.pop("max_matching_ngram_size", None)

            # Wild card
            self.generation_kwargs = kwargs.pop("generation_kwargs", {})

            # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
            # interface.
            self._from_model_config = kwargs.pop("_from_model_config", False)
            self._commit_hash = kwargs.pop("_commit_hash", None)
            self.transformers_version = kwargs.pop("transformers_version", __version__)

            # Additional attributes without default values
            if not self._from_model_config:
                # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
                # model's default configuration file
                for key, value in kwargs.items():
                    try:
                        setattr(self, key, value)
                    except AttributeError as err:
                        logger.error(f"Can't set {key} with value {value} for {self}")
                        raise err

            # Validate the values of the attributes
            self.validate(is_init=True)