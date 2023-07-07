# Copyright (c) 2023 Yurts AI.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import os
from typing import Any

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import Pipeline
from transformers import pipeline


def get_tokenizer(
    tokenizer_name_or_path: str,
    padding_side: str | None = None,
) -> AutoTokenizer:
    """Get the tokenizer for a pretrained LLM model.

    Args:
        tokenizer_name_or_path (str, optional): The name or path of the tokenizer
            to use.

    Returns:
        AutoTokenizer: The tokenizer for the pretrained LLM model.

    """
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if padding_side is not None:
        tokenizer.padding_side = padding_side

    return tokenizer


def load_pipeline(
    model_name_or_path: str,
    tokenizer: AutoTokenizer | None = None,
) -> Pipeline:
    """Load a pipeline for a pretrained LLM model.

    Args:
        model_name_or_path (str): The name or path of the pretrained LLM model.
        tokenizer (AutoTokenizer, optional): The tokenizer to use.
            If None, defaults to model's tokenizer.
            Defaults to None.

    """

    pipe = pipeline(
        'text-generation',
        model=model_name_or_path,
        tokenizer=tokenizer,
        device_map='auto',
        trust_remote_code=True,
    )

    return pipe


def load_model(
    model_name_or_path: str,
) -> AutoModelForCausalLM:
    """Load a pretrained LLM model.

    Args:
        model_name_or_path (str): The name or path of the pretrained LLM model.

    Returns:
        AutoModelForCausalLM: The pretrained LLM model.

    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map='auto',
        trust_remote_code=True,
    )

    return model


def get_save_path(
    model_name_or_path: str,
    dataset_name_or_path: str,
    save_dir: str,
    model_is_hf_hub: bool = True,
    dataset_is_hf_hub: bool = False,
) -> str:
    """Get the path to save the evaluation results.

    Args:
        model_name_or_path (str): The name or path of the pretrained LLM model.
        dataset_name_or_path (str): The name or path of the dataset to evaluate on.
        save_dir (str): The base directory to save the evaluation results.

    Returns:
        str: The file path to save the evaluation results.

    """
    if model_is_hf_hub:
        model_name_or_path = model_name_or_path.split('/')[-1]
    else:
        # FIXME(victor-iyi): Handle this case properly.
        model_name_or_path = os.path.splitext(model_name_or_path)[0]

    if dataset_is_hf_hub:
        dataset_name_or_path = dataset_name_or_path.split('/')[-1]
    else:
        dataset_name_or_path = os.path.splitext(dataset_name_or_path)[0]

    model_name = model_name_or_path.split('/')[-1]
    dataset_name = dataset_name_or_path.split('/')[-1]
    save_path = os.path.join(save_dir, f'{model_name}_{dataset_name}.jsonl')

    # Create the directory to save the evaluation results.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    return save_path


def collator(data: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Collator for the dataset."""
    return {key: [d[key] for d in data] for key in data[0]}
