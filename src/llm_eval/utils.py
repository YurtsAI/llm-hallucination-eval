import os

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import Pipeline
from transformers import pipeline


def get_tokenizer(
    tokenizer_name_or_path: str,
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
    return tokenizer


def load_pipeline(
    model_name_or_path: str,
    tokenizer_name_or_path: str | None = None,
) -> Pipeline:
    """Load a pipeline for a pretrained LLM model.

    Args:
        model_name_or_path (str): The name or path of the pretrained LLM model.
        tokenizer_name_or_path (str, optional): The name or path of the tokenizer
            to use. If None, defaults to ``model_name_or_path``.
            Defaults to None.

    """
    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = model_name_or_path

    pipe = pipeline(
        'text-generation',
        model=model_name_or_path,
        tokenizer=tokenizer_name_or_path,
        device_map='auto',
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
) -> str:
    """Get the path to save the evaluation results.

    Args:
        model_name_or_path (str): The name or path of the pretrained LLM model.
        dataset_name_or_path (str): The name or path of the dataset to evaluate on.
        save_dir (str): The base directory to save the evaluation results.

    Returns:
        str: The file path to save the evaluation results.

    """
    model_name = model_name_or_path.split('/')[-1]
    dataset_name = dataset_name_or_path.split('/')[-1]
    save_path = os.path.join(save_dir, f'{model_name}_{dataset_name}.jsonl')
    return save_path
