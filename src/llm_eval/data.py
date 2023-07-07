# Copyright (c) 2023 Yurts AI.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
# mypy: disable-error-code="union-attr"
import os
from itertools import islice
from typing import Any

import jsonlines
from datasets import Dataset
from transformers import AutoTokenizer


# Default prompt to use for the LLM.
DEFAULT_PROMPT: str = 'Facts: {}\nTell me more about {} using ONLY the facts above.'


def load_data(
    name_or_path: str,
    limit: int | None = None,
    is_hf_hub: bool = False,
) -> Dataset:
    """Load data from the data directory or HuggingFace Hub name/path.

    Args:
        name_or_path (str): The name or path of the dataset to load.
        limit (int, optional): The maximum number of examples to load from the dataset.
            Defaults to None.
        is_hf_hub (bool, optional): Whether the dataset is from the HuggingFace Hub.
            Defaults to False.

    Returns:
        Dataset: The loaded dataset.

    """
    if not os.path.exists(name_or_path) and not is_hf_hub:
        raise FileNotFoundError(
            f'Dataset {name_or_path} not found. '
            'Please make sure you have the dataset downloaded or '
            'pass a valid dataset name from the HuggingFace Hub.',
        )

    # TODO(victor-iyi): Implement loading from HuggingFace Hub
    if is_hf_hub:
        # Loading from HuggingFace Hub is not yet implemented.
        return NotImplemented

    # Load data from the data directory.
    with jsonlines.open(name_or_path) as reader:
        data = [line for line in islice(reader, limit)]

    context, question = [], []
    for line in data:
        context.append(line['context'])
        question.append(line['question'])

    ds = Dataset.from_dict({
        'context': context,
        'question': question,
    })
    return ds


def pre_process(
    ds: Dataset,
    prompt_format: str | None = None,
    token_format: str | None = None,
    tokenizer: AutoTokenizer | None = None,
    max_length: int = 1024,
    num_proc: int | None = None,
    **tokenizer_kwargs: Any,
) -> Dataset:
    """Pre-process a dataset for hallucination evaluation with LLM.

    Args:
        ds (Dataset): The dataset to add the prompt to.
        prompt_format (str, optional): The prompt to add to the context.
            Defaults to ``DEFAULT_PROMPT``.
        token_format (str, optional): The format of the prompt to use for the LLM.
            Defaults to None.
        tokenizer (AutoTokenizer, optional): The tokenizer to use to tokenize the
            prompt.
            Defaults to None.
        max_length (int, optional): The maximum length of the prompt.
            Defaults to 1024.
        num_proc (int, optional): The number of processes to use for multiprocessing.
            Defaults to None.

    Keyword Args:
        tokenizer_kwargs (Any): Additional keyword arguments to pass to the tokenizer.

    Returns:
        Dataset: The dataset with the prompt added to the context.

    """
    if prompt_format is None:
        prompt_format = DEFAULT_PROMPT

    def _map_fn(example: dict[str, Any]) -> dict[str, Any]:
        # Add the natural language prompt to the context.
        example['prompt'] = prompt_format.format(example['context'], example['question'])

        # Add special tokens to the NL prompt.
        if token_format is not None:
            example['prompt'] = token_format.format(example['prompt'])
        else:
            example['prompt'] = f'{example["prompt"]}\nAnswer: '

        # Tokenize the prompt.
        if tokenizer is not None:
            example['input_ids'] = tokenizer(
                example['prompt'],
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                **tokenizer_kwargs,
            ).input_ids

        return example
    ds = ds.map(
        _map_fn,
        num_proc=num_proc,
    )
    return ds


if __name__ == '__main__':
    ds = load_data('res/data/tech-crunch.jsonl')
    print(ds)

    ds = pre_process(ds, token_format='<|prompter|>{}<|endoftext|><|assistant|>')
    print(ds)
    print(ds[0])
