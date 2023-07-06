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
) -> Dataset:
    """Load data from the data directory or HuggingFace Hub name/path.

    Args:
        name_or_path (str): The name or path of the dataset to load.

    Returns:
        Dataset: The loaded dataset.

    """
    is_hf_hub = False

    if not os.path.exists(name_or_path):
        msg = (
            f'Dataset {name_or_path} not found. '
            'Please make sure you have the dataset downloaded or '
            'pass a valid dataset name from the HuggingFace Hub.'
        )
        if len(name_or_path.split('/')) == 2:
            is_hf_hub = True
            if not is_hf_hub:
                raise FileNotFoundError(msg)
        else:
            raise FileNotFoundError(msg)

    # TODO(victor-iyi): Implement loading from HuggingFace Hub
    if is_hf_hub:
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
    prompt: str = DEFAULT_PROMPT,
    token_format: str | None = None,
    tokenizer: AutoTokenizer | None = None,
    max_length: int = 1024,
    **tokenizer_kwargs: Any,
) -> Dataset:
    """Pre-process a dataset for hallucination evaluation with LLM.

    Args:
        ds (Dataset): The dataset to add the prompt to.
        prompt (str, optional): The prompt to add to the context.
            Defaults to ``DEFAULT_PROMPT``.
        token_format (str, optional): The format of the prompt to use for the LLM.
            Defaults to None.
        tokenizer (AutoTokenizer, optional): The tokenizer to use to tokenize the
            prompt.
            Defaults to None.
        max_length (int, optional): The maximum length of the prompt.
            Defaults to 1024.

    Keyword Args:
        tokenizer_kwargs (Any): Additional keyword arguments to pass to the tokenizer.

    Returns:
        Dataset: The dataset with the prompt added to the context.

    """
    def _add_prompt(example: dict[str, Any]) -> dict[str, Any]:
        # Add the natural language prompt to the context.
        example['prompt'] = prompt.format(example['context'], example['question'])

        # Add special tokens to the NL prompt.
        if token_format is not None:
            example['prompt'] = token_format.format(example['prompt'])
        else:
            example['prompt'] = f'{example["prompt"]}\nAnswer: '

        # Tokenize the prompt.
        if tokenizer is not None:
            example['input_ids'] = tokenizer(
                example['prompt'],
                return_tensors='pt',
                max_length=max_length,
                padding='max_length',
                truncation=True,
                **tokenizer_kwargs,
            )['input_ids']

        return example
    ds = ds.map(
        _add_prompt,
        # remove_columns=['context', 'question'],
    )
    return ds


if __name__ == '__main__':
    ds = load_data('res/data/tech-crunch.jsonl')
    print(ds)

    ds = pre_process(ds, token_format='<|prompter|>{}<|endoftext|><|assistant|>')
    print(ds)
    print(ds[0])
