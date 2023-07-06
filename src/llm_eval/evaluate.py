import logging
import os
from typing import Any

import jsonlines
import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import Pipeline
from transformers import PreTrainedModel


def evaluate(
    model: PreTrainedModel | Pipeline,
    loader: DataLoader[dict[str, str]],
    tokenizer: AutoTokenizer,
    save_path: str,
    use_pipeline: bool = False,
    compute_reward: bool = False,
    **gen_kwargs: Any,
) -> None:
    """Evaluate a pretrained LLM model on a dataset.

    Args:
        model (PreTrainedModel | Pipeline): The pretrained LLM model
            or pipeline object.
        loader (DataLoader): The dataset to evaluate on.
        save_path (str): The path to save the evaluation results.

    Keyword Args:
        gen_kwargs (Any): Additional keyword arguments to pass to the model's

    """
    if os.path.isfile(save_path):
        logging.warning(f'File {save_path} already exists. Overwriting...')

    # Generate hallucinations for each example in the dataset.
    with jsonlines.open(save_path, 'w') as writer:
        for batch in tqdm(loader, desc='Evaluating'):
            # Generate model response for the batch.
            if use_pipeline:
                responses = generate_with_pipeline(
                    pipeline=model,
                    texts=batch['prompt'],
                    **gen_kwargs,
                )
            else:
                responses = generate_with_model(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=batch['input_ids'],
                    **gen_kwargs,
                )

            # Get hallucination reward for each generated response (optional).
            if compute_reward:
                scores = get_rewards(batch['prompt'], responses)
                for i, (response, score) in enumerate(zip(responses, scores)):
                    writer.write({
                        'context': batch['context'][i],
                        'question': batch['question'][i],
                        'prompt': batch['prompt'][i],
                        'response': response,
                        'reward': score,
                    })
            else:
                # Write the generated hallucinations to the output file.
                for i, response in enumerate(responses):
                    writer.write({
                        'context': batch['context'][i],
                        'question': batch['question'][i],
                        'prompt': batch['prompt'][i],
                        'response': response,
                    })

    logging.info(f'Evaluation results saved to {save_path}.')


def generate_with_model(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    **gen_kwargs: Any,
) -> list[str]:
    """Generate hallucinations for a batch of examples.

    Args:
        model (PreTrainedModel): The pretrained LLM model.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        input_ids (torch.Tensor): The input IDs for the model.

    Keyword Args:
        gen_kwargs (Any): Additional keyword arguments to pass to the model's
            ``generate`` method.

    Returns:
        list[str]: The generated model response.

    """
    # Generate model response for the batch.
    encode = model.generate(
        input_ids=input_ids,
        output_scores=True,
        return_dict_in_generate=True,
        **gen_kwargs,
    )
    seq_len = len(encode['scores'])
    tokens = encode['sequences'][:, -seq_len:]
    outputs = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    return outputs


def generate_with_pipeline(
    pipeline: Pipeline,
    texts: list[str],
    return_full_text: bool = False,
    **gen_kwargs: Any,
) -> list[str]:
    """Generate text using a pipeline.

    Args:
        pipeline (Pipeline): A pipeline object.
        texts (list[str]): The text to generate.
        return_full_text (bool, optional): Whether to return all text or just the
            generated text.
            Defaults to False.

    Keyword Args:
        **gen_kwargs: Keyword arguments for the ``model.generate()`` method.

    Returns:
        list[str]: Generated texts.

    """
    outputs = pipeline(
        texts,
        return_full_text=return_full_text,
        num_return_sequences=1,
        **gen_kwargs,
    )
    outputs = [output[0]['generated_text'] for output in outputs]
    return outputs


def get_rewards(prompts: list[str], responses: list[str]) -> list[float]:
    """Get the hallucination reward for a batch of generated responses.

    Args:
        prompts (list[str]): The prompts used to generate the responses.
        responses (list[str]): The generated responses.

    Returns:
        list[float]: The hallucination reward for each generated response.

    """
    return [0.0 for _ in range(len(prompts))]
