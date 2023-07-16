# Copyright (c) 2023 Yurts AI.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import logging
import os
from typing import Any

import jsonlines
import torch
from llm_eval.generate import generate_with_model
from llm_eval.generate import generate_with_pipeline
from llm_eval.reward import get_ner
from llm_eval.reward import t1_hallucination
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import Pipeline
from transformers import PreTrainedModel


def evaluate(
    model: PreTrainedModel | Pipeline,
    loader: DataLoader[dict[str, str]],
    save_path: str,
    tokenizer: AutoTokenizer | None = None,
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
        tokenizer (AutoTokenizer | None): The tokenizer to use for the model.
            Defaults to None.
        use_pipeline (bool): Whether to use the model as a pipeline.
            Defaults to False.
        compute_reward (bool): Whether to compute hallucination reward.
            Defaults to False.

    Keyword Args:
        gen_kwargs (Any): Additional keyword arguments to pass to the model's

    """
    if os.path.isfile(save_path):
        logging.warning(f'File {save_path} already exists. Overwriting...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ner_model = get_ner(device=device)

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
                input_ids = torch.tensor(batch['input_ids'], dtype=torch.int64, device=device)
                input_ids = input_ids.view(-1, input_ids.shape[-1])
                responses = generate_with_model(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    **gen_kwargs,
                )

            # Get hallucination reward for each generated response (optional).
            if compute_reward:
                scores = t1_hallucination(ner_model, batch['prompt'], responses)
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
