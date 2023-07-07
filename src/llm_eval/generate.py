# Copyright (c) 2023 Yurts AI.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from typing import Any

import torch
from transformers import AutoTokenizer
from transformers import Pipeline
from transformers import PreTrainedModel


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
