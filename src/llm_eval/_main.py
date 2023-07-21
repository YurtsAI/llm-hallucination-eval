# Copyright (c) 2023 Yurts AI.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import logging

from llm_eval.args import LLMArguments
from llm_eval.data import load_data
from llm_eval.data import pre_process
from llm_eval.evaluate import evaluate
from llm_eval.utils import collator
from llm_eval.utils import get_save_path
from llm_eval.utils import get_tokenizer
from llm_eval.utils import load_model
from llm_eval.utils import load_pipeline
from torch.utils.data import DataLoader


def _evaluate(args: LLMArguments) -> None:
    """Run the evaluation for an LLM on a given dataset.

    Args:
        args: The arguments for the evaluation.

    """

    # Check if the dataset and model are from HuggingFace Hub.
    ds_is_hf_hub = len(args.dataset_name_or_path.split('/')) == 2
    model_is_hf_hub = len(args.model_name_or_path.split('/')) == 2

    # Load the model and tokenizer.
    tokenizer = get_tokenizer(
        tokenizer_name_or_path=args.tokenizer_name_or_path or args.model_name_or_path,
    )
    if args.use_pipeline:
        model = load_pipeline(
            model_name_or_path=args.model_name_or_path,
            tokenizer=tokenizer,
        )
    else:
        model = load_model(
            model_name_or_path=args.model_name_or_path,
        )

    # Load the dataset.
    ds = load_data(
        name_or_path=args.dataset_name_or_path,
        limit=args.data_max_size,
        is_hf_hub=ds_is_hf_hub,
    )
    ds = pre_process(
        ds=ds,
        prompt_format=args.prompt_format,
        input_format=args.input_format,
        tokenizer=tokenizer,
        max_length=args.max_length,
        num_proc=args.num_proc,
    )

    logging.info(f'Loaded {len(ds):,} examples from {args.dataset_name_or_path}.')
    logging.info(ds)

    # Create a dataloader for the dataset.
    loader = DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        collate_fn=collator,
    )

    # Get the path to save the evaluation results.
    save_path = get_save_path(
        model_name_or_path=args.model_name_or_path,
        dataset_name_or_path=args.dataset_name_or_path,
        save_dir=args.save_path,
        model_is_hf_hub=model_is_hf_hub,
        dataset_is_hf_hub=ds_is_hf_hub,
    )

    # Set the keyword arguments for the generation.
    gen_kwargs = {
        'top_k': 50,
        'top_p': 0.9,
        'do_sample': True,
        'temperature': 0.2,
        'repetition_penalty': 1.2,
        'min_new_tokens': args.output_min_length,
        'max_new_tokens': args.output_max_length,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }

    # Evaluate the model on the dataset.
    evaluate(
        model=model,
        loader=loader,
        save_path=save_path,
        tokenizer=tokenizer,
        use_pipeline=args.use_pipeline,
        compute_reward=args.compute_reward,
        **gen_kwargs,
    )
