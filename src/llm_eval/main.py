import logging

from llm_eval.args import parse_args
from llm_eval.data import load_data
from llm_eval.data import pre_process
from llm_eval.evaluate import evaluate
from llm_eval.utils import get_save_path
from llm_eval.utils import get_tokenizer
from llm_eval.utils import load_model
from llm_eval.utils import load_pipeline
from torch.utils.data import DataLoader


def main() -> None:
    """Run the main function."""
    args = parse_args()

    # Check if the dataset and model are from HuggingFace Hub.
    ds_is_hf_hub = len(args.dataset_name_or_path.split('/')) == 2
    model_is_hf_hub = len(args.model_name_or_path.split('/')) == 2

    if args.use_pipeline:
        model = load_pipeline(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
        )
        tokenizer = None
    else:
        tokenizer = get_tokenizer(
            tokenizer_name_or_path=args.tokenizer_name_or_path or args.model_name_or_path,
        )
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
        token_format=args.special_token_format,
        tokenizer=tokenizer,
        max_length=args.max_length,
        num_proc=args.num_proc,
    )

    logging.info(f'Loaded {len(ds):,} examples from {args.dataset_name_or_path}.')
    logging.info(ds)

    # Create a dataloader for the dataset.
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
    )

    # Get the path to save the evaluation results.
    save_path = get_save_path(
        model_name_or_path=args.model_name_or_path,
        dataset_name_or_path=args.dataset_name_or_path,
        save_dir=args.save_path,
        model_is_hf_hub=model_is_hf_hub,
        dataset_is_hf_hub=ds_is_hf_hub,
    )

    # Evaluate the model on the dataset.
    evaluate(
        model=model,
        loader=loader,
        save_path=save_path,
        tokenizer=tokenizer,
        use_pipeline=args.use_pipeline,
        compute_reward=args.compute_reward,
    )


if __name__ == '__main__':
    main()
