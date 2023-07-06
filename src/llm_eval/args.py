import argparse
import logging
from dataclasses import dataclass
from dataclasses import field

from rich.logging import RichHandler
from transformers import HfArgumentParser


@dataclass
class LLMArguments:
    """Arguments for hallucination evaluation with LLM."""
    model_name_or_path: str = field(
        metadata={'help': 'Path to pretrained model or model identifier from HuggingFace Hub'},
    )

    dataset_name_or_path: str = field(
        default='res/data/tech-crunch.jsonl',
        metadata={'help': 'The HF Hub name or path of the dataset to evaluate on.'},
    )

    tokenizer_name_or_path: str | None = field(
        default=None,
        metadata={'help': 'The name or path of the tokenizer to use.'},
    )

    save_path: str = field(
        default='res/eval/',
        metadata={'help': 'The path to save the evaluation results.'},
    )

    data_max_size: int | None = field(
        default=None,
        metadata={'help': 'The maximum number of examples to load from the dataset.'},
    )

    prompt_format: str | None = field(
        default=None,
        metadata={'help': 'The format of the prompt to use for the LLM.'},
    )

    max_legnth: int = field(
        default=1024,
        metadata={'help': 'The maximum length of the prompt.'},
    )

    use_pipeline: bool = field(
        default=False,
        metadata={'help': 'Whether to use the pipeline API for generation.'},
    )

    log_level: str = field(
        default='INFO',
        metadata={
            'help': 'The logging level.',
            'choices': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        },
    )


def _parse_log_level(log_level: str) -> int:
    """Parse the logging level."""
    log_level = log_level.upper()
    # Set logging level.
    LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }

    if log_level not in LEVELS:
        raise ValueError(f'Invalid logging level: {log_level}')

    return LEVELS[log_level]


def parse_args() -> LLMArguments:
    """Parse command-line arguments for hallucination evaluation with LLM.

    Returns:
        LLMArguments: The parsed command-line arguments.

    """
    kwargs = {
        'description': 'Evaluate a pretrained LLM model on a dataset.',
        'formatter_class': lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=200),
    }

    # Parse command-line arguments.
    parser = HfArgumentParser(LLMArguments, **kwargs)
    args = parser.parse_args_into_dataclasses()[0]

    # Set logging level.
    logging.basicConfig(
        level=_parse_log_level(args.log_level),
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            RichHandler(),
        ],
    )

    return args
