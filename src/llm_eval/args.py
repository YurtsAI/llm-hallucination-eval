from dataclasses import dataclass
from dataclasses import field


@dataclass
class LLMArguments:
    """Arguments for hallucination evaluation with LLM."""
    model_name_or_path: str = field(
        metadata={'help': 'Path to pretrained model or model identifier from HuggingFace Hub'},
    )

    dataset_name_or_path: str = field(
        metadata={'help': 'The name or path of the dataset to evaluate on.'},
    )

    data_max_size: int | None = field(
        default=None,
        metadata={'help': 'The maximum number of examples to load from the dataset.'},
    )

    prompt_format: str | None = field(
        default=None,
        metadata={'help': 'The format of the prompt to use for the LLM.'},
    )
