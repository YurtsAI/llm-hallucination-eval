# Copyright (c) 2023 Yurts AI.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

def t1_hallucination(prompts: list[str], responses: list[str]) -> list[float]:
    """Get the Type-1 hallucination reward for a batch of generated responses.

    Args:
        prompts (list[str]): The prompts used to generate the responses.
        responses (list[str]): The generated responses.

    Returns:
        list[float]: The hallucination reward for each generated response.

    """
    return [0.0 for _ in range(len(prompts))]
