# Copyright (c) 2023 Yurts AI.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
# mypy: disable-error-code="attr-defined"
import spacy
from spacy.language import Language


def t1_hallucination(doc: Language, prompts: list[str], responses: list[str]) -> list[float]:
    """Get the Type-1 hallucination reward for a batch of generated responses.

    Args:
        doc (Language): The Named Entity Recognition model.
        prompts (list[str]): The prompts used to generate the responses.
        responses (list[str]): The generated responses.

    Returns:
        list[float]: The hallucination reward for each generated response.

    """
    prompt_entities = [
        {item.text.lower() for item in doc(prompt).ents}
        for prompt in prompts
    ]

    model_entities = [
        {item.text.lower() for item in doc(response).ents}
        for response in responses
    ]
    rewards = []

    for prompt, prompt_entity, model_entity in zip(
        prompts, prompt_entities, model_entities,
    ):
        score, flag = 0, 0

        if len(model_entity) == 0:
            score = 0
            flag = 1

        for entity in model_entity:
            if entity not in prompt_entity and entity not in prompt.lower():
                score += 1
                flag = 1

        if flag == 0:
            score = 1

        rewards.append(float(score))
    return rewards


def get_ner(device: str = 'cuda') -> spacy.language.Language:
    """Get the Named Entity Recognition (NER) model.

    Args:
        device (str, optional): The device to use for the NER model.
            Defaults to "cuda".

    Returns:
        spacy.language.Language: The NER model.

    """
    device = device or 'cpu'
    if device != 'cpu':
        spacy.prefer_gpu()

    return spacy.load('en_core_web_trf')
