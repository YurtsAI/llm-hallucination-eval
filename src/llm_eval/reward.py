# Copyright (c) 2023 Yurts AI.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from spacy.language import Language


def t1_hallucination(model: Language, prompts: list[str], responses: list[str]) -> list[float]:
    """Get the Type-1 hallucination reward for a batch of generated responses.

    The algorithm is described in the blog post <https://yurts.ai/blogs/rlhf-hallucination>.
    The Type-1 hallucination reward is defined as the number of entities in the generated
    response that are not present in the prompt.

    How it works:
        1. For each prompt, extract the entities using the NER model.
        2. For each generated response, extract the entities using the NER model.
        3. For each generated response, calculate the number of entities that are not present
              in the prompt.
        4. Return the list of numbers.  The lower the figure, the more the model hallucinates.

    Args:
        model (Language): The Named Entity Recognition model.
        prompts (list[str]): The prompts used to generate the responses.
        responses (list[str]): The generated responses.

    Returns:
        list[float]: The hallucination reward for each generated response.

    """
    component_cfg = {
        'fastcoref': {'resolve_text': True},
    }
    # Get the entities for each prompt.
    prompt_entities = [
        {item.text.lower() for item in model(prompt, component_cfg=component_cfg).ents}
        for prompt in prompts
    ]

    # Get the entities for each generated response.
    model_entities = [
        {item.text.lower() for item in model(response, component_cfg=component_cfg).ents}
        for response in responses
    ]

    # Calculate the hallucination reward for each generated response.
    rewards = []

    for prompt, prompt_entity, model_entity in zip(
        prompts, prompt_entities, model_entities,
    ):
        score, is_hallucinating = 0, False

        # Generated response has no entities.
        if len(model_entity) == 0:
            is_hallucinating = True  # Neutral

        # Check if the entities in the generated response are present in the prompt.
        for entity in model_entity:
            if entity not in prompt_entity and entity not in prompt.lower():
                score += -1
                is_hallucinating = True

        # No hallucination found.
        if not is_hallucinating:
            score = 1

        rewards.append(float(score))
    return rewards
