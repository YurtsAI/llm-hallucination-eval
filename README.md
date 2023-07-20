<!--
 Copyright (c) 2023 Yurts AI.

 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->

# LLM Hallucination Evaluation

[YurtsAI][yurtsai] developed a pipeline to evaluate the famous hallucination
problem of large language models.
Refer to [Reward engineering coupled with RLHF to reduce LLM hallucinations][blog]
to learn more about hallucinations and the evaluation pipeline.

![Evaluation Pipeline][eval-pipeline]

## :wrench: Setup

**Requirements:**
- Python 3.10+
- [Poetry][`poetry`]
- :hugs:[HuggingFace token][hf-token]

Firstly, create a virtual environment and activate it:

```sh
python3.10 -m virtualenv .venv
source .venv/bin/activate
```

To install the required dependencies, assuming you have [`poetry`] installed, run:
> It also logs you into :hugs:Hub which prompts for your :hugs:Hub token.

```sh
make install
```

or in dev mode:

```sh
make install-dev
```

## :gear: Evaluation

To evaluate the model on the given TechCrunch dataset, run:

```sh
python -m llm_eval \
    --model_name_or_path tiiuae/falcon-7b-instruct \
    --max_length 512 \
    --data_max_size 100 \
    --num_proc 4 \
    --batch_size 8 \
    --compute_reward
```

> For more information, run `llm_eval --help` or `python -m llm_eval --help`.

Some models have different input formatting, i.e addition of special token or
formatted in certain ways. To handle this, you can use the `--input_format`
flag. For example, to preprocess the input for the
`OpenAssistant/falcon-7b-sft-mix-2000` model, run:

```sh
python -m llm_eval \
    --model_name_or_path OpenAssistant/falcon-7b-sft-mix-2000 \
    --data_max_size 100 \
    --input_format "<|prompter|>{}<|endoftext|><|assistant|>" \
    --batch_size 8 \
    --shuffle \
    --max_length 512 \
    --compute_reward
```

## :bar_chart: Visualize

If you'd like further data exploration, you can use `pandas` or your
favorite data analysis library to visualize the data.

> If you're not familiar with `pandas`, you can use the following snippet.
> Make sure to `pip install pandas` first.

```python
>>> import pandas as pd

>>> # Load the data to a pandas dataframe.
>>> df = pd.read_json('res/eval/tech-crunch_falcon-7b-instruct.jsonl', lines=True)

>>> # Filter Type-1 hallucinations.
>>> good = df[df.reward == 1]
>>> neutral = df[df.reward == 0]
>>> bad = df[df.reward < 0]

>>> # Get the number of good, neutral, and bad responses.
>>> n, n_good, n_neutral, n_bad = len(df), len(good), len(neutral), len(bad)

>>> print(f'Good: {n_good} ({n_good / n:.2%})')
>>> print(f'Neutral: {n_neutral} ({n_neutral / n:.2%})')
>>> print(f'Bad: {n_bad} ({n_bad / n:.2%})')
```

You're welcome to submit a [pull request] with your visualizations!

## :technologist: Contribution

You are very welcome to modify and use them in your own projects.

Please keep a link to the [original repository]. If you have made a fork with
substantial modifications that you feel may be useful, then please [open a new
issue on GitHub][issues] with a link and short description.

## :balance_scale: License (MIT)

This project is opened under the [MIT][license] which allows very
broad use for both private and commercial purposes.

A few of the images used for demonstration purposes may be under copyright.
These images are included under the "fair usage" laws.

[`poetry`]: https://python-poetry.org/docs/
[tech-crunch]: https://techcrunch.com/
[yurtsai]: https://yurts.ai/
[blog]: https://yurts.ai/blog/
[hf-token]: https://huggingface.co/docs/hub/security-tokens/
[eval-pipeline]: ./res/images/eval-pipeline.png
[pull request]: https://github.com/YurtsAI/llm-hallucination-eval/pulls
[original repository]: https://github.com/YurtsAI/llm-hallucination-eval
[issues]: https://github.com/YurtsAI/llm-hallucination-eval/issues
[license]: ./LICENSE
