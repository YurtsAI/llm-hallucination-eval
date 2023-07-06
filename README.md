# LLM Hallucination Evaluation

[YurtsAI][yurtsai] developed a pipeline to evaluate the famous hallucination
problem of large language models.

## :wrench: Setup

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

## :bar_chart: Evaluation

To evaluate the model on the given TechCrunch dataset, run:

```sh
python -m llm_eval.main \
    --model_name_or_path tiiuae/falcon-7b-instruct \
    --max_length 512 \
    --data_max_size 100 \
    --num_proc 4 \
    --batch_size 8
```

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

[`poetry`]: https://python-poetry.org/
[tech-crunch]: https://techcrunch.com/
[yurtsai]: https://yurts.ai/

[original repository]: https://github.com/YurtsAI/llm-hallucination-eval
[issues]: https://github.com/YurtsAI/llm-hallucination-eval/issues
[license]: ./LICENSE
