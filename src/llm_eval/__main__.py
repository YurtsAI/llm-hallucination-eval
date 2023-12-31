# Copyright (c) 2023 Yurts AI.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from llm_eval._main import _evaluate
from llm_eval.args import parse_args


def main() -> int:
    """Run the main function."""
    args = parse_args()
    _evaluate(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
