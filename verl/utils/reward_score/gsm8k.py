# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from decimal import Decimal, InvalidOperation

_SOLUTION_CLIP_CHARS = 300
_NUMBER_PATTERN = r"-?(?:\d[\d,]*(?:\.\d*)?|\.\d+)"
_STRICT_ANSWER_PATTERN = re.compile(
    rf"####\s*(?:\$\s*)?(?:\\boxed\s*\{{\s*({_NUMBER_PATTERN})\s*\}}|({_NUMBER_PATTERN}))(?:\s*\$)?"
)


def _normalize_number(value):
    """Normalize equivalent GSM8K numeric spellings before comparison."""
    value = str(value).strip().replace(",", "")
    try:
        number = Decimal(value)
    except InvalidOperation:
        return value

    if not number.is_finite():
        return value
    if number == 0:
        return "0"
    return format(number.normalize(), "f")


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # Keep the required ``####`` marker while accepting the common GSM8K
        # renderings ``#### 12``, ``#### \\boxed{12}``, and
        # ``#### $\\boxed{12}$``.
        solutions = [match.group(1) or match.group(2) for match in _STRICT_ANSWER_PATTERN.finditer(solution_str)]
        if not solutions:
            final_answer = None
        else:
            # take the last solution
            final_answer = _normalize_number(solutions[-1])
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if answer == _normalize_number(ground_truth):
            return score
        else:
            return format_score
