# ruff: noqa: T201

import aixen as ai


@ai.fn
def add(a: int, b: int) -> int:
    """
    Adds two numbers
    """
    return a + b


@ai.chat_fn
def factorial(n: int) -> int:
    """
    Calculates the factorial of a number
    """


@ai.chat_fn
def series_formula(seq: list[int]) -> str:
    """
    You are given a sequence of integers $x_i$, and you need to find the formula that
    generates this sequence.

    Provide the result as a LaTeX expression.
    ONLY output the LaTeX expression, no text or additional formatting.
    <|user|>
    {% for x in seq %}{{ x }}, {% endfor %}...
    """


if __name__ == "__main__":
    with ai.Context() as context:
        print(f"2 + 3 = {add(2, 3)}")
        print(f"Factorial of 5 = {factorial(5)}")

        seq = [1, 3, 5, 7]
        print(f"Series formula: {series_formula(seq)}")

        print(f"Cost: ${context.usage_cost_usd}")
