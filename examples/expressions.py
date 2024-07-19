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


if __name__ == "__main__":
    with ai.Context() as context:
        print(f"2 + 3 = {add(2, 3)}")
        print(f"Factorial of 5 = {factorial(5)}")

        print(f"Cost: ${context.usage_cost_usd}")
