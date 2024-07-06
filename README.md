# Aixen: AI x Engineering

[![PyPI version](https://badge.fury.io/py/aixen.svg)](https://badge.fury.io/py/aixen)

An open-source toolkit for the rapid development of AI-powered applications.

Features:

- **Clean Pipelines**: Write AI pipelines as simple Python functions for readability and easy team collaboration.
- **Declarative**: Use type annotations and Pydantic for function signatures, enabling auto-checks and code generation.
- **Tracking**: Track and debug AI runs with statistics, performance evaluation, and API cost tracking. Gather fine-tuning samples.
- **Learnable Functions**: _(Coming soon)_ Optimize functions with hyperparameters automatically.


## Quick Start

Install using pip:
```sh
pip install aixen
```

Add API keys to the environment or [`.env` file](.env.example) in your working directory:
```
OPENAI_API_KEY=sk-xxx
```

A simple pipeline:
```python
import aixen as ai
from pydantic import BaseModel

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


class GreetingCard(BaseModel):
    text: str
    image_description: str


@ai.chat_fn
def greet(name: str) -> GreetingCard:
    """
    Creates a greeting media message for a person specified by the user.
    """


with ai.Context() as context:
    result = add(a=1, b=2)
    fact5 = factorial(5)
    card = greet(name="Lev")

    print(f"Greeting card: {card}")
    print(f"Used: ${context.usage_cost_usd}")
```

See the [examples folder](examples).


## Community and Support

Join our community to stay updated and get support:

- [GitHub Discussions](https://github.com/romovpa/aixen/discussions)

If you encounter any issues, please open an [issue on GitHub](https://github.com/romovpa/aixen/issues).


## Contributing

We welcome contributions from the community!

<!-- Here's how you can get started: TBD -->
