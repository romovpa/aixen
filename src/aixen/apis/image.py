from typing import Any, Optional

import replicate
from pydantic import AnyUrl

from aixen.context import get_context, processor

REPLICATE_DEFAULT_MODEL_NAME = "stability-ai/stable-diffusion-3"
REPLICATE_DEFAULT_SETTINGS = {
    "cfg": 3.5,
    "steps": 28,
    "aspect_ratio": "9:16",
    "output_format": "jpg",
    "output_quality": 90,
    "negative_prompt": "",
    "prompt_strength": 0.85,
}


@processor
def replicate_generate(
    prompt: str,
    model_name: str = REPLICATE_DEFAULT_MODEL_NAME,
    settings: Optional[dict[str, Any]] = None,
) -> AnyUrl:
    """
    Generate an image using the Replicate API.

    >>> replicate_generate(prompt="a cute fox in the forest")
    'https://replicate.delivery/yhqm/RFglkDNGuHYyBlygQfevtHrapiTJA7zHgzPelQIp8w5aUtLmA/R8_SD3_00001_.jpg'
    """
    context = get_context()
    client = replicate.Client(api_token=context.environment["REPLICATE_API_KEY"])
    if settings is None:
        settings = REPLICATE_DEFAULT_SETTINGS
    response = client.run(
        ref=model_name,
        input={
            "prompt": prompt,
            **settings,
        },
    )
    # TODO Usage and costs
    return response[0]
