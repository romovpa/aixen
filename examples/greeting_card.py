# ruff: noqa: T201

import webbrowser

import aixen as ai
import httpx
import pygame
from aixen.apis.image import replicate_generate
from aixen.apis.voice import elevenlabs_generate
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, HttpUrl


class GreetingCard(BaseModel):
    text: str = Field(
        ...,
        description="A greeting message that will be spoken by a voice synthesis model",
    )
    image_description: str = Field(
        ...,
        description=(
            "Description of an image that will be displayed on a greeting card. "
            "This description will be used by an image synthesis model."
        ),
    )


@ai.fn
def get_name_from_github(url: HttpUrl) -> str | None:
    """
    Extracts the user's name from a GitHub profile page.
    """
    response = httpx.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    name_tag = soup.find("span", class_="p-name vcard-fullname d-block overflow-hidden")
    return name_tag.text.strip() if name_tag else None


@ai.chat_fn
def greet(name: str) -> GreetingCard:
    """
    Creates a greeting media message for a person specified by the user.
    """


@ai.chat_fn
def describe_image(image_url: HttpUrl) -> str:
    """
    Describe the image

    <|user|>
    {{ image_url | image(max_dim=512) }}
    """


if __name__ == "__main__":
    url = "https://github.com/simonw"

    with ai.Context() as context:
        name = get_name_from_github(url)

        card = greet(name)

        voice = elevenlabs_generate(card.text)
        image_url = replicate_generate(card.image_description)
        image_descr = describe_image(image_url)

        print("Greeting:", card.text)
        print("Image description:", card.image_description)
        print("Audio:", voice.audio_file.local_path)
        print("Image:", image_url)
        print("Image description:", image_descr)

        print(f"Cost: ${context.usage_cost_usd}")

        webbrowser.open_new(image_url)

        pygame.mixer.init()
        pygame.mixer.music.load(voice.audio_file.local_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
