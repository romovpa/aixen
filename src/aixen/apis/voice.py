import base64

import httpx
from loguru import logger
from pydantic import BaseModel

from aixen.context import File, Usage, get_context, processor, save

ELEVENLABS_DEFAULT_MODEL = "eleven_turbo_v2"
ELEVENLABS_DEFAULT_VOICE = "pNInz6obpgDQGcFmaJgB"
ELEVENLABS_DEFAULT_VOICE_SETTINGS = {
    "stability": 0.5,
    "similarity_boost": 0.75,
}


class VoiceAlignment(BaseModel):
    characters: list[str]
    character_start_times_seconds: list[float]
    character_end_times_seconds: list[float]


class GeneratedVoice(BaseModel):
    audio_file: File
    alignment: VoiceAlignment


class ElevenlabsUsage(Usage):
    characters: int


@processor
def elevenlabs_generate(
    text: str,
    model_id: str | None = None,
    voice_id: str | None = None,
    voice_settings: dict[str, float] | None = None,
) -> GeneratedVoice:
    """
    Synthesise speech using the ElevenLabs API.

    Settings:
    ```js
    {
        "elevenlabs.model": "eleven_turbo_v2",
        "elevenlabs.voice": "pNInz6obpgDQGcFmaJgB",
        "elevenlabs.voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
        },
    }
    ```

    >>> elevenlabs_generate("Hello, World!")
    GeneratedVoice(audio_file='.../000_elevenlabs_generate.mp3', alignment=...)
    """
    context = get_context()

    model_id = model_id or context.settings.get(
        "elevenlabs.model", ELEVENLABS_DEFAULT_MODEL
    )
    voice_id = voice_id or context.settings.get(
        "elevenlabs.voice", ELEVENLABS_DEFAULT_VOICE
    )
    voice_settings = voice_settings or context.settings.get(
        "elevenlabs.voice_settings", ELEVENLABS_DEFAULT_VOICE_SETTINGS
    )

    response = httpx.post(
        url=f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps",
        json={
            "text": text,
            "model_id": model_id,
            "voice_settings": voice_settings,
        },
        headers={
            "Content-Type": "application/json",
            "xi-api-key": context.environment["ELEVENLABS_API_KEY"],
        },
    )
    if not response.is_success:
        logger.error(
            f"Error encountered, status: {response.status_code}, "
            f"content: {response.text}"
        )
        response.raise_for_status()
    response_data = response.json()

    context.record(
        usage=ElevenlabsUsage(
            characters=len(text),
            cost_usd=len(text) / 1000 * 0.22,
        )
    )

    audio_bytes = base64.b64decode(response_data["audio_base64"])
    audio_file = save(audio_bytes, ext="mp3")

    return GeneratedVoice(
        audio_file=audio_file,
        alignment=VoiceAlignment(**response_data["alignment"]),
    )
