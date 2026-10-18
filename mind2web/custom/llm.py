import logging
import re
import os
import inspect
import tiktoken

logger = logging.getLogger("main")

import openai
openai.api_key = os.environ["OPENAI_API_KEY"]
from openai import OpenAI
client = OpenAI()


def generate_response(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    stop_tokens: list[str] | None = None,
    use_tools: bool = False,
    n: int = 20
) -> tuple[list[str], dict[str, int]]:
    """Send a request to the OpenAI API."""

    logger.info(
        f"Send a request to the language model from {inspect.stack()[1].function}"
    )
    gen_kwargs = {}

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stop=stop_tokens if stop_tokens else None,
        n=n,
        **gen_kwargs
    )

    message = [*map(lambda x: x.message.content, response.choices)]

    info = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    return message, info