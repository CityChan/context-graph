"""Short-lived judge requests with deterministic HTTP client cleanup."""

import asyncio
import os


async def call_openai_raw(messages, model="gpt-4o-mini", max_retries=3, *, raise_errors=False):
    from openai import AsyncOpenAI

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    async with AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")) as client:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(model=model, messages=messages)
                return response.choices[0].message.content or ""
            except Exception as exc:
                if attempt == max_retries - 1:
                    if raise_errors:
                        raise
                    return f"Error: {exc}"
                await asyncio.sleep(attempt + 1)
    return ""
