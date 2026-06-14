"""OpenAI GPT vision classifier — Responses API."""

import base64
import json
import os
import re
import time

import openai
from dotenv import load_dotenv

from ._prompt import build_prompt

load_dotenv()

_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")


def _parse(text: str) -> dict:
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    return json.loads(m.group()) if m else {}


def classify(img_bytes: bytes, taxonomy: dict) -> dict:
    t0     = time.time()
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=60.0)
    b64    = base64.b64encode(img_bytes).decode()

    for attempt in range(3):
        try:
            resp = client.responses.create(
                model=_MODEL,
                reasoning={"effort": "low"},
                input=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{b64}",
                        },
                        {
                            "type": "input_text",
                            "text": build_prompt(taxonomy),
                        },
                    ],
                }],
                max_output_tokens=300,
            )
            break
        except openai.RateLimitError:
            if attempt == 2:
                raise
            time.sleep(65)

    data = _parse(resp.output_text or "")
    return {
        "key":        [v for v in data.get("key",  []) if v in taxonomy["key"]],
        "base":       [v for v in data.get("base", []) if v in taxonomy["base"]],
        "key_probs":  None,
        "base_probs": None,
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }
