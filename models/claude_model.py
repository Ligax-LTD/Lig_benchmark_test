"""Anthropic Claude vision classifier."""

import base64
import json
import os
import re
import time

import anthropic
from dotenv import load_dotenv

from ._prompt import build_prompt

load_dotenv()

_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")


def _parse(text: str) -> dict:
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    return json.loads(m.group()) if m else {}


def classify(img_bytes: bytes, taxonomy: dict) -> dict:
    t0     = time.time()
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    b64    = base64.b64encode(img_bytes).decode()

    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=_MODEL,
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                        },
                        {"type": "text", "text": build_prompt(taxonomy)},
                    ],
                }],
            )
            break
        except anthropic.RateLimitError:
            if attempt == 2:
                raise
            time.sleep(65)

    data = _parse(resp.content[0].text or "")
    return {
        "key":        [v for v in data.get("key",  []) if v in taxonomy["key"]],
        "base":       [v for v in data.get("base", []) if v in taxonomy["base"]],
        "key_probs":  None,
        "base_probs": None,
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }
