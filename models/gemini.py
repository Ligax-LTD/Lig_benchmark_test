"""Google Gemini vision classifier."""

import json
import os
import re
import time

from google import genai
from google.genai import types
from dotenv import load_dotenv

from ._prompt import build_prompt

load_dotenv()

_MODEL = os.environ.get("GOOGLE_MODEL", "gemini-2.5-flash")


def _parse(text: str) -> dict:
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    return json.loads(m.group()) if m else {}


def classify(img_bytes: bytes, taxonomy: dict) -> dict:
    t0     = time.time()
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=_MODEL,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    build_prompt(taxonomy),
                ],
                config=types.GenerateContentConfig(
                    max_output_tokens=300,
                    temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            break
        except Exception as exc:
            if "429" not in str(exc) and "RESOURCE_EXHAUSTED" not in str(exc):
                raise
            if attempt == 2:
                raise
            time.sleep(65)

    data = _parse(resp.text or "")
    return {
        "key":        [v for v in data.get("key",  []) if v in taxonomy["key"]],
        "base":       [v for v in data.get("base", []) if v in taxonomy["base"]],
        "key_probs":  None,
        "base_probs": None,
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }
