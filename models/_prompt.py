"""Shared classification prompt for all frontier models."""

import json


def build_prompt(taxonomy: dict) -> str:
    key_list  = ", ".join(taxonomy["key"])
    base_list = ", ".join(taxonomy["base"])
    return f"""Classify the hairstyle in this image.

Return ONLY valid JSON, no other text:
{{"key": ["<value>", ...], "base": ["<value>", ...]}}

KEY — choose all that apply from this exact list (or empty array if none match):
{key_list}

BASE — choose all that apply from this exact list (or empty array if none match):
{base_list}

Only use values from the lists above. Do not invent new values."""
