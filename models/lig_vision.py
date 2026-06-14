"""LIG Vision API client — black-box image classification via REST endpoint."""

import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

_URL = os.environ.get("LIG_VISION_URL", "").rstrip("/")

# per-class key thresholds: mean of 5-fold CV optima (each fold calibrated on k-1 folds,
# evaluated on the held-out fold — no image appears in its own calibration set)
_KEY_THRESHOLDS = {
    "alicia_keys":      0.25,
    "boho":             0.29,
    "box_braids":       0.32,
    "butterfly_locs":   0.47,
    "faux_locs":        0.38,
    "flat_twist":       0.25,
    "fulani_braids":    0.29,
    "invisible_locs":   0.45,
    "island_twist":     0.45,
    "lemonade_braids":  0.45,
    "marley_twists":    0.14,
    "microlocs":        0.20,
    "mini_twists":      0.49,
    "passion_twists":   0.59,
    "ponytails":        0.03,
    "puff":             0.08,
    "soft_locs":        0.10,
    "space_buns":       0.38,
}

# per-class base thresholds: mean of 5-fold CV optima
_BASE_THRESHOLDS = {
    "braids":   0.83,
    "cornrows": 0.59,
    "crochet":  0.65,
    "locs":     0.60,
    "twist":    0.35,
}

# strategy 3: base-family → key-family mapping for hierarchical confidence boosting
_BASE_KEY_FAMILIES = {
    "locs":   {"faux_locs", "butterfly_locs", "soft_locs", "invisible_locs", "microlocs"},
    "twist":  {"marley_twists", "passion_twists", "mini_twists", "island_twist", "flat_twist"},
    "braids": {"box_braids", "fulani_braids", "lemonade_braids", "boho"},
}
_BOOST_BASE_MIN  = 0.80   # base prob above which family key thresholds are reduced
_BOOST_FACTOR    = 0.65   # multiply key threshold by this factor when base is confident
_BOOST_FLOOR     = 0.20   # never boost a threshold below this value


def _select_key(probs: dict | None, taxonomy_classes: list, base_probs: dict | None) -> list:
    if not probs:
        return []

    # start from per-class optimal thresholds
    thresholds = {c: _KEY_THRESHOLDS.get(c, 0.15) for c in taxonomy_classes}

    # strategy 3: relax thresholds for key classes whose base family is confident
    if base_probs:
        for base_cls, key_family in _BASE_KEY_FAMILIES.items():
            if base_probs.get(base_cls, 0.0) >= _BOOST_BASE_MIN:
                for kc in key_family:
                    if kc in thresholds and thresholds[kc] > _BOOST_FLOOR:
                        thresholds[kc] = max(thresholds[kc] * _BOOST_FACTOR, _BOOST_FLOOR)

    selected = [c for c in taxonomy_classes if c in probs and probs[c] >= thresholds[c]]

    return selected


def _select_base(probs: dict | None) -> list:
    if not probs:
        return []
    return [c for c, p in probs.items() if p >= _BASE_THRESHOLDS.get(c, 0.20)]


def classify(img_bytes: bytes, taxonomy: dict) -> dict:
    if not _URL:
        raise EnvironmentError(
            "LIG_VISION_URL is not set. Lig Vision results are pre-cached in "
            "results/lig_vision.json. Use the Ligax AI playground to verify individual predictions."
        )
    t0 = time.time()

    resp = requests.post(
        f"{_URL}/api/playground/classify/onnx",
        files={"file": ("image.jpg", img_bytes, "image/jpeg")},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    latency_ms = (time.time() - t0) * 1000

    key_probs  = None
    base_probs = None
    if "taxonomy" in data:
        key_probs  = {c: v["prob"] for c, v in data["taxonomy"].get("key",  {}).items()}
        base_probs = {c: v["prob"] for c, v in data["taxonomy"].get("base", {}).items()}

    # base computed first so it can inform key selection (strategy 3)
    base_pred = _select_base(base_probs)
    key_pred  = _select_key(key_probs, taxonomy["key"], base_probs)

    return {
        "key":        key_pred,
        "base":       base_pred,
        "key_probs":  key_probs,
        "base_probs": base_probs,
        "latency_ms": round(latency_ms, 1),
    }
