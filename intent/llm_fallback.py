"""
LLM fallback - only invoked when intent.rules.parse() returns UNKNOWN.

Uses a local Ollama model so the whole pipeline stays local-first (no
cloud API calls, no data leaving the machine). The LLM's ONLY job is to
emit the same structured Intent JSON the rule parser produces - it never
executes anything itself. Output is validated with pydantic before it's
allowed anywhere near the executor layer.
"""

from __future__ import annotations
import json
from loguru import logger

from intent.schema import Intent, IntentType
from config.loader import load_settings, known_app_names

_SYSTEM_PROMPT = """You convert a user's spoken command into a single JSON object \
matching this schema exactly, with no extra keys and no prose:

{{
  "type": one of {types},
  "name": string or null,
  "extension": string or null,
  "target_folder": string or null,
  "new_name": string or null,
  "app_name": string or null,
  "url_or_query": string or null,
  "confidence": float between 0 and 1
}}

Known applications: {apps}

Respond with ONLY the JSON object, nothing else."""


def _client():
    import ollama  # imported lazily so the rest of the app works without it installed
    return ollama


def classify(text: str) -> Intent:
    settings = load_settings()
    llm_cfg = settings.get("llm_fallback", {})

    if not llm_cfg.get("enabled", False):
        logger.debug("LLM fallback disabled in settings; returning UNKNOWN intent.")
        return Intent(raw_text=text, type=IntentType.UNKNOWN, confidence=0.0)

    try:
        client = _client()
    except ImportError:
        logger.warning("`ollama` package not installed - can't run LLM fallback. "
                        "`pip install ollama` and make sure the Ollama daemon is running.")
        return Intent(raw_text=text, type=IntentType.UNKNOWN, confidence=0.0)

    system_prompt = _SYSTEM_PROMPT.format(
        types=[t.value for t in IntentType],
        apps=known_app_names(),
    )

    try:
        response = client.chat(
            model=llm_cfg.get("model", "qwen2.5:1.5b"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            format="json",
        )
        raw = response["message"]["content"]
        data = json.loads(raw)
        data["raw_text"] = text
        intent = Intent(**data)
        logger.info(f"LLM fallback classified '{text}' as {intent.type} (confidence={intent.confidence})")
        return intent

    except ConnectionError as e:
        logger.debug(f"LLM fallback: Ollama not reachable ({e}). Run `ollama serve` to enable.")
        return Intent(raw_text=text, type=IntentType.UNKNOWN, confidence=0.0)
    except Exception as e:
        # Silently absorb model-not-found / API errors - log to file only, not stderr
        err_str = str(e)
        if "not found" in err_str or "404" in err_str:
            logger.debug(f"LLM fallback: model not available ({e}). Pull it with `ollama pull {llm_cfg.get('model', 'qwen2.5:1.5b')}`")
        else:
            logger.debug(f"LLM fallback failed for {text!r}: {e}")
        return Intent(raw_text=text, type=IntentType.UNKNOWN, confidence=0.0)