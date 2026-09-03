"""Shared Groq model selection for the two LLM touchpoints (rag, ingest).

The model id lived inline at four call sites across ``rag.py`` and
``ingest.py``. Groq decommissioned ``llama-3.3-70b-versatile`` while the app
was idle, which broke every one of them at once and had to be fixed in four
places. Hosted open-weight models get retired on the provider's schedule, so
the id belongs in one place — and is env-overridable, so the next retirement
can be worked around by setting ``GROQ_MODEL`` in Vercel without a code
change or a redeploy of new source.
"""

import os

# Reasoning model: it returns chain-of-thought in a separate ``reasoning``
# field and the usable output in ``content``, so it is drop-in compatible with
# the existing JSON-mode parsing. Its reasoning tokens do count against
# max_tokens, so keep the per-call budgets at or above the current values.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


def groq_model() -> str:
    """The chat model to use for RAG and ingest extraction."""
    return os.environ.get("GROQ_MODEL") or DEFAULT_GROQ_MODEL
