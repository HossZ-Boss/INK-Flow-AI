"""FastAPI dependency providers."""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from fastapi import HTTPException, status

from ai_brain.errors import ConfigurationError
from ai_brain.learning import LearningStore
from ai_brain.processor import StudioAIBrain

LOGGER = logging.getLogger(__name__)

_DISABLED_VALUES = {"0", "false", "no", "off"}


@lru_cache(maxsize=1)
def get_learning_store() -> LearningStore | None:
    """Load the persistent learning index once, or None when disabled."""
    if os.getenv("LEARNING_ENABLED", "true").strip().lower() in _DISABLED_VALUES:
        return None
    try:
        return LearningStore.from_environment()
    except ConfigurationError:
        raise
    except Exception:
        LOGGER.exception("Learning layer unavailable; continuing without it.")
        return None


@lru_cache(maxsize=1)
def get_ai_brain() -> StudioAIBrain:
    """Create and cache the AI Brain for reuse across requests."""
    try:
        store = get_learning_store()
        return StudioAIBrain(
            vector_store=store.vector_store if store is not None else None
        )
    except ConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service is not configured.",
        ) from exc
