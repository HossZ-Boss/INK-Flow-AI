"""Persistent learning index fed by staff-reviewed request outcomes."""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from .artist_config import ArtistConfigManager
from .decision_schemas import DecisionHistoryExample
from .vector_store import VectorStoreManager

LOGGER = logging.getLogger(__name__)

DEFAULT_LEARNING_INDEX_PATH = "data/learning.faiss"


class LearningConflictError(ValueError):
    """An example ID is already stored with different content."""


class LearningStore:
    """Own the FAISS learning index, its file on disk, and safe writes."""

    def __init__(
        self,
        vector_store: VectorStoreManager,
        index_path: str | None = None,
        artist_config: ArtistConfigManager | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._index_path = index_path
        self._artist_config = artist_config or ArtistConfigManager()
        self._write_lock = threading.Lock()

    @classmethod
    def from_environment(cls) -> LearningStore:
        """Load the index from LEARNING_INDEX_PATH, or start an empty one."""
        index_path = os.getenv(
            "LEARNING_INDEX_PATH",
            DEFAULT_LEARNING_INDEX_PATH,
        ).strip() or DEFAULT_LEARNING_INDEX_PATH
        vector_store = VectorStoreManager()
        if Path(index_path).is_file():
            try:
                vector_store.load_index(index_path)
            except Exception:
                LOGGER.exception(
                    "Learning index at %s is unreadable; moving it aside and "
                    "starting a new index.",
                    index_path,
                )
                _move_aside(index_path)
                vector_store = VectorStoreManager()
        LOGGER.info(
            "Learning layer active with %s staff-reviewed records.",
            vector_store.record_count,
        )
        return cls(vector_store=vector_store, index_path=index_path)

    @property
    def vector_store(self) -> VectorStoreManager:
        """Return the shared vector store used for retrieval."""
        return self._vector_store

    @property
    def record_count(self) -> int:
        """Return how many staff-reviewed examples are indexed."""
        return self._vector_store.record_count

    @property
    def approved_price_count(self) -> int:
        """Return how many indexed examples carry an approved price."""
        return sum(
            1
            for record in self._vector_store.records
            if record.approved_price_range is not None
        )

    def artist_key(self, name_or_key: str) -> str | None:
        """Map an artist display name or key to its configured key."""
        wanted = name_or_key.strip().casefold()
        if not wanted:
            return None
        for artist in self._artist_config.config.artists:
            if wanted in {
                artist.artist_key.casefold(),
                artist.display_name.casefold(),
            }:
                return artist.artist_key
        return None

    def add_example(self, payload: dict[str, Any]) -> DecisionHistoryExample:
        """Validate, index, and persist one staff-reviewed example.

        ``payload`` holds DecisionHistoryExample fields except
        ``example_id``, plus an optional ``inquiry_id`` used to build a
        stable identifier for idempotent retries.
        """
        data = dict(payload)
        inquiry_id = data.pop("inquiry_id", None)
        example = DecisionHistoryExample.model_validate(
            {**data, "example_id": _example_id(inquiry_id, data)}
        )
        with self._write_lock:
            for record in self._vector_store.records:
                if record.example_id == example.example_id:
                    if record != example:
                        raise LearningConflictError(
                            "This inquiry was already stored with different "
                            f"details: {example.example_id}."
                        )
                    return record
            self._vector_store.add_records([example])
            if self._index_path:
                self._vector_store.save_index(self._index_path)
        return example


def _example_id(inquiry_id: str | None, data: dict[str, Any]) -> str:
    """Build a stable example ID from the request ID or its content."""
    if inquiry_id:
        return f"feedback-{inquiry_id}"[:80]
    digest = sha256(
        json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:24]
    return f"feedback-{digest}"


def _move_aside(index_path: str) -> None:
    """Keep an unreadable index for inspection instead of overwriting it."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for path in (Path(index_path), Path(f"{index_path}.records.json")):
        if path.exists():
            path.rename(path.with_name(f"{path.name}.broken-{stamp}"))
