"""FAISS-backed storage for verified studio decision history."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import faiss
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from .config import get_settings
from .decision_schemas import DecisionHistoryExample

LOGGER = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSION = 1536
_METADATA_SUFFIX = ".records.json"


class _IndexMetadata(BaseModel):
    """Serializable metadata required to restore vector-to-record mappings."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    embedding_model: str = Field(min_length=1)
    embedding_dimension: int = Field(gt=0)
    records: list[DecisionHistoryExample] = Field(default_factory=list)


def decision_history_to_text(record: DecisionHistoryExample) -> str:
    """Render the searchable features of one verified decision as plain text."""
    styles = ", ".join(record.style_tags) or "unknown"
    placement = record.placement or "unknown"
    size = record.size_estimate_cm or "unknown"
    color = record.color_preference or "unknown"
    return (
        f"Tattoo styles: {styles}. Placement: {placement}. "
        f"Size in centimeters: {size}. Color preference: {color}."
    )


class VectorStoreManager:
    """Manage normalized decision embeddings in an in-memory FAISS index.

    Embeddings are L2-normalized before insertion. ``IndexFlatIP`` therefore
    produces cosine-similarity scores while retaining exact nearest-neighbor
    search. The embedding model and index are injectable for deterministic
    tests and alternative runtime composition.
    """

    def __init__(
        self,
        embedding_model: Embeddings | None = None,
        index: faiss.Index | None = None,
        records: Sequence[DecisionHistoryExample] | None = None,
        embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        """Initialize an empty or caller-supplied cosine-similarity index.

        Args:
            embedding_model: LangChain embedding implementation. When omitted,
                an OpenAI client is created from the validated app settings.
            index: Optional FAISS inner-product index for dependency injection.
            records: Records already aligned with a populated injected index.
            embedding_dimension: Expected vector dimension.
            embedding_model_name: Model identifier stored with persisted data.

        Raises:
            ValueError: If dimensions, metric type, or record counts disagree.
        """
        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be greater than zero.")
        if not embedding_model_name.strip():
            raise ValueError("embedding_model_name must not be empty.")

        self._embedding_dimension = embedding_dimension
        self._embedding_model_name = embedding_model_name.strip()
        self._embedding_model = (
            embedding_model
            if embedding_model is not None
            else self._create_embeddings()
        )
        self._index = (
            index if index is not None else faiss.IndexFlatIP(embedding_dimension)
        )
        self._records = list(records) if records is not None else []
        self._validate_index_state()

    @property
    def index(self) -> faiss.Index:
        """Return the managed FAISS index for diagnostics and composition."""
        return self._index

    @property
    def records(self) -> tuple[DecisionHistoryExample, ...]:
        """Return an immutable snapshot of indexed decision records."""
        return tuple(self._records)

    @property
    def record_count(self) -> int:
        """Return the number of unique verified records in the index."""
        return len(self._records)

    def add_records(self, records: list[DecisionHistoryExample]) -> None:
        """Embed and add verified decision examples to the FAISS index.

        Args:
            records: Validated history examples to embed and retain in index
                order. An empty list is accepted as a no-op.

        Raises:
            ValueError: If the embedding response has an invalid shape, count,
                dimension, contains a zero-length vector, or reuses an example
                identifier for different content.
            RuntimeError: If the embedding provider cannot process records.
        """
        new_records = self._select_new_records(records)
        if not new_records:
            return

        texts = [decision_history_to_text(record) for record in new_records]
        try:
            raw_embeddings = self._embedding_model.embed_documents(texts)
        except Exception as exc:
            raise RuntimeError(
                "Unable to generate embeddings for learning records."
            ) from exc
        embeddings = self._prepare_embeddings(
            raw_embeddings,
            len(new_records),
        )
        previous_count = len(self._records)
        self._records.extend(new_records)
        try:
            self._index.add(embeddings)
        except Exception:
            del self._records[previous_count:]
            raise

    def search_similar_cases(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> list[tuple[float, DecisionHistoryExample]]:
        """Return the most similar verified decisions for a text query.

        Similarity is cosine similarity because document and query vectors are
        normalized before exact inner-product search. Higher scores indicate
        stronger similarity.

        Args:
            query_text: Searchable description of the current tattoo request.
            top_k: Maximum number of nearest records to return.

        Returns:
            Ordered ``(similarity, record)`` pairs, highest similarity first.
            An empty index returns an empty list without calling the embedding
            provider. Embedding-provider and FAISS failures also return an
            empty list so routing can use its deterministic fallback.

        Raises:
            ValueError: If the query is blank or ``top_k`` is not positive.
            RuntimeError: If FAISS returns an unmapped record identifier.
        """
        normalized_query = query_text.strip()
        if not normalized_query:
            raise ValueError("query_text must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if self._index.ntotal == 0:
            return []

        try:
            raw_embedding = self._embedding_model.embed_query(
                normalized_query
            )
            query_embedding = self._prepare_embeddings([raw_embedding], 1)
        except Exception as exc:  # pragma: no cover - provider-dependent
            LOGGER.warning("Vector query embedding failed: %s", exc)
            return []
        result_count = min(top_k, int(self._index.ntotal))
        try:
            scores, record_ids = self._index.search(
                query_embedding,
                result_count,
            )
        except Exception as exc:  # pragma: no cover - FAISS defensive branch
            LOGGER.warning("FAISS similarity search failed: %s", exc)
            return []
        return self._map_search_results(scores[0], record_ids[0])

    def _select_new_records(
        self,
        records: Sequence[DecisionHistoryExample],
    ) -> list[DecisionHistoryExample]:
        """Return unique additions and reject conflicting identifier reuse.

        Exact repeats are treated as idempotent retries. Reusing an existing
        identifier for different content is rejected before any embedding or
        index mutation occurs.
        """
        known_records = {
            record.example_id: record for record in self._records
        }
        selected: list[DecisionHistoryExample] = []
        for record in records:
            existing = known_records.get(record.example_id)
            if existing is not None:
                if existing != record:
                    raise ValueError(
                        "Decision example ID is already mapped to different "
                        f"content: {record.example_id}."
                    )
                continue
            known_records[record.example_id] = record
            selected.append(record)
        return selected

    def save_index(self, path: str) -> None:
        """Persist the FAISS index and its record mapping to disk.

        The FAISS binary is written to ``path``. A Pydantic-validated JSON
        sidecar is written beside it because FAISS does not store application
        metadata or Python objects.

        Args:
            path: Destination path for the FAISS binary index.
        """
        index_path = self._resolve_index_path(path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = _IndexMetadata(
            embedding_model=self._embedding_model_name,
            embedding_dimension=self._embedding_dimension,
            records=self._records,
        )
        faiss.write_index(self._index, str(index_path))
        self._metadata_path(index_path).write_text(
            metadata.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def load_index(self, path: str) -> None:
        """Load a FAISS index and its validated record mapping from disk.

        Args:
            path: Source path for the FAISS binary index.

        Raises:
            FileNotFoundError: If the index or record sidecar does not exist.
            ValueError: If persisted vectors are incompatible with this
                manager's model, dimension, metric, or record mapping.
        """
        index_path = self._resolve_index_path(path)
        metadata_path = self._metadata_path(index_path)
        if not index_path.is_file():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"FAISS record metadata not found: {metadata_path}"
            )

        metadata = _IndexMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
        self._validate_loaded_metadata(metadata)
        loaded_index = faiss.read_index(str(index_path))
        self._validate_candidate_index(loaded_index, len(metadata.records))
        self._index = loaded_index
        self._records = list(metadata.records)

    def _create_embeddings(self) -> OpenAIEmbeddings:
        """Create the default OpenAI embedding client from app settings."""
        settings = get_settings()
        return OpenAIEmbeddings(
            api_key=settings.api_key.get_secret_value(),
            model=self._embedding_model_name,
            dimensions=self._embedding_dimension,
            max_retries=settings.max_retries,
            timeout=settings.timeout_seconds,
        )

    def _prepare_embeddings(
        self,
        raw_embeddings: list[list[float]],
        expected_count: int,
    ) -> NDArray[np.float32]:
        """Validate and normalize document vectors for cosine similarity."""
        embeddings = np.asarray(raw_embeddings, dtype=np.float32)
        expected_shape = (expected_count, self._embedding_dimension)
        if embeddings.shape != expected_shape:
            raise ValueError(
                "Embedding response shape does not match the expected "
                f"shape {expected_shape}; received {embeddings.shape}."
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("Embedding response contains non-finite values.")

        norms = np.linalg.norm(embeddings, axis=1)
        if np.any(norms == 0):
            raise ValueError("Embedding response contains a zero-length vector.")
        faiss.normalize_L2(embeddings)
        return embeddings

    def _map_search_results(
        self,
        scores: NDArray[np.float32],
        record_ids: NDArray[np.int64],
    ) -> list[tuple[float, DecisionHistoryExample]]:
        """Map FAISS result identifiers to their stored Pydantic records."""
        results: list[tuple[float, DecisionHistoryExample]] = []
        for raw_score, raw_record_id in zip(scores, record_ids, strict=True):
            record_id = int(raw_record_id)
            if record_id < 0:
                continue
            if record_id >= len(self._records):
                raise RuntimeError(
                    "FAISS returned an identifier without a mapped record."
                )
            results.append((float(raw_score), self._records[record_id]))
        return results

    def _validate_index_state(self) -> None:
        """Validate the injected index and its optional record mapping."""
        self._validate_unique_record_ids(self._records)
        self._validate_candidate_index(self._index, len(self._records))

    def _validate_unique_record_ids(
        self,
        records: Sequence[DecisionHistoryExample],
    ) -> None:
        """Reject record collections containing duplicate identifiers."""
        example_ids = [record.example_id for record in records]
        if len(example_ids) != len(set(example_ids)):
            raise ValueError(
                "Decision history records must use unique example IDs."
            )

    def _validate_candidate_index(
        self,
        index: faiss.Index,
        record_count: int,
    ) -> None:
        """Ensure an index is compatible with normalized inner-product data."""
        if index.d != self._embedding_dimension:
            raise ValueError(
                "FAISS index dimension does not match embedding_dimension."
            )
        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("FAISS index must use the inner-product metric.")
        if index.ntotal != record_count:
            raise ValueError(
                "FAISS vector count must equal the mapped record count."
            )

    def _validate_loaded_metadata(self, metadata: _IndexMetadata) -> None:
        """Reject metadata created for another vector space."""
        self._validate_unique_record_ids(metadata.records)
        if metadata.embedding_model != self._embedding_model_name:
            raise ValueError(
                "Persisted embedding model does not match this manager."
            )
        if metadata.embedding_dimension != self._embedding_dimension:
            raise ValueError(
                "Persisted embedding dimension does not match this manager."
            )

    def _resolve_index_path(self, path: str) -> Path:
        """Return a validated filesystem path for index persistence."""
        if not path.strip():
            raise ValueError("Index path must not be empty.")
        return Path(path).expanduser().resolve()

    def _metadata_path(self, index_path: Path) -> Path:
        """Return the record-sidecar path for a FAISS binary path."""
        return index_path.with_name(f"{index_path.name}{_METADATA_SUFFIX}")
