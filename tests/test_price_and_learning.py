"""Tests for staff-only price suggestions and the learning feedback layer."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_openai import ChatOpenAI

from ai_brain.decision_schemas import DecisionHistoryExample, MoneyRange
from ai_brain.learning import LearningStore
from ai_brain.price_suggestion import TattooPriceSuggester
from ai_brain.processor import StudioAIBrain
from ai_brain.schemas import (
    AIExtractionOutput,
    PriceSuggestion,
    TattooExtractionDraft,
    TattooInquiryInput,
)
from ai_brain.summary import HighRiskSummaryBuilder
from ai_brain.vector_store import VectorStoreManager
from api.main import create_app
from api.routes import _optional_learning_store
from tests.test_processor import (
    StubRouter,
    StubTextExtractor,
    StubVisionAnalyzer,
)

_PRICE_JSON = json.dumps(
    {
        "minimum": 347,
        "maximum": 512,
        "estimated_hours": "3-4 hours",
        "complexity": "medium",
        "confidence_level": "medium",
        "reasoning": "Medium detailed blackwork on the forearm.",
    }
)


class RecordingLLM:
    """Return a fixed response and keep the messages it received."""

    def __init__(self, content: str = _PRICE_JSON) -> None:
        self._content = content
        self.calls: list[Any] = []

    def invoke(self, messages: Any) -> SimpleNamespace:
        self.calls.append(messages)
        return SimpleNamespace(content=self._content)


class FailingLLM:
    """Simulate an unavailable model provider."""

    def invoke(self, messages: Any) -> SimpleNamespace:
        raise RuntimeError("External LLM calls are disabled in tests.")


class FakeVectorStore:
    """Return configured matches without embeddings."""

    def __init__(self, records: list[DecisionHistoryExample]) -> None:
        self.records = tuple(records)

    def search_similar_cases(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> list[tuple[float, DecisionHistoryExample]]:
        return [(0.9, record) for record in self.records][:top_k]


def _analysis(risk_level: str = "high") -> AIExtractionOutput:
    return AIExtractionOutput(
        tattoo_idea="Wolf head with forest",
        style_tags=["blackwork"],
        placement="forearm",
        size_estimate_cm="15 cm",
        color_preference="black-and-grey",
        suggested_artist="Sandra",
        confidence_level="high",
        ai_reasoning="Matches blackwork specialty.",
        missing_information=[],
        risk_level=risk_level,
        draft_reply="Thanks, the studio will review this.",
    )


def _approved_example(
    example_id: str = "feedback-1",
    minimum: int = 400,
    maximum: int = 550,
) -> DecisionHistoryExample:
    return DecisionHistoryExample(
        example_id=example_id,
        channel="whatsapp",
        style_tags=["blackwork"],
        placement="forearm",
        size_estimate_cm="14 cm",
        color_preference="black-and-grey",
        final_artist_key="sandra",
        original_ai_action="artist_review",
        final_action="accept_request",
        ai_suggestion_outcome="correct",
        approved_price_range=MoneyRange(
            currency="EUR",
            minimum=Decimal(minimum),
            maximum=Decimal(maximum),
        ),
    )


def _suggester(
    llm: Any,
    records: list[DecisionHistoryExample] | None = None,
    images: list[str] | None = None,
) -> TattooPriceSuggester:
    return TattooPriceSuggester(
        llm=cast(ChatOpenAI, llm),
        vector_store=(
            cast(VectorStoreManager, FakeVectorStore(records))
            if records is not None
            else None
        ),
        image_loader=(lambda urls: list(images or [])),
    )


def test_market_estimate_is_rounded_and_uses_reference_images() -> None:
    llm = RecordingLLM()
    suggestion = _suggester(
        llm,
        images=["data:image/png;base64,AAAA"],
    ).suggest(
        _analysis(),
        image_urls=["https://example.com/wolf.png"],
        visual_description="Detailed wolf head with pine trees.",
    )

    assert suggestion is not None
    assert (suggestion.minimum, suggestion.maximum) == (350, 510)
    assert suggestion.label() == "€350-510"
    assert suggestion.basis == "ai_market_estimate"
    assert suggestion.internal_only is True
    assert suggestion.client_safe_to_share is False
    human_content = llm.calls[0][1].content
    assert any(part.get("type") == "image_url" for part in human_content)
    assert "pine trees" in human_content[0]["text"]


def test_approved_history_anchors_the_estimate() -> None:
    llm = RecordingLLM()
    suggestion = _suggester(llm, records=[_approved_example()]).suggest(
        _analysis()
    )

    assert suggestion is not None
    assert suggestion.basis == "ai_with_approved_history"
    assert suggestion.similar_example_ids == ["feedback-1"]
    assert '"minimum": "400"' in llm.calls[0][1].content[0]["text"]


def test_history_only_fallback_when_model_unavailable() -> None:
    suggestion = _suggester(
        FailingLLM(),
        records=[
            _approved_example("feedback-1", 400, 550),
            _approved_example("feedback-2", 450, 600),
        ],
    ).suggest(_analysis())

    assert suggestion is not None
    assert suggestion.basis == "approved_history"
    assert (suggestion.minimum, suggestion.maximum) == (400, 600)


def test_no_suggestion_without_model_or_history() -> None:
    assert _suggester(FailingLLM()).suggest(_analysis()) is None
    assert _suggester(RecordingLLM("not json")).suggest(_analysis()) is None


def _brain(risk_level: str, llm: Any) -> StudioAIBrain:
    return StudioAIBrain(
        vision_analyzer=StubVisionAnalyzer(tags=["blackwork"]),
        text_extractor=StubTextExtractor(
            draft=TattooExtractionDraft(
                tattoo_idea="Wolf head with forest",
                style_tags=["blackwork"],
                placement="forearm",
                size_estimate_cm="15 cm",
                color_preference="black-and-grey",
                missing_information=[],
            )
        ),
        router=StubRouter(output=_analysis(risk_level)),
        price_suggester=_suggester(llm),
    )


def test_only_high_risk_requests_get_a_price_suggestion() -> None:
    high_llm = RecordingLLM()
    high = _brain("high", high_llm).process_inquiry(
        current_message="How much for a wolf on my forearm?",
        existing_db_state={"reference_images": ["https://example.com/a.png"]},
    )
    low_llm = RecordingLLM()
    low = _brain("low", low_llm).process_inquiry(
        current_message="Hi, I want a wolf tattoo.",
    )

    assert high.price_suggestion is not None
    assert high.price_suggestion.label() == "€350-510"
    assert low.price_suggestion is None
    assert low_llm.calls == []


def test_price_failure_never_breaks_the_reply() -> None:
    result = _brain("high", FailingLLM()).process_inquiry(
        current_message="How much for a wolf on my forearm?",
    )

    assert result.price_suggestion is None
    assert result.draft_reply == "Thanks, the studio will review this."


def test_telegram_summary_shows_the_price_line() -> None:
    analysis = _analysis().model_copy(
        update={
            "telegram_review_required": True,
            "price_suggestion": PriceSuggestion(
                minimum=350,
                maximum=500,
                estimated_hours="3-4 hours",
                confidence_level="medium",
            ),
        }
    )
    summary = HighRiskSummaryBuilder().build(
        TattooInquiryInput(current_message="How much?"),
        analysis,
    )

    assert summary.endswith(
        "Suggested price (AI estimate, not approved): €350-500, "
        "about 3-4 hours, medium confidence."
    )


def _store(tmp_path: Path) -> LearningStore:
    return LearningStore(
        vector_store=VectorStoreManager(
            embedding_model=DeterministicFakeEmbedding(size=8),
            embedding_dimension=8,
        ),
        index_path=str(tmp_path / "learning.faiss"),
    )


def _learning_client(store: LearningStore | None) -> TestClient:
    application = create_app()
    application.dependency_overrides[_optional_learning_store] = lambda: store
    return TestClient(application)


def _feedback(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "inquiry_id": "lead-129",
        "style_tags": ["blackwork"],
        "placement": "forearm",
        "size_estimate_cm": "15 cm",
        "color_preference": "black-and-grey",
        "ai_suggested_artist": "Sandra",
        "final_artist": "Hossam",
        "ai_suggestion_outcome": "corrected",
        "correction_reason": "Hossam had availability.",
        "approved_price_range": {
            "currency": "EUR",
            "minimum": "400",
            "maximum": "550",
        },
    }
    payload.update(overrides)
    return payload


def test_feedback_is_indexed_and_saved_to_disk(tmp_path: Path) -> None:
    store = _store(tmp_path)
    client = _learning_client(store)

    response = client.post("/api/v1/learning/feedback", json=_feedback())
    status = client.get("/api/v1/learning/status").json()

    assert response.status_code == 201
    assert response.json() == {"example_id": "feedback-lead-129", "record_count": 1}
    assert status == {
        "enabled": True,
        "record_count": 1,
        "records_with_approved_price": 1,
    }
    record = store.vector_store.records[0]
    assert record.original_ai_artist_key == "sandra"
    assert record.final_artist_key == "hoss"

    reloaded = VectorStoreManager(
        embedding_model=DeterministicFakeEmbedding(size=8),
        embedding_dimension=8,
    )
    reloaded.load_index(str(tmp_path / "learning.faiss"))
    assert reloaded.record_count == 1


def test_feedback_retries_and_conflicts(tmp_path: Path) -> None:
    client = _learning_client(_store(tmp_path))

    first = client.post("/api/v1/learning/feedback", json=_feedback())
    retry = client.post("/api/v1/learning/feedback", json=_feedback())
    conflict = client.post(
        "/api/v1/learning/feedback",
        json=_feedback(final_artist="Lana"),
    )

    assert first.status_code == 201
    assert retry.status_code == 201
    assert retry.json()["record_count"] == 1
    assert conflict.status_code == 409


def test_feedback_rejects_unknown_artist(tmp_path: Path) -> None:
    response = _learning_client(_store(tmp_path)).post(
        "/api/v1/learning/feedback",
        json=_feedback(final_artist="Somebody Else"),
    )

    assert response.status_code == 422


def test_learning_disabled(tmp_path: Path) -> None:
    client = _learning_client(None)

    assert client.post(
        "/api/v1/learning/feedback",
        json=_feedback(),
    ).status_code == 503
    assert client.get("/api/v1/learning/status").json()["enabled"] is False
