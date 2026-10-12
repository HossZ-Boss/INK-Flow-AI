"""Regression tests for routing, LLM, and vision safeguards."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

import ai_brain.llm as llm_module
from ai_brain.config import LLMSettings
from ai_brain.routing import TattooRouter
from ai_brain.schemas import Message, TattooExtractionDraft
from ai_brain.vector_store import VectorStoreManager
from ai_brain.vision import TattooVisionAnalyzer


class FailingLLM:
    """LLM stub that always forces deterministic fallback logic."""

    def invoke(self, messages: object) -> object:
        """Raise a predictable provider failure."""
        raise RuntimeError("Simulated LLM failure")


class CapturingVisionLLM:
    """LLM stub that records messages sent by the vision analyzer."""

    def __init__(self) -> None:
        self.messages: list[BaseMessage] = []

    def invoke(self, messages: list[BaseMessage]) -> SimpleNamespace:
        """Capture messages and return a valid JSON style response."""
        self.messages = messages
        return SimpleNamespace(content='["fine-line"]')


def _router_with_failing_llm() -> TattooRouter:
    """Build a router whose LLM call always enters fallback logic."""
    llm = cast(ChatOpenAI, FailingLLM())
    vector_store = Mock(spec=VectorStoreManager)
    vector_store.records = tuple(range(10))
    vector_store.search_similar_cases.return_value = []
    return TattooRouter(
        llm=llm,
        vector_store=cast(VectorStoreManager, vector_store),
    )


def test_sensitive_intent_does_not_override_missing_information() -> None:
    """Pricing and advice remain low risk until intake is complete."""
    draft = TattooExtractionDraft(
        tattoo_idea="Client requests a price quote and complex design advice",
        style_tags=["unknown"],
        placement="",
        size_estimate_cm="",
        color_preference="",
        missing_information=[
            "size in cm",
            "placement",
            "reference images",
            "color preference",
            "preferred date",
        ],
    )

    result = _router_with_failing_llm().route(
        draft,
        current_message="I need complex design advice and a price quote.",
    )

    assert result.risk_level == "low"
    assert result.suggested_artist == "Unclear"
    assert result.confidence_level == "low"
    assert "What size would you prefer in centimetres?" in result.draft_reply
    assert result.draft_reply.count("?") <= 2
    assert result.auto_reply_allowed is True
    assert result.telegram_review_required is False


def test_price_history_remains_low_risk_while_information_is_missing() -> None:
    """A prior pricing request cannot bypass incomplete intake collection."""
    draft = TattooExtractionDraft(
        tattoo_idea="20cm black floral tattoo",
        style_tags=["floral"],
        placement="hand",
        size_estimate_cm="20cm",
        color_preference="black-and-grey",
        missing_information=["reference images", "preferred date"],
    )

    result = _router_with_failing_llm().route(
        draft,
        current_message="I would like to confirm the details.",
        recent_chat_history=[
            Message(
                role="assistant",
                content="Please share reference images for the appointment.",
            ),
            Message(role="user", content="What will be the price?"),
        ],
    )

    assert result.risk_level == "low"
    assert result.auto_reply_allowed is True
    assert result.telegram_review_required is False


def test_current_price_question_is_low_risk_while_fields_are_missing() -> None:
    """A current pricing request continues the intake collection flow."""
    draft = TattooExtractionDraft(
        tattoo_idea="20cm black floral tattoo",
        style_tags=["floral"],
        placement="hand",
        size_estimate_cm="20cm",
        color_preference="black-and-grey",
        missing_information=["reference images", "preferred date"],
    )

    result = _router_with_failing_llm().route(
        draft,
        current_message="How much would this tattoo cost?",
    )

    assert result.risk_level == "low"
    assert result.auto_reply_allowed is True
    assert result.telegram_review_required is False


def test_many_missing_fields_remain_low_risk() -> None:
    """Any non-empty missing-information list produces low risk."""
    draft = TattooExtractionDraft(
        tattoo_idea="A tattoo idea without intake details",
        style_tags=["unknown"],
        placement="",
        size_estimate_cm="",
        color_preference="",
        missing_information=[
            "size in cm",
            "placement",
            "reference images",
            "color preference",
            "preferred date",
        ],
    )

    result = _router_with_failing_llm().route(
        draft,
        current_message="I want to discuss a tattoo idea.",
    )

    assert result.risk_level == "low"
    assert result.auto_reply_allowed is True
    assert result.telegram_review_required is False


def test_router_keeps_basic_missing_information_low_risk() -> None:
    """Only missing size and placement remains a low-risk follow-up."""
    draft = TattooExtractionDraft(
        tattoo_idea="Minimal floral tattoo",
        style_tags=["minimal", "floral"],
        placement="",
        size_estimate_cm="",
        color_preference="black-and-grey",
        missing_information=["size in cm", "placement"],
    )

    result = _router_with_failing_llm().route(draft)

    assert result.risk_level == "low"
    assert result.auto_reply_allowed is True
    assert result.telegram_review_required is False
    assert result.suggested_artist == "Lana"
    assert result.confidence_level == "medium"
    assert "black-and-grey minimal floral tattoo" in result.draft_reply
    assert "Does that sound right" in result.draft_reply
    assert "- Style:" not in result.draft_reply
    assert "Unknown" not in result.draft_reply


def test_complete_intake_is_high_risk_without_sensitive_intent() -> None:
    """A request with no missing information always requires high-risk review."""
    draft = TattooExtractionDraft(
        tattoo_idea="Fine-line floral tattoo",
        style_tags=["fine-line"],
        placement="inner wrist",
        size_estimate_cm="8cm",
        color_preference="black-and-grey",
        missing_information=[],
    )

    result = _router_with_failing_llm().route(
        draft,
        current_message="These are all the details for my tattoo.",
    )

    assert result.risk_level == "high"
    assert result.auto_reply_allowed is False
    assert result.telegram_review_required is True


def test_complete_price_request_is_high_risk() -> None:
    """Pricing reaches staff review after every requirement is collected."""
    draft = TattooExtractionDraft(
        tattoo_idea="Fine-line floral tattoo",
        style_tags=["fine-line", "floral"],
        placement="inner wrist",
        size_estimate_cm="8cm",
        color_preference="black-and-grey",
        date="2026-09-04",
        time="14:30",
        missing_information=[],
    )

    result = _router_with_failing_llm().route(
        draft,
        current_message="Now that the details are complete, what is the price?",
    )

    assert result.risk_level == "high"
    assert result.auto_reply_allowed is False
    assert result.telegram_review_required is True
    assert "studio team review" in result.draft_reply


def test_artist_routing_does_not_force_incomplete_intake_to_high_risk() -> None:
    """Artist routing does not override missing-based risk."""
    router = TattooRouter(llm=cast(ChatOpenAI, FailingLLM()))
    draft = TattooExtractionDraft(
        tattoo_idea="Floral tattoo",
        style_tags=["floral"],
        placement="",
        size_estimate_cm="",
        color_preference="",
        missing_information=["size in cm", "placement"],
    )

    result = router.route(draft, current_message="I want a floral tattoo.")

    assert result.suggested_artist == "Lana"
    assert result.risk_level == "low"
    assert result.auto_reply_allowed is True
    assert result.telegram_review_required is False


@pytest.mark.parametrize(
    "current_message",
    [
        "Can you give me a price?",
        "I want to book this tattoo.",
        "Can I pay the deposit now?",
        "Please cancel my appointment.",
        "Can we reschedule this?",
        "I need to make a complaint.",
        "Please process my refund.",
        "I need complex design advice.",
        "I have a preferred artist.",
    ],
)
def test_incomplete_sensitive_intents_remain_low_risk(
    current_message: str,
) -> None:
    """Intent keywords cannot bypass the completeness-based risk gate."""
    draft = TattooExtractionDraft(
        tattoo_idea="Fine-line floral tattoo",
        style_tags=["fine-line"],
        placement="inner wrist",
        size_estimate_cm="8cm",
        color_preference="black-and-grey",
        missing_information=["preferred date"],
    )

    result = _router_with_failing_llm().route(
        draft,
        current_message=current_message,
    )

    assert result.risk_level == "low"
    assert result.auto_reply_allowed is True
    assert result.telegram_review_required is False


def test_chat_model_forces_zero_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chat model remains deterministic regardless of environment setting."""
    captured: dict[str, Any] = {}
    settings = LLMSettings(
        api_key="test-key",
        temperature=1.0,
        timeout_seconds=30,
        max_retries=2,
    )

    class FakeChatOpenAI:
        """Capture constructor options without creating an OpenAI client."""

        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    llm_module.get_chat_model.cache_clear()
    monkeypatch.setattr(llm_module, "get_settings", lambda: settings)
    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)

    llm_module.get_chat_model(model_name="determinism-test")

    assert captured["temperature"] == 0.0
    llm_module.get_chat_model.cache_clear()


def test_latest_reasoning_model_uses_responses_without_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPT-6 Astra follows the documented reasoning-model parameters."""
    captured: dict[str, Any] = {}
    settings = LLMSettings(
        api_key="test-key",
        model_name="gpt-6-astra",
        reasoning_effort="low",
        temperature=1.0,
        timeout_seconds=30,
        max_retries=2,
    )

    class FakeChatOpenAI:
        """Capture constructor options without creating an API client."""

        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    llm_module.get_chat_model.cache_clear()
    monkeypatch.setattr(llm_module, "get_settings", lambda: settings)
    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)

    llm_module.get_chat_model()

    assert captured["model"] == "gpt-6-astra"
    assert captured["reasoning_effort"] == "low"
    assert captured["use_responses_api"] is True
    assert "temperature" not in captured
    llm_module.get_chat_model.cache_clear()


def test_vision_user_message_contains_only_images() -> None:
    """Vision user payload avoids redundant text instructions."""
    captured_llm = CapturingVisionLLM()
    analyzer = TattooVisionAnalyzer(
        llm=cast(ChatOpenAI, captured_llm),
    )

    response = analyzer._invoke_vision_model(
        ["data:image/jpeg;base64,dGVzdA=="]
    )

    assert response == '["fine-line"]'
    assert len(captured_llm.messages) == 2

    human_message = captured_llm.messages[1]
    assert isinstance(human_message, HumanMessage)
    assert isinstance(human_message.content, list)
    assert human_message.content == [
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,dGVzdA==",
            },
        }
    ]
