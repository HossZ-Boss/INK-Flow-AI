"""Tests for learning-aware and configuration-driven artist routing."""

from __future__ import annotations

from typing import cast

import pytest
from langchain_openai import ChatOpenAI

from ai_brain.artist_config import ArtistConfigManager, ArtistProfile
from ai_brain.decision import StudioDecisionEngine
from ai_brain.decision_schemas import (
    ArtistOption,
    DecisionHistoryExample,
    StudioDecisionContext,
)
from ai_brain.routing import TattooRouter
from ai_brain.routing_rules import (
    RoutingRule,
    RoutingRuleEngine,
    StudioRoutingRulesConfig,
)
from ai_brain.schemas import (
    AIExtractionOutput,
    StyleTag,
    TattooExtractionDraft,
)
from ai_brain.vector_store import VectorStoreManager


class FailingLLM:
    """Force deterministic routing and reply fallbacks in unit tests."""

    def invoke(self, messages: object) -> object:
        """Simulate an unavailable external model provider."""
        raise RuntimeError("External LLM calls are disabled in tests.")


class FakeVectorStore:
    """Expose deterministic records and similarity results without FAISS."""

    def __init__(
        self,
        records: list[DecisionHistoryExample],
        matches: list[tuple[float, DecisionHistoryExample]] | None = None,
    ) -> None:
        """Store an immutable record snapshot and optional search results."""
        self.records = tuple(records)
        self._matches = list(matches if matches is not None else [])
        self.search_calls: list[tuple[str, int]] = []

    def search_similar_cases(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> list[tuple[float, DecisionHistoryExample]]:
        """Return configured matches while recording the vector query."""
        self.search_calls.append((query_text, top_k))
        return self._matches[:top_k]


def _history_example(
    example_id: str,
    artist_key: str,
    style: StyleTag = "watercolor",
) -> DecisionHistoryExample:
    """Build one verified historical assignment for routing tests."""
    return DecisionHistoryExample.model_validate(
        {
            "example_id": example_id,
            "channel": "whatsapp",
            "style_tags": [style],
            "placement": "upper arm",
            "size_estimate_cm": "15cm",
            "color_preference": "color",
            "original_ai_artist_key": None,
            "final_artist_key": artist_key,
            "original_ai_action": "artist_review",
            "final_action": "ready_to_book",
            "ai_suggestion_outcome": "corrected",
            "correction_reason": "Assignment approved by studio staff.",
        }
    )


def _history_records(
    assignments: list[str],
    style: StyleTag = "watercolor",
) -> list[DecisionHistoryExample]:
    """Build uniquely identified records for the supplied artist keys."""
    return [
        _history_example(
            example_id=f"case-{index}",
            artist_key=artist_key,
            style=style,
        )
        for index, artist_key in enumerate(assignments)
    ]


def _analysis() -> AIExtractionOutput:
    """Build a complete analysis for internal decision-engine tests."""
    return AIExtractionOutput(
        tattoo_idea="Watercolor eagle tattoo",
        style_tags=["watercolor"],
        placement="upper arm",
        size_estimate_cm="15cm",
        color_preference="color",
        suggested_artist="Hossam",
        confidence_level="medium",
        ai_reasoning="Initial routing result.",
        missing_information=[],
        risk_level="low",
        draft_reply="The studio will review your tattoo request.",
    )


def _draft() -> TattooExtractionDraft:
    """Build extracted features that do not match the default custom rules."""
    return TattooExtractionDraft(
        tattoo_idea="Watercolor eagle tattoo",
        style_tags=["watercolor"],
        placement="upper arm",
        size_estimate_cm="15cm",
        color_preference="color",
        missing_information=[],
    )


def _decision_context() -> StudioDecisionContext:
    """Return the artist catalog required by internal decisions."""
    return StudioDecisionContext(
        channel="whatsapp",
        artist_options=[
            ArtistOption(key="hoss", display_name="Hossam"),
            ArtistOption(key="sliva", display_name="Sliva"),
        ],
    )


def _empty_rule_engine() -> RoutingRuleEngine:
    """Create a rule engine that leaves vector voting unmodified."""
    return RoutingRuleEngine(config=StudioRoutingRulesConfig())


def _router(
    vector_store: FakeVectorStore,
    artist_config: ArtistConfigManager | None = None,
    rule_engine: RoutingRuleEngine | None = None,
) -> TattooRouter:
    """Build an offline router with explicit learning dependencies."""
    return TattooRouter(
        llm=cast(ChatOpenAI, FailingLLM()),
        vector_store=cast(VectorStoreManager, vector_store),
        artist_config_manager=artist_config,
        routing_rule_engine=rule_engine or _empty_rule_engine(),
    )


@pytest.mark.parametrize("record_count", [0, 1, 3, 9])
def test_learning_is_used_without_a_minimum_record_count(
    record_count: int,
) -> None:
    """There is no cold-start gate: any reviewed history is searched."""
    records = _history_records(["hoss"] * record_count)
    matches = [(0.99, record) for record in records]
    vector_store = FakeVectorStore(records=records, matches=matches)
    engine = StudioDecisionEngine(
        vector_store=cast(VectorStoreManager, vector_store)
    )

    result = engine.decide(
        analysis=_analysis(),
        current_message="I want a watercolor eagle tattoo.",
        context=_decision_context(),
    )
    routed = _router(vector_store).route(
        extracted=_draft(),
        current_message="I want a watercolor eagle tattoo.",
    )

    assert "Cold start" not in result.analysis.ai_reasoning
    assert "Cold start" not in routed.ai_reasoning
    assert routed.suggested_artist == "Hossam"
    assert result.artist_suggestion.artist_key == "hoss"
    assert len(vector_store.search_calls) == 2


def test_history_votes_select_the_artist() -> None:
    """Similar-case frequency selects the active artist."""
    records = _history_records(["hoss"] * 6 + ["sliva"] * 4)
    matches = [
        (1.0 - index / 100, record)
        for index, record in enumerate(records)
    ]
    vector_store = FakeVectorStore(records=records, matches=matches)

    result = _router(vector_store).route(
        extracted=_draft(),
        current_message="I want a watercolor eagle tattoo.",
    )

    assert result.suggested_artist == "Hossam"
    assert "6 similar historical cases" in result.ai_reasoning
    assert len(vector_store.search_calls) == 1


def test_history_artist_must_match_detected_style() -> None:
    """Historical votes cannot bypass configured artist specialties."""
    draft = TattooExtractionDraft(
        tattoo_idea="Fine-line flower",
        style_tags=["fine-line"],
        placement="inner forearm",
        size_estimate_cm="8cm",
        color_preference="black-and-grey",
        missing_information=[],
    )
    records = _history_records(["hoss"] * 10, style="fine-line")
    matches = [(0.99, record) for record in records]

    result = _router(
        FakeVectorStore(records=records, matches=matches)
    ).route(extracted=draft)

    assert result.suggested_artist == "Unclear"
    assert "no active artist assignments" in result.ai_reasoning.casefold()


def test_artist_config_validation() -> None:
    """Default profiles are valid and inactive artists are never suggested."""
    manager = ArtistConfigManager()
    artists = manager.config.artists

    assert {artist.artist_key for artist in artists} == {
        "lana",
        "nina",
        "hoss",
        "sliva",
        "sandra",
        "mila",
    }
    assert all(artist.display_name for artist in artists)
    assert all(artist.specialties for artist in artists)
    assert all(artist.profile_summary for artist in artists)
    assert all(artist.portfolio_handles for artist in artists)
    assert {artist.display_name for artist in artists} == {
        "Lana",
        "Nina",
        "Hossam",
        "Sliva",
        "Sandra",
        "Mila",
    }
    assert manager.validate_artist_assignment(
        "hoss",
        ["watercolor"],
    )
    assert not manager.validate_artist_assignment(
        "hoss",
        ["fine-line"],
    )
    manager.set_artist_active("hoss", False)

    records = _history_records(["hoss"] * 10)
    matches = [(0.95, record) for record in records]
    result = _router(
        FakeVectorStore(records=records, matches=matches),
        artist_config=manager,
    ).route(extracted=_draft())

    assert result.suggested_artist == "Unclear"
    assert "no active artist assignments" in result.ai_reasoning.casefold()


@pytest.mark.parametrize(
    ("style", "expected_artist"),
    [
        ("botanical", "Lana"),
        ("fine-illustrative", "Nina"),
        ("watercolor", "Hossam"),
        ("pixel-art", "Sliva"),
        ("blackwork", "Sandra"),
        ("new-school", "Mila"),
    ],
)
def test_updated_portfolio_styles_route_to_the_right_artist(
    style: StyleTag,
    expected_artist: str,
) -> None:
    """Distinctive updated portfolio styles select their specialist."""
    draft = TattooExtractionDraft(
        tattoo_idea=f"A {style} design",
        style_tags=[style],
        placement="forearm",
        size_estimate_cm="10 cm",
        color_preference="color",
        missing_information=["appointment type"],
    )

    result = TattooRouter(
        llm=cast(ChatOpenAI, FailingLLM()),
    ).route(extracted=draft)

    assert result.suggested_artist == expected_artist


def test_routing_rules_engine() -> None:
    """Matching high-priority rules override lower rules and vector history."""
    rule_engine = RoutingRuleEngine(
        config=StudioRoutingRulesConfig(
            rules=[
                RoutingRule(
                    name="Fine-line preference",
                    condition="style_tags contains 'fine-line'",
                    action="suggest_artist: nina",
                    priority=10,
                ),
                RoutingRule(
                    name="Small forearm specialist",
                    condition=(
                        "style_tags contains 'fine-line' AND size < 10cm "
                        "AND placement contains 'forearm'"
                    ),
                    action="suggest_artist: lana",
                    priority=100,
                ),
            ]
        )
    )
    draft = TattooExtractionDraft(
        tattoo_idea="Fine-line flower",
        style_tags=["fine-line"],
        placement="inner forearm",
        size_estimate_cm="8cm",
        color_preference="black-and-grey",
        missing_information=[],
    )
    records = _history_records(["hoss"] * 10, style="fine-line")
    matches = [(0.98, record) for record in records]

    applicable = rule_engine.evaluate(draft)
    result = _router(
        FakeVectorStore(records=records, matches=matches),
        rule_engine=rule_engine,
    ).route(extracted=draft)

    assert [rule.priority for rule in applicable] == [100, 10]
    assert result.suggested_artist == "Lana"
    assert "Small forearm specialist" in result.ai_reasoning
    assert "priority 100" in result.ai_reasoning
    assert "Applied routing rules" in result.ai_reasoning


def test_artist_expansion() -> None:
    """A runtime-added artist can win history voting without code changes."""
    manager = ArtistConfigManager()
    manager.upsert_artist(
        ArtistProfile(
            artist_key="marcus",
            display_name="Marcus",
            specialties=["traditional"],
            min_size_cm=5,
            max_size_cm=30,
            is_active=True,
        )
    )
    records = _history_records(
        ["marcus"] * 6 + ["hoss"] * 4,
        style="traditional",
    )
    matches = [(0.99, record) for record in records]

    draft = TattooExtractionDraft(
        tattoo_idea="Traditional eagle tattoo",
        style_tags=["traditional"],
        placement="upper arm",
        size_estimate_cm="15cm",
        color_preference="color",
        missing_information=[],
    )
    result = _router(
        FakeVectorStore(records=records, matches=matches),
        artist_config=manager,
    ).route(extracted=draft)

    assert manager.validate_artist_assignment("marcus") is True
    assert result.suggested_artist == "Marcus"
    assert "6 similar historical cases" in result.ai_reasoning
