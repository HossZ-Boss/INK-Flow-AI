"""Regressions from the belly email thread (request #133)."""

from __future__ import annotations

from typing import Any, cast

import pytest
from langchain_openai import ChatOpenAI

from ai_brain.extraction import TattooTextExtractor
from ai_brain.schemas import Message


class FailingLLM:
    """Force the deterministic extraction path."""

    def invoke(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("LLM unavailable")


def _extractor() -> TattooTextExtractor:
    return TattooTextExtractor(llm=cast(ChatOpenAI, FailingLLM()))


@pytest.mark.parametrize(
    ("message", "placement"),
    [
        ("I need a tattoo on my belly.", "belly"),
        ("Something on my lower belly please.", "lower belly"),
        ("I'd like it on my tummy.", "stomach"),
        (
            "I have a tattoo on my hand, but in the belly I don't have any "
            "tattoos, so yeah.",
            "belly",
        ),
        ("I would like to have a tattoo on my hand.", "hand"),
        ("I want to cover up the tattoo on my hand.", "hand"),
    ],
)
def test_placement_from_text(message: str, placement: str) -> None:
    """Belly is a placement; an existing tattoo's location is not."""
    assert _extractor()._extract_placement_from_text(message) == placement


def test_existing_tattoo_mention_does_not_move_placement() -> None:
    """Explaining an old hand tattoo keeps the belly placement."""
    result = _extractor().extract(
        current_message=(
            "Actually, I want to visit the studio directly. I have a tattoo "
            "on my hand, but in the belly I don't have any tattoos, so yeah."
        ),
        style_tags=["neo-traditional", "illustrative"],
        existing_db_state={
            "lead": {"name": "Fahim Sarker", "source": "outlook"},
            "intake": {"placement": "belly"},
        },
    )

    assert result.placement == "belly"


def test_clarifying_question_is_not_the_tattoo_idea() -> None:
    """Asking what the assistant meant does not replace the design."""
    result = _extractor().extract(
        current_message="Yes, that's a new tattoo then.",
        style_tags=["neo-traditional", "illustrative"],
        existing_db_state={
            "lead": {"name": "Fahim Sarker", "source": "outlook"},
            "intake": {
                "tattoo_idea": "Horned beast head from reference",
                "placement": "belly",
            },
        },
        recent_chat_history=[
            Message(role="user", content="I need a tattoo on my belly."),
            Message(
                role="user",
                content=(
                    "What do you mean by the upper elements in the tattoo? "
                    "Are you talking about the butterfly/moth in the upper "
                    "belly or something else?"
                ),
            ),
        ],
    )

    assert result.tattoo_idea == "Horned beast head from reference"
    assert result.placement == "belly"
