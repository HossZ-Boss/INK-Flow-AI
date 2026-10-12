"""A bare weekday is confirmed with the client before it is saved."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, cast

import pytest
from langchain_openai import ChatOpenAI

from ai_brain.extraction import TattooTextExtractor
from ai_brain.reply import ConversationReplyComposer, date_confirmation_question
from ai_brain.routing import TattooRouter
from ai_brain.schemas import Message


class FailingLLM:
    """Force the deterministic extraction and template paths."""

    def invoke(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("LLM unavailable")


def _extractor() -> TattooTextExtractor:
    return TattooTextExtractor(llm=cast(ChatOpenAI, FailingLLM()))


def _upcoming(weekday: int) -> date:
    today = date.today()
    return today + timedelta(days=((weekday - today.weekday()) % 7) or 7)


_STATE = {
    "lead": {"name": "Fahim Sarker", "source": "whatsapp"},
    "intake": {"tattoo_idea": "Rose", "placement": "forearm"},
}


@pytest.mark.parametrize(
    "message",
    [
        "I want to come on Thursday.",
        "Can I come next Thursday?",
        "This thursday works for me",
    ],
)
def test_bare_weekday_is_not_saved_as_the_date(message: str) -> None:
    """The weekday becomes a date to confirm, not the preferred date."""
    result = _extractor().extract(
        current_message=message,
        style_tags=["fine-line"],
        existing_db_state=_STATE,
    )

    assert result.date == ""
    assert result.date_to_confirm == _upcoming(3).isoformat()
    assert "preferred dates or availability" in result.missing_information


def test_explicit_calendar_date_is_saved_directly() -> None:
    """A weekday with a calendar date needs no confirmation."""
    target = _upcoming(3)
    result = _extractor().extract(
        current_message=f"Thursday, {target.day} {target:%B} works for me.",
        style_tags=["fine-line"],
        existing_db_state=_STATE,
    )

    assert result.date == target.isoformat()
    assert result.date_to_confirm == ""


@pytest.mark.parametrize("reply", ["Yes", "Yes, that's right.", "Thursday works"])
def test_confirmation_saves_the_offered_date(reply: str) -> None:
    """A yes to the studio's exact-date question saves that date."""
    target = _upcoming(3)
    history = [
        Message(role="user", content="I want to come on Thursday."),
        Message(
            role="assistant",
            content=f"Got it. {date_confirmation_question(target.isoformat())}",
        ),
    ]
    result = _extractor().extract(
        current_message=reply,
        style_tags=["fine-line"],
        existing_db_state=_STATE,
        recent_chat_history=history,
    )

    assert result.date == target.isoformat()
    assert result.date_to_confirm == ""
    assert "preferred dates or availability" not in result.missing_information


def test_template_reply_asks_for_the_exact_date() -> None:
    """The fallback reply asks "do you mean Thursday, 15 October?"."""
    target = _upcoming(3)
    extracted = _extractor().extract(
        current_message="I want to come on Thursday.",
        style_tags=["fine-line"],
        existing_db_state=_STATE,
        recent_chat_history=[
            Message(role="user", content="I want a rose on my forearm."),
            Message(role="assistant", content="Lovely! What size?"),
        ],
    )
    reply = ConversationReplyComposer().compose(
        extracted=extracted,
        current_message="I want to come on Thursday.",
        recent_chat_history=[
            Message(role="user", content="I want a rose on my forearm."),
            Message(role="assistant", content="Lovely! What size?"),
        ],
        risk_level="low",
    )

    assert f"do you mean {target:%A}, {target.day} {target:%B}?" in reply


def test_llm_draft_without_the_date_question_falls_back() -> None:
    """A model draft that skips the confirmation is rejected."""
    extracted = _extractor().extract(
        current_message="I want to come on Thursday.",
        style_tags=["fine-line"],
        existing_db_state=_STATE,
    )
    router = TattooRouter(llm=cast(ChatOpenAI, FailingLLM()))

    with pytest.raises(ValueError, match="confirm the exact date"):
        router._validate_draft_reply(
            "Great, Thursday it is! What size would you like?",
            extracted,
            "whatsapp",
            "I want to come on Thursday.",
        )
