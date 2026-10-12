"""Regression tests for defensive Outlook and Gmail reply cleanup."""

from __future__ import annotations

from datetime import date as calendar_date
from types import SimpleNamespace
from typing import cast

from langchain_openai import ChatOpenAI

from ai_brain.email_cleaning import strip_quoted_email_content
from ai_brain.extraction import TattooTextExtractor
from ai_brain.reply import ConversationReplyComposer
from ai_brain.routing import TattooRouter
from ai_brain.schemas import TattooInquiryInput


class FailingExtractionLLM:
    """Force field-specific deterministic extraction."""

    def invoke(self, messages: object) -> SimpleNamespace:
        """Simulate an unavailable extraction provider."""
        raise RuntimeError("simulated extraction failure")


def test_plain_gmail_quote_is_removed_from_outlook_input() -> None:
    """Only the newest reply reaches AI extraction and routing."""
    raw_message = (
        "3 cm, colorful flower, I prefer Hoss, a studio visit, and a new "
        "tattoo.\n\n"
        "On Wed, Sep 9, 2026 at 3:23 PM Tattoo Hysteria wrote:\n"
        "> Would you like colour or black and grey?\n"
        "> Choose an artist or say no preference.\n"
        "> Is this a new tattoo, cover-up, continuation, or touch-up?"
    )

    inquiry = TattooInquiryInput(
        current_message=raw_message,
        message_source="outlook",
    )

    assert inquiry.current_message == (
        "3 cm, colorful flower, I prefer Hoss, a studio visit, and a new "
        "tattoo."
    )


def test_outlook_headers_html_quotes_and_angle_quotes_are_removed() -> None:
    """Common Outlook, Gmail HTML, and plain quote markers are supported."""
    assert strip_quoted_email_content(
        "I prefer Hoss.\n\nFrom: Tattoo Hysteria\nSent: Wednesday"
    ) == "I prefer Hoss."
    assert strip_quoted_email_content(
        '<div>Studio visit<br>please</div><div class="gmail_quote">old</div>'
    ) == "Studio visit\nplease"
    assert strip_quoted_email_content(
        "A colorful flower.\n> Previous message"
    ) == "A colorful flower."
    assert strip_quoted_email_content(
        "A colorful flower.\n\nOn Wed, Sep 9, 2026 at 3:23 PM\n"
        "Tattoo Hysteria <studio@example.com> wrote:\nOld message"
    ) == "A colorful flower."


def test_first_email_with_only_logistics_keeps_tattoo_idea_missing() -> None:
    """Placement and scheduling details are not promoted into a concept."""
    extractor = TattooTextExtractor(
        llm=cast(ChatOpenAI, FailingExtractionLLM()),
    )

    result = extractor.extract(
        current_message=(
            "Hello Tattoo Hysteria, I'm very excited to get a tattoo on my "
            "hand at your shop this Saturday."
        ),
        style_tags=["unknown"],
        existing_db_state={"lead": {"name": "Fahim Sarker"}},
    )

    assert result.tattoo_idea == ""
    assert "tattoo idea" in result.missing_information


def test_realistic_second_email_extracts_only_client_answers() -> None:
    """Quoted questionnaire choices cannot override explicit client answers."""
    inquiry = TattooInquiryInput(
        current_message=(
            "3 cm, colorful flower, I prefer Hoss and will visit the studio. "
            "This is a new tattoo.\n\n"
            "On Wed, Sep 9, 2026 at 3:23 PM Tattoo Hysteria wrote:\n"
            "> Would you like colour or black and grey?\n"
            "> Choose Hoss, Nina, Lana, Sandra, Silva, or no preference.\n"
            "> Is this a new tattoo, cover-up, continuation, or touch-up?"
        ),
        message_source="outlook",
        new_image_urls=["https://example.com/reference.jpg"],
        existing_db_state={
            "lead": {"name": "Fahim Sarker", "source": "outlook"},
            "intake": {
                "placement": "hand",
                "date": "2026-09-12",
                "availability": "2026-09-12",
            },
        },
    )
    extractor = TattooTextExtractor(
        llm=cast(ChatOpenAI, FailingExtractionLLM()),
    )

    result = extractor.extract(
        current_message=inquiry.current_message,
        style_tags=["floral"],
        visual_color_preference="unknown",
        new_image_urls=inquiry.new_image_urls,
        existing_db_state=inquiry.existing_db_state,
        recent_chat_history=inquiry.recent_chat_history,
    )

    assert result.tattoo_idea == "Colorful flower"
    assert result.placement == "hand"
    assert result.size_estimate_cm == "3 cm"
    assert result.color_preference == "color"
    assert result.date == "2026-09-12"
    assert result.time == ""
    assert result.preferred_artist == "Hossam"
    assert result.appointment_type == "studio_visit"
    assert result.tattoo_project_type == "new tattoo"
    assert result.missing_information == ["tattoo style"]

    reply = ConversationReplyComposer().compose_outlook_email(
        result,
        existing_db_state=inquiry.existing_db_state,
    )
    assert "3 cm colour floral colorful flower tattoo" in reply
    assert "To confirm what I have so far" in reply
    assert "- Preferred time:" not in reply
    assert "recorded the following" not in reply.casefold()
    assert "- Availability:" not in reply
    assert "What tattoo style would you prefer?" in reply
    assert "fine-line, watercolor, minimal, floral" in reply
    assert "On Wed" not in reply


def test_artist_only_followup_preserves_every_other_stored_value() -> None:
    """A narrow correction updates only the field the client mentions."""
    inquiry = TattooInquiryInput(
        current_message=(
            "I'd prefer Hoss as my artist.\n\n"
            "On Wed, Sep 9, 2026 at 3:24 PM Tattoo Hysteria wrote:\n"
            "> Please choose an artist or say no preference."
        ),
        message_source="outlook",
        new_image_urls=["https://example.com/reference.jpg"],
        existing_db_state={
            "lead": {"name": "Fahim Sarker"},
            "intake": {
                "tattoo_idea": "Colorful flower",
                "placement": "hand",
                "size_estimate_cm": "3 cm",
                "color_preference": "color",
                "date": "2026-09-12",
                "availability": "2026-09-12",
                "appointment_type": "studio_visit",
                "tattoo_project_type": "new tattoo",
            },
        },
    )
    extractor = TattooTextExtractor(
        llm=cast(ChatOpenAI, FailingExtractionLLM()),
    )

    result = extractor.extract(
        current_message=inquiry.current_message,
        style_tags=["fine-line"],
        new_image_urls=inquiry.new_image_urls,
        existing_db_state=inquiry.existing_db_state,
    )

    assert result.tattoo_idea == "Colorful flower"
    assert result.color_preference == "color"
    assert result.time == ""
    assert result.preferred_artist == "Hossam"
    assert result.appointment_type == "studio_visit"
    assert result.tattoo_project_type == "new tattoo"


def test_calligraphy_style_without_wording_keeps_tattoo_idea_missing() -> None:
    """A technique name is not accepted as the requested tattoo content."""
    message = (
        "Hello Tattoo Hysteria, I want to get a tattoo in a calligraphy "
        "style on my hand. How much will it cost and when can I come to "
        "your studio?"
    )
    extractor = TattooTextExtractor(
        llm=cast(ChatOpenAI, FailingExtractionLLM()),
    )

    result = extractor.extract(
        current_message=message,
        style_tags=["unknown"],
        existing_db_state={"lead": {"name": "Fahim Sarker"}},
    )

    assert result.tattoo_idea == ""
    assert result.style_tags == ["calligraphy"]
    assert result.placement == "hand"
    assert result.appointment_type == "studio_visit"
    assert "tattoo idea" in result.missing_information
    assert "tattoo style" not in result.missing_information
    assert "appointment type" not in result.missing_information

    reply = ConversationReplyComposer().compose_outlook_email(
        result,
        existing_db_state={"lead": {"name": "Fahim Sarker"}},
        current_message=message,
    )
    assert "the artist can provide a custom estimate" in reply
    assert "What is your tattoo idea or background story?" in reply
    assert "online appointment or a studio visit" not in reply


def test_real_email_answers_normalize_inches_and_black_and_grey() -> None:
    """Imperial size and a specific colour phrase resolve deterministically."""
    first_message = (
        "I want a tattoo in a calligraphy style on my hand. How much will it "
        "cost and when can I come to your studio?"
    )
    second_message = (
        "It's roughly about 5 inches. Black and gray color. I'd prefer Silva. "
        "I'll visit the studio. I'd prefer Thursday on the next week. "
        "That's going to be a new tattoo."
    )
    history = [
        {"role": "user", "content": first_message},
        {
            "role": "assistant",
            "content": "Please send the remaining details for a price.",
        },
    ]
    inquiry = TattooInquiryInput(
        current_message=second_message,
        message_source="outlook",
        existing_db_state={
            "lead": {"name": "Fahim Sarker", "source": "outlook"},
            "intake": {
                "tattoo_idea": "Calligraphy",
                "placement": "hand",
                "appointment_type": "studio_visit",
            },
        },
        recent_chat_history=history,
    )
    extractor = TattooTextExtractor(
        llm=cast(ChatOpenAI, FailingExtractionLLM()),
    )

    result = extractor.extract(
        current_message=inquiry.current_message,
        style_tags=["calligraphy"],
        existing_db_state=inquiry.existing_db_state,
        recent_chat_history=inquiry.recent_chat_history,
    )

    next_thursday_offset = (3 - calendar_date.today().weekday()) % 7 or 7
    expected_date = calendar_date.fromordinal(
        calendar_date.today().toordinal() + next_thursday_offset
    ).isoformat()
    assert result.size_estimate_cm == "12.7 cm"
    assert result.color_preference == "black-and-grey"
    assert result.date == ""
    assert result.date_to_confirm == expected_date
    assert result.preferred_artist == "Sliva"
    assert result.appointment_type == "studio_visit"
    assert result.tattoo_project_type == "new tattoo"
    assert result.tattoo_idea == ""
    assert result.missing_information == [
        "tattoo idea",
        "reference images",
        "preferred dates or availability",
    ]

    reply = ConversationReplyComposer().compose_outlook_email(
        result,
        existing_db_state=inquiry.existing_db_state,
        current_message=inquiry.current_message,
        recent_chat_history=inquiry.recent_chat_history,
    )
    assert "Thanks, I've noted the tattoo type" in reply
    assert "We still need the following information" not in reply
    assert "custom estimate" not in reply
    assert "12.7 cm black-and-grey calligraphy tattoo" not in reply
    assert "What size would you prefer" not in reply

    routed = TattooRouter(
        llm=cast(ChatOpenAI, FailingExtractionLLM()),
    ).route(
        extracted=result,
        current_message=inquiry.current_message,
        recent_chat_history=inquiry.recent_chat_history,
        existing_db_state=inquiry.existing_db_state,
        message_source="outlook",
    )
    assert routed.risk_level == "low"
    assert routed.telegram_review_required is False


def test_latest_centimetre_answer_overrides_earlier_inches() -> None:
    """A later 5 cm correction replaces the converted 5-inch value."""
    extractor = TattooTextExtractor(
        llm=cast(ChatOpenAI, FailingExtractionLLM()),
    )

    result = extractor.extract(
        current_message="It's going to be about 5 cm.",
        style_tags=["calligraphy"],
        existing_db_state={
            "lead": {"name": "Fahim Sarker"},
            "intake": {
                "tattoo_idea": "Name and date in calligraphy",
                "placement": "hand",
                "size_estimate_cm": "12.7 cm",
                "color_preference": "black-and-grey",
                "preferred_artist": "Silva",
                "appointment_type": "studio_visit",
                "availability": "2026-09-17",
                "tattoo_project_type": "new tattoo",
            },
        },
        new_image_urls=["https://example.com/reference.jpg"],
    )

    assert result.size_estimate_cm == "5 cm"
    assert result.tattoo_idea == "Name and date in calligraphy"


def test_legacy_artist_names_normalize_to_updated_display_names() -> None:
    """Existing Hoss and Silva records remain compatible with the new names."""
    extractor = TattooTextExtractor(
        llm=cast(ChatOpenAI, FailingExtractionLLM()),
    )

    hossam = extractor.extract(
        current_message="I prefer Hoss.",
        style_tags=["watercolor"],
        existing_db_state={"lead": {"name": "Fahim Sarker"}},
    )
    sliva = extractor.extract(
        current_message="I prefer Silva.",
        style_tags=["pixel-art"],
        existing_db_state={"lead": {"name": "Fahim Sarker"}},
    )
    mila = extractor.extract(
        current_message="I prefer Mila.",
        style_tags=["new-school"],
        existing_db_state={"lead": {"name": "Fahim Sarker"}},
    )

    assert hossam.preferred_artist == "Hossam"
    assert sliva.preferred_artist == "Sliva"
    assert mila.preferred_artist == "Mila"


def test_calligraphic_wording_conversation_preserves_the_complete_quote() -> None:
    """Quoted wording wins over subject words appearing inside the sentence."""
    first_message = (
        "Hello Tattoo Hysteria, I want to get a tattoo on my leg. How much "
        "will it cost? The tattoo will be in English wording, but in a "
        "calligraphic style."
    )
    extractor = TattooTextExtractor(
        llm=cast(ChatOpenAI, FailingExtractionLLM()),
    )
    first = extractor.extract(
        current_message=first_message,
        style_tags=["unknown"],
        existing_db_state={"lead": {"name": "Fahim Sarker"}},
    )

    assert first.tattoo_idea == ""
    assert first.style_tags == ["calligraphy"]
    assert first.placement == "leg"
    assert "tattoo idea" in first.missing_information
    assert "tattoo style" not in first.missing_information

    first_reply = ConversationReplyComposer().compose_outlook_email(
        first,
        existing_db_state={"lead": {"name": "Fahim Sarker"}},
        current_message=first_message,
    )
    assert "calligraphy tattoo on your leg" in first_reply
    assert "What tattoo style would you prefer?" not in first_reply

    next_wednesday_offset = (
        2 - calendar_date.today().weekday()
    ) % 7 or 7
    expected_wednesday = calendar_date.fromordinal(
        calendar_date.today().toordinal() + next_wednesday_offset
    ).isoformat()
    wednesday = calendar_date.fromisoformat(expected_wednesday)
    second_message = (
        'Tattoo idea is on the leg: "You Cannot reason with a tiger when your '
        'head is in his mouth!!" will be written in a calligraphic style. '
        "Size will be approximately 12 centimeters. Black and gray color. "
        "I'd prefer Hoss. I'd visit the studio; my preferred date is "
        f"{wednesday.day} {wednesday:%B}. And that's going to be a new tattoo."
    )
    history = [
        {"role": "user", "content": first_message},
        {"role": "assistant", "content": first_reply},
    ]
    inquiry = TattooInquiryInput(
        current_message=second_message,
        message_source="outlook",
        existing_db_state={
            "lead": {"name": "Fahim Sarker", "source": "outlook"},
            "intake": {
                "placement": first.placement,
                "style_tags": first.style_tags,
            },
        },
        recent_chat_history=history,
    )
    second = extractor.extract(
        current_message=inquiry.current_message,
        style_tags=["unknown"],
        existing_db_state=inquiry.existing_db_state,
        recent_chat_history=inquiry.recent_chat_history,
    )

    assert second.tattoo_idea == (
        'Calligraphy wording: "You Cannot reason with a tiger when your head '
        'is in his mouth!!"'
    )
    assert second.style_tags == ["calligraphy"]
    assert second.placement == "leg"
    assert second.size_estimate_cm == "12 cm"
    assert second.color_preference == "black-and-grey"
    assert second.date == expected_wednesday
    assert second.preferred_artist == "Hossam"
    assert second.appointment_type == "studio_visit"
    assert second.tattoo_project_type == "new tattoo"
    assert second.missing_information == ["reference images"]

    third_message = "I said multiple times that I need it in calligraphy style."
    complete_history = [
        *history,
        {"role": "user", "content": second_message},
        {
            "role": "assistant",
            "content": "Please share a reference or inspiration image.",
        },
    ]
    final_inquiry = TattooInquiryInput(
        current_message=third_message,
        message_source="outlook",
        new_image_urls=["https://example.com/reference.png"],
        existing_db_state={
            "lead": {"name": "Fahim Sarker", "source": "outlook"},
            "intake": second.model_dump(),
        },
        recent_chat_history=complete_history,
    )
    final = extractor.extract(
        current_message=final_inquiry.current_message,
        style_tags=["unknown"],
        new_image_urls=final_inquiry.new_image_urls,
        existing_db_state=final_inquiry.existing_db_state,
        recent_chat_history=final_inquiry.recent_chat_history,
    )

    assert final.tattoo_idea == second.tattoo_idea
    assert final.style_tags == ["calligraphy"]
    assert final.missing_information == []

    routed = TattooRouter(
        llm=cast(ChatOpenAI, FailingExtractionLLM()),
    ).route(
        extracted=final,
        current_message=final_inquiry.current_message,
        recent_chat_history=final_inquiry.recent_chat_history,
        existing_db_state=final_inquiry.existing_db_state,
        message_source="outlook",
    )
    assert routed.risk_level == "high"
    assert routed.auto_reply_allowed is False
    assert routed.draft_reply.startswith("Dear Fahim,")
    assert "sorry for asking again" in routed.draft_reply
    assert "calligraphy" not in routed.draft_reply.casefold()
    assert "That completes the details we need." in routed.draft_reply
    assert "contact you with pricing and the next steps" in routed.draft_reply
