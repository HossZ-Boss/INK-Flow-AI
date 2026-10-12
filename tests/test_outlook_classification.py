"""Tests for the Outlook tattoo-intake auto-reply gate."""

from ai_brain.outlook_classification import OutlookInquiryClassifier
from ai_brain.schemas import TattooInquiryInput


def _classify(
    message: str,
    *,
    email: str = "client@example.com",
    history: list[dict[str, str]] | None = None,
    intake: dict[str, object] | None = None,
) -> bool:
    """Return the classifier decision for one Outlook email."""
    inquiry = TattooInquiryInput(
        current_message=message,
        message_source="outlook",
        existing_db_state={
            "lead": {"email": email, "source": "outlook"},
            "intake": intake or {"source": "outlook"},
        },
        recent_chat_history=history or [],
    )
    return OutlookInquiryClassifier().classify(inquiry).is_tattoo_inquiry


def test_direct_tattoo_inquiry_is_eligible() -> None:
    """A genuine client tattoo request passes the Outlook gate."""
    assert _classify(
        "I want a fine-line rose tattoo on my wrist. What will it cost?"
    )


def test_implied_tattoo_request_is_eligible() -> None:
    """Strong design, style, and placement evidence works without the keyword."""
    assert _classify("I want a minimal lotus design on my inner wrist.")


def test_artist_booking_request_is_eligible() -> None:
    """A named-artist booking request is valid without saying tattoo."""
    assert _classify("Could I book an appointment with Hoss next Thursday?")


def test_no_reply_sender_is_never_eligible() -> None:
    """Tattoo wording cannot override an automated sender address."""
    assert not _classify(
        "New tattoo offers are available this week.",
        email="no-reply@marketing.example.com",
    )


def test_unrelated_human_email_is_not_eligible() -> None:
    """Normal Outlook email is rejected when it is not tattoo intake."""
    assert not _classify(
        "Please review my job application and attached resume."
    )


def test_automated_notification_is_not_eligible() -> None:
    """Automated messages are rejected even without a no-reply address."""
    assert not _classify(
        "Automatic reply: I am out of office until next Monday."
    )


def test_complaint_or_medical_message_requires_manual_handling() -> None:
    """Tattoo-related sensitive mail must not be sent automatically."""
    assert not _classify(
        "My tattoo looks infected and I need medical advice."
    )


def test_short_answer_in_tattoo_thread_is_eligible() -> None:
    """A short intake answer is accepted only with established thread context."""
    history = [
        {
            "role": "user",
            "content": "I want a flower tattoo on my wrist.",
        },
        {
            "role": "assistant",
            "content": (
                "Dear Sam, would you like colour or black and grey? "
                "Kind regards, Tattoo Hysteria"
            ),
        },
    ]

    assert _classify(
        "Black and grey, please.",
        history=history,
        intake={"tattoo_idea": "Flower", "placement": "wrist"},
    )


def test_studio_name_alone_is_not_a_tattoo_request() -> None:
    """Addressing "Tattoo Hysteria" in a sales email is not intake."""
    assert not _classify(
        "Hoi team Tattoo Hysteria, Ik hoop dat je het niet erg vindt dat ik "
        "nog één keer van me laat horen: Waarschijnlijk staat er nu even "
        "niets op de planning bij Tattoo Hysteria. Mocht er later dit jaar "
        "behoefte zijn aan een gezellige pizza-avond op eigen locatie, "
        "zonder dat het een vermogen kost, dan weet je ons te vinden. "
        "P.S. Weet jij wie hier binnen jullie team normaal over gaat? "
        "Met smakelijke groet, Alphons de Vries | Mede-eigenaar Holy Moly "
        "Pizza. Als dit niet past, laat het gerust weten, dan hou ik het "
        "hierbij.",
        email="alphonsdevries.av@enjoyholymolypizza.com",
    )
    assert not _classify(
        "Hello Tattoo Hysteria team, we offer office cleaning in Amsterdam."
    )


def test_dutch_tattoo_requests_are_eligible() -> None:
    """Dutch requests still pass once the studio name is ignored."""
    assert _classify("Hoi Tattoo Hysteria, ik wil graag een tattoo laten zetten.")
    assert _classify("Hallo, hoeveel kost een kleine tattoo?")
    assert _classify("Ik zou graag een afspraak maken voor een tatoeage.")


def test_courtesy_only_email_does_not_trigger_another_reply() -> None:
    """A thank-you email must not create an unnecessary response loop."""
    assert not _classify("Thank you!")


def test_proceed_confirmation_continues_established_tattoo_thread() -> None:
    """A proceed instruction reaches review instead of being suppressed."""
    history = [
        {
            "role": "user",
            "content": "I want a 3 cm tattoo on my tongue.",
        },
        {
            "role": "assistant",
            "content": "The studio must review this specialised request.",
        },
    ]

    assert _classify("Okay then proceed", history=history)


def test_status_question_continues_established_tattoo_thread() -> None:
    """A tattoo-thread status request reaches manual review safely."""
    history = [
        {
            "role": "user",
            "content": "I want a 3 cm tattoo on my tongue.",
        },
        {
            "role": "assistant",
            "content": "The studio team needs to review this request.",
        },
    ]

    assert _classify("Is there any update?", history=history)
