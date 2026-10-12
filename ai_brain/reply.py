"""Context-aware client reply composition for natural studio chat."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date as calendar_date

from .review_policy import (
    SPECIALISED_PLACEMENT_REASON,
    STATUS_UPDATE_REASON,
    is_proceed_confirmation,
    specialised_placement_requires_review,
)
from .schemas import (
    ARTIST_PREFERENCE_OPTIONS,
    MISSING_INFORMATION_OPTIONS,
    STYLE_TAG_OPTIONS,
    Message,
    RiskLevel,
    TattooExtractionDraft,
)

_GREETING_PATTERN = re.compile(
    r"^(?:hi|hello|hey|good morning|good afternoon|good evening)"
    r"(?:\s+(?:team|inkflow|ink flow))*[!.\s]*$",
    flags=re.IGNORECASE,
)
_CORRECTION_TERMS = (
    "i said",
    "already said",
    "just said",
    "actually",
    "i meant",
    "i told you",
    "mentioned earlier",
    "as i mentioned",
)
_REPEATED_DETAIL_TERMS = (
    "i said",
    "already said",
    "just said",
    "i told you",
    "mentioned earlier",
    "as i mentioned",
)
_CONFIRMATION_PATTERN = re.compile(
    r"\b(?:yes|correct|confirmed|all good|looks good|that's right|"
    r"that is right|details are right)\b",
    flags=re.IGNORECASE,
)
_CHANGE_PATTERN = re.compile(
    r"\b(?:change|make it|instead|different|update)\b",
    flags=re.IGNORECASE,
)
_SCHEDULE_TERMS = (
    "today",
    "tomorrow",
    "tonight",
    "preferred date",
    "preferred time",
    "availability",
)
_CLOCK_TIME_PATTERN = re.compile(
    r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|\b\d{1,2}:\d{2}\b",
    flags=re.IGNORECASE,
)
_PRICING_PATTERN = re.compile(
    r"\b(?:price|pricing|cost|quote|estimate|how much|budget)\b"
)
_POSSIBLE_PATTERN = re.compile(
    r"\b(?:is|would)\b.{0,30}\bpossible\b",
    flags=re.IGNORECASE,
)
ARTIST_ROSTER_PATTERN = re.compile(
    r"\b(?:who\s+are\s+(?:the|your)\s+artists|"
    r"artists?\s+in\s+(?:the|your)\s+(?:shop|studio)|"
    r"guide\s+me\s+about\s+(?:them|the\s+artists)|"
    r"(?:tell|let)\s+me\s+(?:know\s+)?about\s+(?:the|your)\s+artists|"
    r"(?:describe|introduce)\s+(?:about\s+)?(?:the|your)\s+artists|"
    r"(?:know|details?|information|info)\s+about\s+(?:the|your)\s+artists)\b",
    flags=re.IGNORECASE,
)
ARTIST_CHOICE_HELP_PATTERN = re.compile(
    r"\b(?:help\s+me\s+(?:to\s+)?(?:choose|pick|select|find|decide)|"
    r"(?:choose|pick|find)\s+(?:a|an|the)\s+(?:suitable|right|good|best)|"
    r"suitable\s+artist|right\s+artist|best\s+artist|"
    r"recommend|suggest|which\s+artist|who\s+(?:would|should|is\s+best))\b",
    flags=re.IGNORECASE,
)
_ARTIST_GUIDANCE_PATTERN = re.compile(
    r"\b(?:recommend|suggest|best|better|which\s+artist|who\s+would|"
    r"portfolio|speciali[sz]|tell\s+me\s+about)\b|"
    + ARTIST_ROSTER_PATTERN.pattern
    + "|"
    + ARTIST_CHOICE_HELP_PATTERN.pattern,
    flags=re.IGNORECASE,
)
_SUMMARY_CONFIRMATION_MARKER = "does that sound right"
_DAY_PATTERN = re.compile(
    r"\b(?:today|tomorrow|tonight|monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday)\b",
    flags=re.IGNORECASE,
)
_STUDIO_AVAILABILITY_PATTERN = re.compile(
    r"\b(?:what|which)\s+(?:days?|dates?|times?|slots?)\b.{0,40}"
    r"\bavailable\b|\bwhen\b.{0,25}\b(?:available|open)\b",
    flags=re.IGNORECASE,
)
_PROJECT_TYPE_PATTERN = re.compile(
    r"\b(?:new\s+tattoo|cover[- ]?up|continuation|touch[- ]?up)\b",
    flags=re.IGNORECASE,
)
_MANUAL_REVIEW_PATTERNS = (
    _PRICING_PATTERN,
    re.compile(r"\b(?:book|booking|booked|deposit)\b"),
    re.compile(
        r"\b(?:cancel|cancellation|reschedule|rescheduling|refund)\b"
    ),
    re.compile(
        r"\b(?:complaint|bad experience|unhappy|dissatisfied)\b"
    ),
    re.compile(r"\b(?:complex design advice|design advice|medical advice)\b"),
    re.compile(r"\b(?:specific|preferred|requested)\s+artist\b"),
    re.compile(
        r"\b(?:underage|minor|pregnant|pregnancy|infection|allergy|"
        r"medical condition)\b"
    ),
)
_APPOINTMENT_TYPE_QUESTION = (
    "Would you prefer an online appointment or a studio visit? "
    "Please reply with online or studio visit."
)
_CLIENT_STYLE_OPTIONS = tuple(
    style for style in STYLE_TAG_OPTIONS if style != "unknown"
)
_STYLE_QUESTION = (
    "What tattoo style would you prefer? Please choose one or more from: "
    + ", ".join(_CLIENT_STYLE_OPTIONS)
    + "."
)
_MISSING_QUESTIONS = {
    "client full name": "What is your full name?",
    "tattoo idea": "What is your tattoo idea or background story?",
    "size in cm": "What size would you prefer in centimetres?",
    "placement": "Where on your body would you like the tattoo?",
    "color preference": "Would you like colour or black and grey?",
    "tattoo style": _STYLE_QUESTION,
    "reference images": (
        "Could you share any reference or inspiration images?"
    ),
    "preferred artist": (
        "Do you have a preferred artist? Please choose "
        + ", ".join(ARTIST_PREFERENCE_OPTIONS[:-1])
        + f", {ARTIST_PREFERENCE_OPTIONS[-1]}, or say no preference."
    ),
    "appointment type": _APPOINTMENT_TYPE_QUESTION,
    "preferred dates or availability": (
        "What are your preferred dates or general availability?"
    ),
    "tattoo project type": (
        "Is this a new tattoo, cover-up, continuation, or touch-up?"
    ),
    "preferred date": "What date works best for you?",
    "preferred time": "What time works best for you?",
}
_MISSING_EMAIL_REQUESTS = dict(_MISSING_QUESTIONS)
_QUESTION_MARKERS = {
    "client full name": ("full name", "your name"),
    "tattoo idea": (
        "tattoo idea",
        "background story",
    ),
    "size in cm": ("size", "centimetres", "centimeters"),
    "placement": ("where on your body", "placement"),
    "color preference": ("black and grey", "colour", "color"),
    "tattoo style": ("tattoo style",),
    "reference images": ("reference", "inspiration image"),
    "preferred artist": ("preferred artist", "choose hoss", "go ahead with"),
    "appointment type": (
        "appointment type",
        "studio visit",
        "online appointment",
    ),
    "preferred dates or availability": ("preferred dates", "availability"),
    "tattoo project type": (
        "new tattoo",
        "cover-up",
        "continuation",
        "touch-up",
    ),
    "preferred date": ("date", "day", "when"),
    "preferred time": ("time", " am", " pm"),
}


def requires_manual_review(
    current_message: str,
    recent_chat_history: Sequence[Message] | None = None,
) -> bool:
    """Return whether current or recent client intent requires staff review."""
    user_messages = [
        message.content
        for message in (recent_chat_history or [])
        if message.role == "user"
    ]
    combined = " ".join([*user_messages, current_message]).casefold()
    return any(pattern.search(combined) for pattern in _MANUAL_REVIEW_PATTERNS)


def pricing_was_requested(
    current_message: str,
    recent_chat_history: Sequence[Message] | None = None,
) -> bool:
    """Return whether the client asked for price in this conversation."""
    user_messages = [
        message.content
        for message in (recent_chat_history or [])
        if message.role == "user"
    ]
    combined = " ".join([*user_messages, current_message]).casefold()
    return bool(_PRICING_PATTERN.search(combined))


def date_confirmation_question(iso_date: str) -> str:
    """Ask whether a bare weekday means this exact calendar date."""
    day = calendar_date.fromisoformat(iso_date)
    return f"Just to confirm, do you mean {day:%A}, {day.day} {day:%B}?"


class ConversationReplyComposer:
    """Create concise replies that feel like an ongoing human conversation."""

    def compose(
        self,
        extracted: TattooExtractionDraft,
        current_message: str,
        recent_chat_history: list[Message] | None,
        risk_level: RiskLevel,
        suggested_artist: str = "Unclear",
        suggested_artist_details: str = "",
        artist_directory: str = "",
    ) -> str:
        """Acknowledge the latest turn and ask only the next useful question."""
        history = recent_chat_history or []
        if extracted.conversation_status == "closed":
            return self.compose_closed(message_source="whatsapp")
        pricing_requested = (
            extracted.pricing_requested
            or pricing_was_requested(current_message, history)
        )
        pricing_acknowledgement_needed = (
            pricing_requested
            and self._pricing_acknowledgement_needed(
                extracted,
                current_message,
                history,
            )
        )
        manual_review = risk_level == "high" and requires_manual_review(
            current_message,
            history,
        )
        if self._is_greeting_only(current_message) and not manual_review:
            return self._greeting_reply(history)

        acknowledgement = self._acknowledgement(
            current_message=current_message,
            history=history,
        )
        complexity_notice = self._complexity_notice(extracted)
        artist_guidance = self._artist_guidance(
            extracted=extracted,
            current_message=current_message,
            suggested_artist=suggested_artist,
            suggested_artist_details=suggested_artist_details,
            artist_directory=artist_directory,
            history=history,
        )
        size_guidance = self._size_guidance(extracted, history)
        if manual_review:
            review_subject = (
                "the details and pricing" if pricing_requested else "this"
            )
            reply_parts = [acknowledgement]
            if complexity_notice:
                reply_parts.append(complexity_notice)
            if artist_guidance:
                reply_parts.append(artist_guidance)
            if size_guidance:
                reply_parts.append(size_guidance)
            reply_parts.append(
                "I'll have the studio team review "
                f"{review_subject} and get back to you."
            )
            if specialised_placement_requires_review(extracted.placement):
                reply_parts.append(self._specialised_placement_notice())
            reply = self._join_reply_parts(reply_parts)
            return self._avoid_exact_repeat(reply, history)

        questions = self._select_questions(
            missing_information=self._question_information_for_turn(
                list(extracted.missing_information),
                current_message,
                suggested_artist,
            ),
            history=history,
        )
        questions = self._prepend_size_confirmation(
            extracted=extracted,
            questions=questions,
            history=history,
        )
        questions = self._personalize_artist_question(
            questions,
            extracted,
            suggested_artist,
            history,
        )
        questions = self._confirm_weekday_date(questions, extracted)
        if questions:
            reply_parts = [acknowledgement]
            specialised_notice = self._specialised_collection_notice(
                extracted,
                history,
            )
            if specialised_notice:
                reply_parts.append(specialised_notice)
            if complexity_notice:
                reply_parts.append(complexity_notice)
            if artist_guidance:
                reply_parts.append(artist_guidance)
            if size_guidance:
                reply_parts.append(size_guidance)
            if pricing_acknowledgement_needed:
                reply_parts.append(
                    self._pricing_message(extracted, history)
                )
            reply = self._join_reply_parts([*reply_parts, *questions])
            return self._avoid_exact_repeat(reply, history)

        if risk_level == "high":
            reply_parts = [acknowledgement]
            if complexity_notice:
                reply_parts.append(complexity_notice)
            reply_parts.append(
                "I'll have the studio team review this and get back to you."
            )
            if specialised_placement_requires_review(extracted.placement):
                reply_parts.append(self._specialised_placement_notice())
            reply = self._join_reply_parts(reply_parts)
            return self._avoid_exact_repeat(reply, history)

        reply_parts = [acknowledgement]
        if complexity_notice:
            reply_parts.append(complexity_notice)
        if artist_guidance:
            reply_parts.append(artist_guidance)
        if size_guidance:
            reply_parts.append(size_guidance)
        reply_parts.append(
            "I've got the main details now. I'll pass this to the team for "
            "a quick review."
        )
        reply = self._join_reply_parts(reply_parts)
        return self._avoid_exact_repeat(reply, history)

    def compose_validation(
        self,
        extracted: TattooExtractionDraft,
        current_message: str,
        recent_chat_history: list[Message] | None,
        risk_level: RiskLevel,
        suggested_artist: str = "Unclear",
        suggested_artist_details: str = "",
        artist_directory: str = "",
    ) -> str:
        """Summarize extracted facts once, unless already confirmed."""
        history = recent_chat_history or []
        if extracted.conversation_status == "closed":
            return self.compose_closed(message_source="whatsapp")
        pricing_requested = (
            extracted.pricing_requested
            or pricing_was_requested(current_message, history)
        )
        pricing_acknowledgement_needed = (
            pricing_requested
            and self._pricing_acknowledgement_needed(
                extracted,
                current_message,
                history,
            )
        )
        if risk_level == "high":
            return self.compose(
                extracted=extracted,
                current_message=current_message,
                recent_chat_history=history,
                risk_level=risk_level,
                suggested_artist=suggested_artist,
                suggested_artist_details=suggested_artist_details,
                artist_directory=artist_directory,
            )
        if self._is_greeting_only(current_message):
            return self.compose(
                extracted=extracted,
                current_message=current_message,
                recent_chat_history=history,
                risk_level=risk_level,
                suggested_artist=suggested_artist,
                suggested_artist_details=suggested_artist_details,
                artist_directory=artist_directory,
            )
        first_reply = not any(
            message.role == "assistant" for message in history
        )
        if (
            (len(extracted.missing_information) != 1 and not first_reply)
            or self._details_already_confirmed(current_message, history)
            or self._summary_already_shown(history)
            or _ARTIST_GUIDANCE_PATTERN.search(current_message)
        ):
            return self.compose(
                extracted=extracted,
                current_message=current_message,
                recent_chat_history=history,
                risk_level=risk_level,
                suggested_artist=suggested_artist,
                suggested_artist_details=suggested_artist_details,
                artist_directory=artist_directory,
            )

        summary = self._natural_summary(extracted)
        if not summary:
            return self.compose(
                extracted=extracted,
                current_message=current_message,
                recent_chat_history=history,
                risk_level=risk_level,
                suggested_artist=suggested_artist,
                suggested_artist_details=suggested_artist_details,
                artist_directory=artist_directory,
            )

        questions = self._select_questions(
            missing_information=self._question_information_for_turn(
                list(extracted.missing_information),
                current_message,
                suggested_artist,
            ),
            history=history,
        )[:1]
        questions = self._personalize_artist_question(
            questions,
            extracted,
            suggested_artist,
            history,
        )
        questions = self._confirm_weekday_date(questions, extracted)
        reply_parts = []
        if _POSSIBLE_PATTERN.search(current_message):
            reply_parts.append("Yes, we can help with that.")
        reply_parts.append(summary)
        specialised_notice = self._specialised_collection_notice(
            extracted,
            history,
        )
        if specialised_notice:
            reply_parts.append(specialised_notice)
        complexity_notice = self._complexity_notice(extracted)
        if complexity_notice:
            reply_parts.append(complexity_notice)
        artist_guidance = self._artist_guidance(
            extracted=extracted,
            current_message=current_message,
            suggested_artist=suggested_artist,
            suggested_artist_details=suggested_artist_details,
            artist_directory=artist_directory,
            history=history,
        )
        if artist_guidance:
            reply_parts.append(artist_guidance)
        size_guidance = self._size_guidance(extracted, history)
        if size_guidance:
            reply_parts.append(size_guidance)
        size_confirmation = self._approximate_size_question(
            extracted,
            history,
        )
        reply_parts.append(
            size_confirmation
            or "Does that sound right, or would you like to change anything?"
        )
        if pricing_acknowledgement_needed:
            reply_parts.append(self._pricing_message(extracted, history))
        reply_parts.extend(questions)
        return self._avoid_exact_repeat(
            self._join_reply_parts(reply_parts),
            history,
        )

    def compose_outlook_email(
        self,
        extracted: TattooExtractionDraft,
        existing_db_state: Mapping[str, object] | None = None,
        current_message: str = "",
        recent_chat_history: Sequence[Message] | None = None,
        suggested_artist: str = "Unclear",
        suggested_artist_details: str = "",
        artist_directory: str = "",
    ) -> str:
        """Create a natural email that asks only the next useful questions."""
        history = recent_chat_history or []
        if extracted.conversation_status == "closed":
            return self.compose_closed(
                message_source="outlook",
                existing_db_state=existing_db_state,
                client_name=extracted.client_name,
            )
        is_follow_up = self._is_outlook_follow_up(
            history,
            existing_db_state,
        )
        pricing_requested = (
            extracted.pricing_requested
            or pricing_was_requested(current_message, history)
        )
        pricing_acknowledgement_needed = (
            pricing_requested
            and self._pricing_acknowledgement_needed(
                extracted,
                current_message,
                history,
            )
        )
        missing_information = list(extracted.missing_information)
        question_information = self._question_information_for_turn(
            missing_information,
            current_message,
            suggested_artist,
        )
        correction = any(
            term in current_message.casefold() for term in _CORRECTION_TERMS
        )
        repeated_detail = any(
            term in current_message.casefold()
            for term in _REPEATED_DETAIL_TERMS
        )
        last_requirement = len(missing_information) == 1
        sections = [
            self._outlook_salutation(
                existing_db_state,
                extracted.client_name,
            ),
            self._outlook_opening(
                current_message=current_message,
                is_follow_up=is_follow_up,
                correction=correction,
                repeated_detail=repeated_detail,
            ),
        ]

        if last_requirement or (not is_follow_up and not missing_information):
            sections.append(
                self._outlook_confirmation_summary(
                    extracted,
                    suggested_artist,
                )
            )
        elif missing_information:
            if not is_follow_up:
                request_summary = self._human_request_summary(extracted)
                if request_summary:
                    sections.append(request_summary)

            complexity_notice = self._complexity_notice(extracted)
            if complexity_notice and not self._assistant_history_contains(
                complexity_notice,
                history,
            ):
                sections.append(complexity_notice)

        artist_guidance = self._artist_guidance(
            extracted=extracted,
            current_message=current_message,
            suggested_artist=suggested_artist,
            suggested_artist_details=suggested_artist_details,
            artist_directory=artist_directory,
            history=history,
        )
        if artist_guidance:
            sections.append(artist_guidance)

        if missing_information:
            specialised_notice = self._specialised_collection_notice(
                extracted,
                history,
            )
            if specialised_notice:
                sections.append(specialised_notice)
        elif specialised_placement_requires_review(extracted.placement):
            sections.append(self._specialised_placement_notice())

        size_guidance = self._size_guidance(extracted, history)
        if size_guidance:
            sections.append(size_guidance)

        if pricing_acknowledgement_needed:
            if self._pricing_previously_acknowledged(history):
                sections.append(
                    "These updated design details will help the artist "
                    "prepare your custom estimate."
                )
            elif "reference images" in missing_information:
                sections.append(
                    "Once we have the remaining design details and any "
                    "reference images, the artist can provide a custom "
                    "estimate."
                )
            else:
                sections.append(
                    "The artist will review the design and provide a custom "
                    "estimate."
                )

        if missing_information:
            if _STUDIO_AVAILABILITY_PATTERN.search(current_message):
                sections.append(
                    "I don't have the live studio calendar in this email "
                    "thread. If you send one or two dates and time windows "
                    "that suit you, the team can check them for you."
                )
                sections.append(
                    "Once we have your preferred options, we'll confirm what "
                    "is available."
                )
                sections.append("Kind regards,\nTattoo Hysteria")
                return "\n\n".join(sections)

            questions = self._select_questions(
                missing_information=question_information,
                history=list(history),
            )
            questions = self._prepend_size_confirmation(
                extracted=extracted,
                questions=questions,
                history=history,
            )
            questions = self._personalize_artist_question(
                questions,
                extracted,
                suggested_artist,
                history,
            )
            questions = self._confirm_weekday_date(questions, extracted)
            if questions:
                if last_requirement and len(questions) == 1:
                    sections.append(f"One last detail: {questions[0]}")
                else:
                    intro = (
                        "To help us move this forward, could you share a "
                        "little more detail?"
                        if not is_follow_up
                        else "Could you help me with the next detail?"
                    )
                    sections.append(" ".join([intro, *questions]))
            if not last_requirement:
                sections.append(
                    "Once we have those details, we'll guide you through the "
                    "next step."
                )
        else:
            sections.append(
                ("That completes the details we need. " if is_follow_up else "")
                + "Our studio team will review everything and "
                + (
                    "contact you with pricing and the next steps."
                    if pricing_requested
                    else "contact you with the next steps."
                )
            )

        sections.append("Kind regards,\nTattoo Hysteria")
        return "\n\n".join(sections)

    def compose_closed(
        self,
        message_source: str,
        existing_db_state: Mapping[str, object] | None = None,
        client_name: str = "",
    ) -> str:
        """Acknowledge withdrawal without continuing intake collection."""
        acknowledgement = (
            "Understood. I've closed your tattoo inquiry, and we won't send "
            "any more intake questions. If you change your mind later, just "
            "get in touch and we'll be happy to help."
        )
        if message_source != "outlook":
            return acknowledgement
        salutation = self._outlook_salutation(existing_db_state, client_name)
        return "\n\n".join(
            [salutation, acknowledgement, "Kind regards,\nTattoo Hysteria"]
        )

    def compose_staff_review(
        self,
        extracted: TattooExtractionDraft,
        current_message: str,
        message_source: str,
        existing_db_state: Mapping[str, object] | None,
        review_reasons: Sequence[str],
    ) -> str:
        """Pause intake and explain a real staff escalation safely."""
        reasons = set(review_reasons)
        sections: list[str] = []
        if message_source == "outlook":
            sections.append(
                self._outlook_salutation(
                    existing_db_state,
                    extracted.client_name,
                )
            )

        if STATUS_UPDATE_REASON in reasons:
            sections.append(
                "I can’t confirm a live update from the information "
                "available in this conversation. I’ve flagged your message "
                "for a studio team member to check and respond with the "
                "current status."
            )
            if SPECIALISED_PLACEMENT_REASON in reasons:
                sections.append(self._specialised_placement_notice())
        else:
            acknowledgement = (
                "Thanks for confirming."
                if is_proceed_confirmation(current_message)
                else "Thank you for sharing the details."
            )
            sections.append(acknowledgement)
            summary = self._human_request_summary(extracted)
            if summary:
                sections.append(summary)
            sections.append(
                "I’ve flagged your request for review by the studio team."
            )
            if SPECIALISED_PLACEMENT_REASON in reasons:
                sections.append(self._specialised_placement_notice())

        if message_source == "outlook":
            sections.append("Kind regards,\nTattoo Hysteria")
        return "\n\n".join(sections)

    def _specialised_collection_notice(
        self,
        extracted: TattooExtractionDraft,
        history: Sequence[Message],
    ) -> str:
        """Mention once that a specialised placement needs studio approval."""
        if not specialised_placement_requires_review(extracted.placement):
            return ""
        already_mentioned = any(
            message.role == "assistant"
            and "tongue" in message.content.casefold()
            and re.search(r"\b(?:approv|review)", message.content.casefold())
            for message in history
        )
        if already_mentioned:
            return ""
        return (
            "Just so you know, tongue tattoos need studio approval before an "
            "artist, price, or booking can be confirmed, so I'll collect your "
            "details for the team to review."
        )

    def _specialised_placement_notice(self) -> str:
        """Return the fixed, non-medical tongue-placement review notice."""
        return (
            "Tongue tattooing is a specialised request. Before discussing "
            "an artist, price, or booking, the studio team must confirm "
            "whether this service is offered and whether the request can "
            "proceed. Nothing has been approved or booked yet."
        )

    def _outlook_opening(
        self,
        current_message: str,
        is_follow_up: bool,
        correction: bool,
        repeated_detail: bool,
    ) -> str:
        """Acknowledge the latest email without repeating a stock opening."""
        if not is_follow_up:
            if _POSSIBLE_PATTERN.search(current_message):
                return (
                    "Yes, we can help with that - thank you for reaching out "
                    "to Tattoo Hysteria."
                )
            return (
                "Thank you for reaching out to Tattoo Hysteria - we'd be "
                "happy to help with your tattoo request."
            )
        if repeated_detail:
            return (
                "You're right - sorry for asking again. I have kept the "
                "details you already provided."
            )
        if correction:
            return "Thanks for clarifying - I've updated that detail."
        if _ARTIST_GUIDANCE_PATTERN.search(current_message):
            return "That's a good question."
        if _STUDIO_AVAILABILITY_PATTERN.search(current_message):
            return "That's a good question."
        normalized = current_message.casefold()
        if re.search(r"\byes\b.{0,30}\b\d", normalized):
            return "Perfect, thanks for confirming the size."
        if re.search(r"\b(?:studio\s+visit|online\s+appointment)\b", normalized):
            return "Thanks, I've noted your appointment preference."
        if _PROJECT_TYPE_PATTERN.search(current_message):
            return "Thanks, I've noted the tattoo type."
        if any(term in normalized for term in _SCHEDULE_TERMS) or (
            _CLOCK_TIME_PATTERN.search(normalized)
        ):
            return "Thanks, I've added your preferred timing."
        return "Thanks, I've added those details."

    def _assistant_history_contains(
        self,
        text: str,
        history: Sequence[Message],
    ) -> bool:
        """Return whether assistant history already contains the same prose."""
        needle = " ".join(text.split()).casefold()
        return any(
            message.role == "assistant"
            and needle in " ".join(message.content.split()).casefold()
            for message in history
        )

    def _artist_guidance(
        self,
        extracted: TattooExtractionDraft,
        current_message: str,
        suggested_artist: str,
        suggested_artist_details: str,
        artist_directory: str,
        history: Sequence[Message],
    ) -> str:
        """Answer artist questions from configured portfolio information."""
        if extracted.client_intent == "size_guidance":
            return ""
        if not _ARTIST_GUIDANCE_PATTERN.search(current_message):
            return ""
        if specialised_placement_requires_review(extracted.placement):
            return (
                "For this placement, the studio team will choose a suitable "
                "artist once the request is approved."
            )
        offers_choice_help = bool(
            ARTIST_CHOICE_HELP_PATTERN.search(current_message)
        ) and suggested_artist in ARTIST_PREFERENCE_OPTIONS
        if ARTIST_ROSTER_PATTERN.search(current_message) and artist_directory:
            if not offers_choice_help:
                return artist_directory
            known_styles = [
                style
                for style in extracted.style_tags
                if style not in {"unknown", "black-and-grey"}
            ][:2]
            style_phrase = (
                f" for your {' and '.join(known_styles)} piece"
                if known_styles
                else " for your piece"
            )
            return (
                f"{artist_directory}\n\n"
                f"{suggested_artist} would be the best match{style_phrase}."
            )
        if suggested_artist == "Unclear":
            return (
                "I don't have enough of a portfolio match to recommend one "
                "artist confidently yet, so the studio team will compare the "
                "design with the artists' work."
            )

        known_styles = [
            style for style in extracted.style_tags if style != "unknown"
        ]
        style_phrase = (
            f" for your {' and '.join(known_styles)} request"
            if known_styles
            else ""
        )
        recommendation = (
            f"Based on the portfolio match{style_phrase}, I'd recommend "
            f"{suggested_artist}."
        )
        if suggested_artist_details:
            recommendation = f"{recommendation} {suggested_artist_details}"
        if self._assistant_history_contains(recommendation, history):
            return (
                f"{suggested_artist} remains my recommendation based on the "
                "style and design details you've shared."
            )
        return recommendation

    def _size_guidance(
        self,
        extracted: TattooExtractionDraft,
        history: Sequence[Message],
    ) -> str:
        """Reassure an uncertain client without delegating size selection."""
        if (
            extracted.client_intent != "size_guidance"
            and extracted.size_description != "not sure"
        ):
            return ""
        if extracted.size_status == "approximate":
            return ""
        guidance = "No problem if you do not know the exact size yet."
        if self._assistant_history_contains(guidance, history):
            return ""
        return guidance

    def _outlook_confirmation_summary(
        self,
        extracted: TattooExtractionDraft,
        suggested_artist: str,
    ) -> str:
        """Summarize completed fields before the final intake question."""
        details: list[str] = []
        idea = self._idea_fragment(extracted.tattoo_idea)
        style_fragment = " and ".join(
            style
            for style in extracted.style_tags
            if style not in {"unknown", "black-and-grey"}
            and style.casefold() not in idea.casefold()
        )
        design_parts = [
            value
            for value in (
                extracted.size_estimate_cm,
                (
                    "black-and-grey"
                    if extracted.color_preference == "black-and-grey"
                    else "colour"
                    if extracted.color_preference == "color"
                    else ""
                ),
                style_fragment,
                idea,
            )
            if value
        ]
        if design_parts or extracted.placement:
            design = " ".join(design_parts).strip() or "tattoo"
            if "tattoo" not in design.casefold():
                design += " tattoo"
            if extracted.placement:
                design += f" on your {extracted.placement}"
            details.append(design)
        if extracted.appointment_type:
            details.append(
                extracted.appointment_type.replace("_", " ")
            )
        if extracted.preferred_artist not in {"", "No preference"}:
            details.append(
                f"preferred artist: {extracted.preferred_artist}"
            )
        elif suggested_artist != "Unclear":
            details.append(f"recommended artist: {suggested_artist}")

        schedule = self._confirmation_schedule(extracted)
        if schedule:
            details.append(f"preferred schedule: {schedule}")
        if extracted.tattoo_project_type:
            details.append(extracted.tattoo_project_type)
        if "reference images" not in extracted.missing_information:
            details.append("reference image received")

        if not details:
            return "We're nearly finished with your request."
        return "To confirm what I have so far: " + "; ".join(details) + "."

    def _confirmation_schedule(
        self,
        extracted: TattooExtractionDraft,
    ) -> str:
        """Format saved date and time for a client-facing confirmation."""
        if extracted.date:
            try:
                parsed = calendar_date.fromisoformat(extracted.date)
                date_text = f"{parsed.day} {parsed.strftime('%B %Y')}"
            except ValueError:
                date_text = extracted.date
            if extracted.time:
                return f"{date_text} at {extracted.time}"
            return date_text
        return extracted.availability

    def _idea_fragment(self, value: str) -> str:
        """Convert a short extracted title into natural sentence casing."""
        value = value.strip().rstrip(".!;,")
        if not value:
            return ""
        if value[:1].isupper() and value[1:].islower():
            return value[:1].lower() + value[1:]
        return value

    def _human_request_summary(
        self,
        extracted: TattooExtractionDraft,
    ) -> str:
        """Describe known client details as prose rather than an audit log."""
        aggregate_summary = self._aggregate_multi_tattoo_summary(extracted)
        if aggregate_summary:
            return aggregate_summary
        if len(extracted.projects) > 1 and extracted.party_size == 1:
            return self._same_client_projects_summary(extracted)
        if len(extracted.projects) > 1:
            project_designs = [
                self._project_design(project)
                for project in extracted.projects
            ]
            if (
                project_designs[0]
                and len(set(project_designs)) == 1
            ):
                design = project_designs[0]
                plural_design = (
                    design[:-6] + "tattoos"
                    if design.endswith("tattoo")
                    else design + "s"
                )
                recipients = [
                    "one " + self._project_recipient(project)
                    for project in extracted.projects
                ]
                return (
                    f"It sounds like you're planning {len(project_designs)} "
                    f"{plural_design}, {self._natural_join(recipients)}."
                )
            project_phrases = [
                self._project_phrase(project, index)
                for index, project in enumerate(extracted.projects)
            ]
            useful = [phrase for phrase in project_phrases if phrase]
            if useful:
                return "It sounds like you're planning " + self._natural_join(
                    useful
                ) + "."
            return (
                f"It sounds like this request is for {extracted.party_size} "
                "people."
            )

        details: list[str] = []
        if extracted.size_estimate_cm:
            details.append(extracted.size_estimate_cm)
        elif (
            extracted.size_description
            and extracted.size_description != "not sure"
        ):
            details.append(extracted.size_description)
        known_styles = {
            tag for tag in extracted.style_tags if tag != "unknown"
        }
        if (
            extracted.color_preference == "color"
            and "watercolor" not in known_styles
        ):
            details.append("colour")
        elif extracted.color_preference == "black-and-grey":
            details.append("black-and-grey")
        details.extend(
            tag
            for tag in extracted.style_tags
            if tag not in {"unknown", "black-and-grey"}
        )
        if extracted.tattoo_idea:
            details.append(self._idea_fragment(extracted.tattoo_idea))
        if not details and not extracted.placement:
            return ""
        design = " ".join(details).strip() or "tattoo"
        if "tattoo" not in design.casefold():
            design += " tattoo"
        placement = (
            f" on your {extracted.placement}" if extracted.placement else ""
        )
        design_phrase = self._with_indefinite_article(design)
        return f"It sounds like you'd like {design_phrase}{placement}."

    def _aggregate_multi_tattoo_summary(
        self,
        extracted: TattooExtractionDraft,
    ) -> str:
        """Render a flat multi-tattoo concept as natural client prose."""
        match = re.fullmatch(
            r"(?P<count>two|three|four|five|six|seven|eight|nine|ten|\d+)"
            r"(?:\s+(?P<body>finger|toe))?\s+tattoos?:\s*"
            r"(?P<ideas>.+)",
            extracted.tattoo_idea.strip(),
            flags=re.IGNORECASE,
        )
        if match is None:
            return ""
        descriptors: list[str] = [match.group("count").casefold()]
        if extracted.size_estimate_cm:
            descriptors.append(extracted.size_estimate_cm)
        if extracted.color_preference == "black-and-grey":
            descriptors.append("black-and-grey")
        elif extracted.color_preference == "color":
            descriptors.append("colour")
        descriptors.extend(
            style
            for style in extracted.style_tags
            if style not in {"unknown", "black-and-grey"}
        )
        if match.group("body"):
            descriptors.append(match.group("body").casefold())
        descriptors.append("tattoos")
        ideas = re.split(
            r"\s*,\s*(?:and\s+)?|\s+and\s+",
            match.group("ideas"),
        )
        idea_phrases = [
            self._with_indefinite_article(idea.strip())
            for idea in ideas
            if idea.strip()
        ]
        return (
            f"It sounds like you're planning {' '.join(descriptors)}: "
            f"{self._natural_join(idea_phrases)}."
        )

    def _question_information_for_turn(
        self,
        missing_information: list[str],
        current_message: str,
        suggested_artist: str = "Unclear",
    ) -> list[str]:
        """Answer artist guidance before requesting an explicit selection."""
        if not _ARTIST_GUIDANCE_PATTERN.search(current_message):
            return missing_information
        if (
            suggested_artist in ARTIST_PREFERENCE_OPTIONS
            and ARTIST_CHOICE_HELP_PATTERN.search(current_message)
        ):
            return missing_information
        return [
            item for item in missing_information if item != "preferred artist"
        ]

    def _same_client_projects_summary(
        self,
        extracted: TattooExtractionDraft,
    ) -> str:
        """Describe multiple tattoos for one client without inventing people."""
        project_ideas = [
            str(project.tattoo_idea).strip().casefold()
            for project in extracted.projects
            if str(project.tattoo_idea).strip()
        ]
        ideas = [
            self._with_indefinite_article(idea)
            for idea in project_ideas
        ]
        descriptors: list[str] = []
        if extracted.size_estimate_cm:
            descriptors.append(extracted.size_estimate_cm)
        if extracted.color_preference == "black-and-grey":
            descriptors.append("black-and-grey")
        elif extracted.color_preference == "color":
            descriptors.append("colour")
        descriptors.extend(
            style
            for style in extracted.style_tags
            if style not in {"unknown", "black-and-grey"}
        )
        placement = extracted.placement.casefold()
        if "finger" in placement:
            descriptors.append("finger")
        elif "toe" in placement:
            descriptors.append("toe")
        count = self._number_word(len(extracted.projects))
        subject = " ".join([count, *descriptors, "tattoos"])
        if not ideas:
            return f"It sounds like you're planning {subject}."
        return (
            f"It sounds like you're planning {subject}: "
            f"{self._natural_join(ideas)}."
        )

    def _number_word(self, value: int) -> str:
        """Return a readable small number for client-facing prose."""
        words = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
        }
        return words.get(value, str(value))

    def _complexity_notice(
        self,
        extracted: TattooExtractionDraft,
    ) -> str:
        """Return a natural staff-review notice for complex requests."""
        if not extracted.multi_entity_detected:
            return ""
        if "existing tattoo" in extracted.complexity_notes.casefold():
            return (
                "I've noted that this design needs to match an existing "
                "tattoo so the artist can review that detail carefully."
            )
        return (
            "I've kept the separate tattoo details together so the artist "
            "can review them accurately."
        )

    def _prepend_size_confirmation(
        self,
        extracted: TattooExtractionDraft,
        questions: list[str],
        history: Sequence[Message],
    ) -> list[str]:
        """Ask once for confirmation of a conservatively inferred size."""
        size_question = self._approximate_size_question(extracted, history)
        if not size_question:
            return questions
        return [size_question, *questions[:1]]

    def _approximate_size_question(
        self,
        extracted: TattooExtractionDraft,
        history: Sequence[Message],
    ) -> str:
        """Build one guiding question for an inferred centimetre range."""
        estimate = extracted.size_estimate_cm.strip()
        if (
            extracted.size_status != "approximate"
            or not extracted.size_description
            or extracted.size_description == "not sure"
            or not estimate
        ):
            return ""
        if any(
            message.role == "assistant"
            and estimate.casefold() in message.content.casefold()
            for message in history
        ):
            return ""
        natural_estimate = re.sub(r"(?<=\d)-(?=\d)", " to ", estimate)
        if "reference-informed planning estimate" in (
            extracted.complexity_notes.casefold()
        ):
            placement = extracted.placement.strip()
            placement_context = (
                f" for the requested {placement} placement" if placement else ""
            )
            return (
                "Based on your reference image"
                f"{placement_context}, I'd suggest around {natural_estimate}. "
                "Does that sound right, or would you like to adjust it?"
            )
        return (
            "Just to confirm, does an estimated size of around "
            f"{natural_estimate} sound right?"
        )

    def _pricing_acknowledgement_needed(
        self,
        extracted: TattooExtractionDraft,
        current_message: str,
        history: Sequence[Message],
    ) -> bool:
        """Acknowledge price questions and one material design update."""
        if _PRICING_PATTERN.search(current_message.casefold()):
            return True
        if not self._pricing_previously_acknowledged(history):
            return True
        needs_contextual_update = (
            extracted.multi_entity_detected
            or bool(extracted.size_description)
        )
        if not needs_contextual_update:
            return False
        return not any(
            message.role == "assistant"
            and "updated design details" in message.content.casefold()
            for message in history
        )

    def _pricing_previously_acknowledged(
        self,
        history: Sequence[Message],
    ) -> bool:
        """Return whether an earlier assistant response addressed pricing."""
        return any(
            message.role == "assistant"
            and _PRICING_PATTERN.search(message.content.casefold())
            for message in history
        )

    def _pricing_message(
        self,
        extracted: TattooExtractionDraft,
        history: Sequence[Message],
    ) -> str:
        """Return an initial or contextual price acknowledgement."""
        if self._pricing_previously_acknowledged(history) and (
            extracted.multi_entity_detected
            or bool(extracted.size_description)
        ):
            return (
                "These updated design details will help the studio prepare "
                "your custom estimate."
            )
        return (
            "The studio can confirm the price after reviewing the remaining "
            "tattoo details."
        )

    def _project_phrase(self, project: object, index: int) -> str:
        """Create one compact, client-facing phrase for a tattoo project."""
        recipient = self._project_recipient(project)
        design = self._project_design(project)
        if not design:
            return recipient or f"tattoo {index + 1}"
        return " ".join(part for part in (design, recipient) if part)

    def _project_design(self, project: object) -> str:
        """Create the design phrase without its recipient."""
        details: list[str] = []
        size = str(getattr(project, "size_estimate_cm", "")).strip()
        size_description = str(getattr(project, "size_description", "")).strip()
        color = str(getattr(project, "color_preference", "")).strip()
        styles = list(getattr(project, "style_tags", []))
        idea = str(getattr(project, "tattoo_idea", "")).strip()
        if size or size_description:
            details.append(size or size_description)
        if color and not (color == "color" and "watercolor" in styles):
            details.append("colour" if color == "color" else color)
        details.extend(
            str(tag) for tag in styles if tag not in {"unknown", "black-and-grey"}
        )
        if idea:
            details.append(idea)
        if not details:
            return ""
        design = " ".join(details)
        if "tattoo" not in design.casefold():
            design += " tattoo"
        return design

    def _project_recipient(self, project: object) -> str:
        """Create a natural recipient phrase for one project."""
        person_label = str(getattr(project, "person_label", "")).strip()
        if person_label.casefold() == "client":
            return "for you"
        return f"for your {person_label.casefold()}" if person_label else ""

    def _natural_join(self, values: Sequence[str]) -> str:
        """Join short phrases using natural English punctuation."""
        if len(values) == 1:
            return values[0]
        if len(values) == 2:
            return f"{values[0]} and {values[1]}"
        return f"{', '.join(values[:-1])}, and {values[-1]}"

    def _is_outlook_follow_up(
        self,
        history: Sequence[Message],
        existing_db_state: Mapping[str, object] | None,
    ) -> bool:
        """Detect an established email thread without relying on body quotes."""
        if any(message.role == "assistant" for message in history):
            return True

        state = existing_db_state or {}
        records: list[Mapping[str, object]] = [state]
        intake = state.get("intake")
        if isinstance(intake, Mapping):
            records.append(intake)
        return any(
            isinstance(record.get("latest_draft_reply"), str)
            and bool(str(record["latest_draft_reply"]).strip())
            for record in records
        )

    def _outlook_salutation(
        self,
        existing_db_state: Mapping[str, object] | None,
        extracted_client_name: str = "",
    ) -> str:
        """Address an Outlook lead by first name when backend data has it."""
        state = existing_db_state or {}
        lead = state.get("lead")
        name: object = None
        if isinstance(lead, Mapping):
            name = lead.get("name")
        if not name:
            name = state.get("lead_name")
        if not name:
            name = extracted_client_name
        if not isinstance(name, str) or not name.strip():
            return "Hello,"

        first_name = name.strip().split(maxsplit=1)[0]
        safe_name = re.sub(r"[^\w.'’\-]", "", first_name)
        if not safe_name:
            return "Hello,"
        return f"Dear {safe_name},"

    def _outlook_known_details(
        self,
        extracted: TattooExtractionDraft,
    ) -> list[tuple[str, str]]:
        """Return known intake facts as concise professional email rows."""
        details: list[tuple[str, str]] = []
        missing = set(extracted.missing_information)
        if (
            "client full name" not in missing
            and self._is_known_reply_value(extracted.client_name)
        ):
            details.append(("Client name", extracted.client_name))
        if (
            "tattoo idea" not in missing
            and self._is_known_reply_value(extracted.tattoo_idea)
        ):
            details.append(
                ("Tattoo concept", self._email_value(extracted.tattoo_idea))
            )

        idea = self._idea_fragment(extracted.tattoo_idea)
        style_tags = [
            tag
            for tag in extracted.style_tags
            if tag != "unknown" and tag.casefold() not in idea.casefold()
        ]
        if style_tags and "tattoo style" not in missing:
            details.append(("Style", ", ".join(style_tags)))
        for missing_item, label, value in (
            ("placement", "Placement", extracted.placement),
            ("size in cm", "Approximate size", extracted.size_estimate_cm),
            (
                "color preference",
                "Color preference",
                extracted.color_preference,
            ),
            ("preferred date", "Preferred date", extracted.date),
            ("preferred time", "Preferred time", extracted.time),
            (
                "preferred artist",
                "Preferred artist",
                extracted.preferred_artist,
            ),
            (
                "preferred dates or availability",
                "Availability",
                extracted.availability,
            ),
            (
                "tattoo project type",
                "Tattoo project type",
                extracted.tattoo_project_type,
            ),
        ):
            if (
                missing_item not in missing
                and self._is_known_reply_value(value)
                and not (
                    missing_item == "preferred dates or availability"
                    and self._availability_duplicates_schedule(
                        extracted,
                    )
                )
            ):
                details.append((label, self._email_value(value)))
        if extracted.appointment_type and "appointment type" not in missing:
            details.append(
                (
                    "Appointment type",
                    extracted.appointment_type.replace("_", " ").capitalize(),
                )
            )
        return details

    def _availability_duplicates_schedule(
        self,
        extracted: TattooExtractionDraft,
    ) -> bool:
        """Avoid repeating an exact date/time as general availability."""
        availability = " ".join(extracted.availability.split()).casefold()
        if not availability:
            return False
        schedule_values = {extracted.date.casefold()}
        if extracted.date and extracted.time:
            schedule_values.add(
                f"{extracted.date} at {extracted.time}".casefold()
            )
        return availability in schedule_values

    def _email_value(self, value: str, limit: int = 240) -> str:
        """Keep a potentially long extracted value within reply limits."""
        normalized = " ".join(value.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."

    def _is_greeting_only(self, message: str) -> bool:
        """Return whether the latest message contains only a greeting."""
        return bool(_GREETING_PATTERN.fullmatch(message.strip()))

    def _details_already_confirmed(
        self,
        current_message: str,
        history: list[Message],
    ) -> bool:
        """Detect the latest applicable client confirmation or correction."""
        user_messages = [
            message.content
            for message in history
            if message.role == "user"
        ]
        if not user_messages or user_messages[-1] != current_message:
            user_messages.append(current_message)

        confirmed = False
        for message in user_messages:
            normalized = message.casefold()
            if (
                any(term in normalized for term in _CORRECTION_TERMS)
                or _CHANGE_PATTERN.search(message)
            ):
                confirmed = False
            if _CONFIRMATION_PATTERN.search(message):
                confirmed = True
        return confirmed

    def _natural_summary(
        self,
        extracted: TattooExtractionDraft,
    ) -> str:
        """Summarize only known details in one conversational sentence."""
        aggregate_summary = self._aggregate_multi_tattoo_summary(extracted)
        if aggregate_summary:
            return aggregate_summary.replace("It sounds like", "Got it,")
        if len(extracted.projects) > 1:
            return self._human_request_summary(extracted)
        idea = self._idea_fragment(extracted.tattoo_idea)
        style_tags = [
            tag
            for tag in extracted.style_tags
            if tag != "unknown" and tag.casefold() not in idea.casefold()
        ]
        color_preference = extracted.color_preference
        if color_preference == "color":
            color_preference = (
                "" if "watercolor" in style_tags else "full-colour"
            )
        if color_preference == "black-and-grey":
            style_tags = [
                tag for tag in style_tags if tag != "black-and-grey"
            ]
        descriptors = [
            value.strip()
            for value in (
                extracted.size_estimate_cm or extracted.size_description,
                color_preference,
                " and ".join(style_tags),
            )
            if self._is_known_reply_value(value)
        ]
        if idea:
            descriptors.append(idea)
        placement = ""
        if self._is_known_reply_value(extracted.placement):
            placement = extracted.placement.strip()
        if not descriptors and not placement:
            return ""

        subject = " ".join(descriptors).strip() or "tattoo"
        if "tattoo" not in subject.casefold():
            subject = f"{subject} tattoo".strip()
        if placement:
            subject = f"{subject} on your {placement}"
        return f"Got it, {self._with_indefinite_article(subject)}."

    def _with_indefinite_article(self, phrase: str) -> str:
        """Prefix singular designs while leaving counted designs natural."""
        normalized = phrase.strip()
        if re.match(
            r"^(?:multiple|several|two|three|four|five|six|seven|eight|"
            r"nine|ten|\d+)\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            return normalized
        article = "an" if normalized[:1].casefold() in "aeiou" else "a"
        return f"{article} {normalized}"

    def _is_known_reply_value(self, value: str) -> bool:
        """Reject empty and placeholder values from client-facing summaries."""
        normalized = value.strip().casefold()
        return normalized not in {
            "",
            "unknown",
            "none",
            "n/a",
            "not provided",
            "not sure",
        }

    def _greeting_reply(self, history: list[Message]) -> str:
        """Start the required intake sequence with the client's full name."""
        return self._avoid_exact_repeat(
            "Hi! What is your full name?",
            history,
        )

    def _acknowledgement(
        self,
        current_message: str,
        history: list[Message],
    ) -> str:
        """Acknowledge corrections and newly supplied scheduling details."""
        normalized = current_message.casefold()
        if any(term in normalized for term in _REPEATED_DETAIL_TERMS):
            return (
                "You're right - sorry for asking again. I've kept the "
                "details you already provided."
            )
        if any(term in normalized for term in _CORRECTION_TERMS):
            return "Thanks for clarifying - I've updated that."
        if any(
            term in normalized for term in _SCHEDULE_TERMS
        ) or _CLOCK_TIME_PATTERN.search(current_message):
            timing = self._timing_phrase(current_message)
            if timing:
                return f"Got it, I've noted {timing}."
            return "Got it - I've noted the timing."

        alternatives = (
            "Got it.",
            "Okay, noted.",
            "Perfect, I've got that.",
        )
        index = self._assistant_message_count(history) % len(alternatives)
        return alternatives[index]

    def _assistant_message_count(self, history: list[Message]) -> int:
        """Count prior assistant turns for deterministic wording variation."""
        return sum(message.role == "assistant" for message in history)

    def _select_questions(
        self,
        missing_information: Sequence[str],
        history: list[Message],
    ) -> list[str]:
        """Choose one follow-up question, or two on the first intake turn."""
        previous = self._last_assistant_message(history).casefold()
        canonical = [
            item
            for item in MISSING_INFORMATION_OPTIONS
            if item in missing_information
        ]
        legacy = [
            item for item in missing_information if item not in canonical
        ]
        ordered_missing = [*canonical, *legacy]
        not_recently_asked = [
            item
            for item in ordered_missing
            if not any(
                marker in previous
                for marker in _QUESTION_MARKERS[item]
            )
        ]
        candidates = not_recently_asked or ordered_missing
        limit = 1 if self._assistant_message_count(history) else 2
        return [_MISSING_QUESTIONS[item] for item in candidates[:limit]]

    def _last_assistant_message(self, history: list[Message]) -> str:
        """Return the most recent assistant message or an empty string."""
        return next(
            (
                message.content
                for message in reversed(history)
                if message.role == "assistant"
            ),
            "",
        )

    def _timing_phrase(self, message: str) -> str:
        """Echo the client's day and clock time, e.g. "today at 5 pm"."""
        day_match = _DAY_PATTERN.search(message)
        time_match = _CLOCK_TIME_PATTERN.search(message)
        day = day_match.group(0).casefold() if day_match else ""
        if day and day not in {"today", "tomorrow", "tonight"}:
            day = day.capitalize()
        clock = (
            " ".join(time_match.group(0).split()).casefold()
            if time_match
            else ""
        )
        clock = re.sub(r"(\d)(am|pm)$", r"\1 \2", clock)
        if day and clock:
            return f"{day} at {clock}"
        return day or clock

    def _join_reply_parts(self, parts: Sequence[str]) -> str:
        """Join sentences with spaces, keeping multi-line blocks separate."""
        reply = ""
        previous = ""
        for part in parts:
            if not part:
                continue
            if not reply:
                reply = part
            elif "\n" in part or "\n" in previous:
                reply = f"{reply}\n\n{part}"
            else:
                reply = f"{reply} {part}"
            previous = part
        return reply

    def _summary_already_shown(self, history: Sequence[Message]) -> bool:
        """Return whether the detail summary was already offered for review."""
        return any(
            message.role == "assistant"
            and _SUMMARY_CONFIRMATION_MARKER in message.content.casefold()
            for message in history
        )

    def _personalize_artist_question(
        self,
        questions: list[str],
        extracted: TattooExtractionDraft,
        suggested_artist: str,
        history: Sequence[Message],
    ) -> list[str]:
        """Offer the recommended artist instead of the generic artist menu."""
        if suggested_artist not in ARTIST_PREFERENCE_OPTIONS:
            return questions
        recommended_before = suggested_artist.casefold() in (
            self._last_assistant_message(list(history)).casefold()
        )
        if not (
            extracted.artist_preference_mode == "recommend"
            or recommended_before
        ):
            return questions
        generic = _MISSING_QUESTIONS["preferred artist"]
        offer = (
            f"Would you like to go ahead with {suggested_artist}, "
            "or would you prefer another artist?"
        )
        return [offer if question == generic else question for question in questions]

    def _confirm_weekday_date(
        self,
        questions: list[str],
        extracted: TattooExtractionDraft,
    ) -> list[str]:
        """Ask the client to confirm the exact date behind a bare weekday."""
        if not extracted.date_to_confirm:
            return questions
        confirmation = date_confirmation_question(extracted.date_to_confirm)
        others = [
            question
            for question in questions
            if question != _MISSING_QUESTIONS["preferred dates or availability"]
        ]
        return [confirmation, *others][: max(1, len(questions))]

    def _avoid_exact_repeat(
        self,
        reply: str,
        history: list[Message],
    ) -> str:
        """Avoid returning the immediately previous assistant message verbatim."""
        previous = self._last_assistant_message(history)
        if not previous or previous.strip().casefold() != reply.casefold():
            return reply
        return f"I've got that. {reply}"
