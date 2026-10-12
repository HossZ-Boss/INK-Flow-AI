"""Staff-facing inquiry review summary composition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .schemas import AIExtractionOutput, TattooInquiryInput


class HighRiskSummaryBuilder:
    """Create concise staff summaries from validated inquiry analysis."""

    def build(
        self,
        inquiry: TattooInquiryInput,
        analysis: AIExtractionOutput,
    ) -> str:
        """Build one concise narrative paragraph for studio staff."""
        if not analysis.telegram_review_required:
            raise ValueError("A Telegram summary requires staff review.")

        state = inquiry.existing_db_state
        client = self._client_description(state, analysis.client_name)
        idea = self._clean_text(analysis.tattoo_idea)
        project_type = self._clean_text(analysis.tattoo_project_type)
        if idea:
            project = project_type or "tattoo"
            opening = f'{client} is requesting a {project}: "{idea}".'
        elif project_type:
            opening = f"{client} is requesting a {project_type}."
        else:
            opening = f"{client} is requesting a tattoo."

        sentences = [opening]
        design_description = self._design_description(analysis)
        if design_description:
            sentences.append(design_description)

        logistics = self._logistics_description(inquiry, analysis)
        if logistics:
            sentences.append(logistics)

        if analysis.missing_information:
            sentences.append(
                "Still needed: "
                f"{self._natural_list(analysis.missing_information)}."
            )
        else:
            sentences.append("All intake details are complete.")

        if analysis.suggested_artist != "Unclear":
            sentences.append(
                f"Suggested artist: {analysis.suggested_artist} "
                f"({analysis.confidence_level} confidence)."
            )
        else:
            sentences.append("Artist assignment is pending.")

        price = self._price_description(analysis)
        if price:
            sentences.append(price)

        return " ".join(
            sentence if sentence.endswith((".", "!", "?")) else sentence + "."
            for sentence in sentences
        )

    def combine_with_draft(self, summary: str, draft_reply: str) -> str:
        """Place the client draft after the staff summary for Telegram."""
        if not draft_reply.strip():
            return (
                f"{summary}\n\nDRAFT REPLY\n(No draft: the AI service is "
                "unavailable. Please reply to the client manually.)"
            )
        return f"{summary}\n\nDRAFT REPLY\n{draft_reply}"

    def _client_description(
        self,
        state: dict[str, Any],
        extracted_client_name: str = "",
    ) -> str:
        """Return the client name without duplicating backend metadata."""
        name = self._state_value(state, "lead_name", "name")
        return name or extracted_client_name or "The client"

    def _design_description(self, analysis: AIExtractionOutput) -> str:
        """Describe known design attributes as one natural sentence."""
        clauses: list[str] = []
        style = self._style(analysis)
        if style:
            if len(style) == 1:
                clauses.append(f"uses a {style[0]} style")
            else:
                clauses.append(
                    f"uses {self._natural_list(style)} styles"
                )
        if self._is_known(analysis.size_estimate_cm):
            clauses.append(
                "measures approximately "
                f"{self._clean_text(analysis.size_estimate_cm)}"
            )
        if self._is_known(analysis.placement):
            clauses.append(
                f"is intended for the {self._clean_text(analysis.placement)}"
            )
        if self._is_known(analysis.color_preference):
            clauses.append(
                "uses "
                f"{self._clean_text(analysis.color_preference)} ink"
            )
        if not clauses:
            return ""
        return f"The design {self._natural_list(clauses)}."

    def _logistics_description(
        self,
        inquiry: TattooInquiryInput,
        analysis: AIExtractionOutput,
    ) -> str:
        """Combine references, preferences, and scheduling into one sentence."""
        clauses: list[str] = []
        count = len(inquiry.new_image_urls)
        if count:
            quantity = "one" if count == 1 else str(count)
            noun = "image" if count == 1 else "images"
            clauses.append(f"provided {quantity} reference {noun}")
        if analysis.artist_preference_mode == "recommend":
            clauses.append("asked for an artist recommendation")
        elif analysis.preferred_artist == "No preference":
            clauses.append("has no artist preference")
        elif analysis.preferred_artist:
            clauses.append(f"prefers {analysis.preferred_artist}")
        if analysis.appointment_type == "online":
            clauses.append("wants an online appointment")
        elif analysis.appointment_type == "studio_visit":
            clauses.append("wants a studio visit")
        if analysis.availability:
            clauses.append(f"is available on {analysis.availability}")
        elif analysis.date and analysis.time:
            clauses.append(
                f"prefers {analysis.date} at {analysis.time}"
            )
        elif analysis.date:
            clauses.append(f"prefers {analysis.date}")
        elif analysis.time:
            clauses.append(f"prefers {analysis.time}")
        if not clauses:
            return ""
        return f"The client {self._natural_list(clauses)}."

    def _price_description(self, analysis: AIExtractionOutput) -> str:
        """Describe the staff-only price suggestion, if one exists."""
        price = analysis.price_suggestion
        if price is None:
            return ""
        source = (
            "based on approved studio prices"
            if price.basis == "approved_history"
            else "AI estimate"
        )
        details = [f"Suggested price ({source}, not approved): {price.label()}"]
        if price.estimated_hours:
            details.append(f"about {self._clean_text(price.estimated_hours)}")
        details.append(f"{price.confidence_level} confidence")
        return ", ".join(details) + "."

    def _style(self, analysis: AIExtractionOutput) -> list[str]:
        """Return known style tags for the narrative description."""
        return [tag for tag in analysis.style_tags if tag != "unknown"]

    def _state_value(self, state: dict[str, Any], *keys: str) -> str:
        """Read a known scalar from root state or a nested lead record."""
        records: list[Mapping[str, Any]] = [state]
        lead = state.get("lead")
        if isinstance(lead, Mapping):
            records.append(lead)
        for record in records:
            for key in keys:
                value = record.get(key)
                if isinstance(value, (str, int, float)):
                    normalized = self._clean_text(str(value))
                    if normalized:
                        return normalized
        return ""

    def _natural_list(self, values: list[str]) -> str:
        """Join values using natural English punctuation."""
        if not values:
            return ""
        if len(values) == 1:
            return values[0]
        if len(values) == 2:
            return f"{values[0]} and {values[1]}"
        return f"{', '.join(values[:-1])}, and {values[-1]}"

    def _is_known(self, value: str) -> bool:
        """Return whether a value is useful in a staff-facing summary."""
        return value.strip().casefold() not in {
            "",
            "unknown",
            "unclear",
            "none",
            "n/a",
            "not provided",
        }

    def _clean_text(self, value: str) -> str:
        """Collapse whitespace so the summary always remains one paragraph."""
        return " ".join(value.split())
