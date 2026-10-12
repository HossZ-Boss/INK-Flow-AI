"""Versioned FastAPI route definitions."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import ValidationError

from ai_brain.config import get_settings
from ai_brain.errors import AnalysisPipelineError, ConfigurationError
from ai_brain.learning import LearningConflictError, LearningStore
from ai_brain.processor import StudioAIBrain
from ai_brain.summary import HighRiskSummaryBuilder

from .constants import SERVICE_NAME, SERVICE_VERSION
from .dependencies import get_ai_brain, get_learning_store
from .schemas import (
    AIExtractionOutput,
    ErrorResponse,
    HealthResponse,
    LearningFeedbackInput,
    LearningFeedbackResponse,
    LearningStatusResponse,
    TattooInquiryInput,
    TelegramSummaryInput,
    TelegramSummaryResponse,
)

LOGGER = logging.getLogger(__name__)

health_router = APIRouter(prefix="/health", tags=["Health"])
inquiry_router = APIRouter(
    prefix="/api/v1/inquiries",
    tags=["Tattoo inquiries"],
)
learning_router = APIRouter(
    prefix="/api/v1/learning",
    tags=["Learning"],
)

BrainDependency = Annotated[StudioAIBrain, Depends(get_ai_brain)]
SUMMARY_BUILDER = HighRiskSummaryBuilder()


def _optional_learning_store() -> LearningStore | None:
    """Return the learning store, mapping missing configuration to 503."""
    try:
        return get_learning_store()
    except ConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service is not configured.",
        ) from exc


LearningDependency = Annotated[
    LearningStore | None,
    Depends(_optional_learning_store),
]


def _process_inquiry(
    payload: TattooInquiryInput,
    brain: StudioAIBrain,
) -> AIExtractionOutput:
    """Run one inquiry analysis and map domain errors to HTTP responses."""
    try:
        return brain.process_inquiry(
            current_message=payload.current_message,
            new_image_urls=payload.new_image_urls,
            existing_db_state=payload.existing_db_state,
            recent_chat_history=payload.recent_chat_history,
            message_source=payload.message_source,
        )
    except AnalysisPipelineError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service is not configured.",
        ) from exc
    except Exception as exc:
        LOGGER.exception("Unhandled AI pipeline failure")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI analysis failed. Please retry later.",
        ) from exc


@health_router.get(
    "/live",
    response_model=HealthResponse,
    summary="Check service liveness",
)
def check_liveness() -> HealthResponse:
    """Confirm that the HTTP process is running."""
    return HealthResponse(
        status="ok",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
    )


@health_router.get(
    "",
    response_model=HealthResponse,
    summary="Check service health",
)
def check_health() -> HealthResponse:
    """Provide the backend-compatible liveness endpoint."""
    return check_liveness()


@health_router.get(
    "/ready",
    response_model=HealthResponse,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "OpenAI configuration is missing or invalid.",
        }
    },
    summary="Check AI service readiness",
)
def check_readiness() -> HealthResponse:
    """Confirm that required AI configuration is valid."""
    try:
        get_settings()
    except ConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service is not configured.",
        ) from exc

    return HealthResponse(
        status="ready",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
    )


@inquiry_router.post(
    "/analyze",
    response_model=AIExtractionOutput,
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "The inquiry cannot be processed.",
        },
        status.HTTP_502_BAD_GATEWAY: {
            "model": ErrorResponse,
            "description": "The upstream AI pipeline failed.",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "The AI service is not configured.",
        },
    },
    summary="Analyze a tattoo inquiry",
)
def analyze_inquiry(
    payload: TattooInquiryInput,
    brain: BrainDependency,
) -> AIExtractionOutput:
    """Forward validated hybrid context to the unified AI Brain."""
    return _process_inquiry(payload, brain)


@inquiry_router.post(
    "/telegram-summary",
    response_model=TelegramSummaryResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "The inquiry cannot be processed.",
        },
        status.HTTP_409_CONFLICT: {
            "model": ErrorResponse,
            "description": "The inquiry does not require staff review.",
        },
        status.HTTP_502_BAD_GATEWAY: {
            "model": ErrorResponse,
            "description": "The upstream AI pipeline failed.",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "The AI service is not configured.",
        },
    },
    summary="Create a staff-review Telegram summary",
)
def create_telegram_summary(
    payload: TelegramSummaryInput,
    brain: BrainDependency,
) -> TelegramSummaryResponse:
    """Return review text and media references without sending Telegram."""
    analysis = _process_inquiry(payload, brain)
    if not analysis.telegram_review_required:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Telegram summary is available only for staff-review inquiries.",
        )

    summary = SUMMARY_BUILDER.build(payload, analysis)
    telegram_message = SUMMARY_BUILDER.combine_with_draft(
        summary=summary,
        draft_reply=analysis.draft_reply,
    )
    return TelegramSummaryResponse(
        risk_level=analysis.risk_level,
        staff_review_required=True,
        review_reasons=analysis.review_reasons,
        reference_image_urls=payload.new_image_urls,
        summary=summary,
        draft_reply=analysis.draft_reply,
        telegram_message=telegram_message,
        price_suggestion=analysis.price_suggestion,
    )


@learning_router.post(
    "/feedback",
    response_model=LearningFeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_409_CONFLICT: {
            "model": ErrorResponse,
            "description": "The inquiry ID was already stored with other data.",
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "model": ErrorResponse,
            "description": "An artist name is not in the studio roster.",
        },
        status.HTTP_502_BAD_GATEWAY: {
            "model": ErrorResponse,
            "description": "The embedding provider failed.",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "The learning layer is disabled or not configured.",
        },
    },
    summary="Store a staff-reviewed outcome for future suggestions",
)
def add_learning_feedback(
    payload: LearningFeedbackInput,
    store: LearningDependency,
) -> LearningFeedbackResponse:
    """Index one reviewed request so future artist and price suggestions use it."""
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Learning layer is disabled.",
        )
    artist_keys: dict[str, str | None] = {}
    for field_name in ("ai_suggested_artist", "final_artist"):
        name = getattr(payload, field_name)
        key = store.artist_key(name) if name else None
        if name and key is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unknown artist in {field_name}: {name}.",
            )
        artist_keys[field_name] = key

    try:
        example = store.add_example(
            {
                "inquiry_id": payload.inquiry_id,
                "channel": payload.channel,
                "style_tags": payload.style_tags,
                "placement": payload.placement,
                "size_estimate_cm": payload.size_estimate_cm,
                "color_preference": payload.color_preference,
                "original_ai_artist_key": artist_keys["ai_suggested_artist"],
                "final_artist_key": artist_keys["final_artist"],
                "original_ai_action": payload.ai_suggested_action,
                "final_action": payload.final_action,
                "ai_suggestion_outcome": payload.ai_suggestion_outcome,
                "correction_reason": payload.correction_reason,
                "approved_price_range": payload.approved_price_range,
            }
        )
    except LearningConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Learning feedback is not a valid example.",
        ) from exc
    except Exception as exc:
        LOGGER.exception("Learning feedback could not be stored")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Learning feedback could not be stored. Please retry later.",
        ) from exc
    return LearningFeedbackResponse(
        example_id=example.example_id,
        record_count=store.record_count,
    )


@learning_router.get(
    "/status",
    response_model=LearningStatusResponse,
    summary="Show how much reviewed history the AI has learned from",
)
def get_learning_status(store: LearningDependency) -> LearningStatusResponse:
    """Report whether learning is on and how many records it holds."""
    if store is None:
        return LearningStatusResponse(
            enabled=False,
            record_count=0,
            records_with_approved_price=0,
        )
    return LearningStatusResponse(
        enabled=True,
        record_count=store.record_count,
        records_with_approved_price=store.approved_price_count,
    )
