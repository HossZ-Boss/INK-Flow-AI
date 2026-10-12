FROM python:3.12-slim

LABEL org.opencontainers.image.title="Tattoo Hysteria AI"
LABEL org.opencontainers.image.description="Private FastAPI AI service"
LABEL com.tattoo-hysteria.service="ai"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install --no-install-recommends --yes libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home app

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --requirement requirements.txt

COPY --chown=app:app ai_brain ./ai_brain
COPY --chown=app:app api ./api
COPY --chown=app:app config ./config
RUN mkdir -p /app/data && chown app:app /app/data

USER app

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-m", "api.healthcheck"]

CMD ["python", "-m", "uvicorn", "api.main:app", \
    "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]

