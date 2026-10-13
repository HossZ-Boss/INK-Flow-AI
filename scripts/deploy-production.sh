#!/usr/bin/env bash

set -Eeuo pipefail

readonly EXPECTED_DIRECTORY="/opt/tattoo-hysteria-ai"
readonly COMPOSE_FILE="docker-compose.prod.yml"
readonly CONTAINER_NAME="tattoo_hysteria_ai"
readonly IMAGE_NAME="tattoo-hysteria-ai:latest"
readonly ROLLBACK_IMAGE="tattoo-hysteria-ai:rollback"
readonly NETWORK_NAME="tattoo_hysteria_net"

current_directory="$(pwd -P)"
if [[ "${current_directory}" != "${EXPECTED_DIRECTORY}" ]]; then
    echo "Deployment must run from ${EXPECTED_DIRECTORY}." >&2
    exit 1
fi

if [[ ! -f ".env" ]]; then
    echo "Missing ${EXPECTED_DIRECTORY}/.env." >&2
    exit 1
fi

docker network inspect "${NETWORK_NAME}" >/dev/null 2>&1 \
    || docker network create "${NETWORK_NAME}" >/dev/null

previous_image_id="$({
    docker inspect --format '{{.Image}}' "${CONTAINER_NAME}" 2>/dev/null
} || true)"

if [[ -n "${previous_image_id}" ]] \
    && ! docker image inspect "${previous_image_id}" >/dev/null 2>&1; then
    echo "Image of the current container no longer exists;" \
        "deploying without a rollback image." >&2
    previous_image_id=""
fi

if [[ -n "${previous_image_id}" ]]; then
    docker image rm "${ROLLBACK_IMAGE}" >/dev/null 2>&1 || true
    docker image tag "${previous_image_id}" "${ROLLBACK_IMAGE}"
fi

wait_for_health() {
    local attempt
    local health_status
    for attempt in $(seq 1 30); do
        health_status="$({
            docker inspect \
                --format '{{if .State.Health}}{{.State.Health.Status}}{{end}}' \
                "${CONTAINER_NAME}" 2>/dev/null
        } || true)"
        if [[ "${health_status}" == "healthy" ]]; then
            return 0
        fi
        if [[ "${health_status}" == "unhealthy" ]]; then
            return 1
        fi
        sleep 2
    done
    return 1
}

rollback() {
    if [[ -z "${previous_image_id}" ]]; then
        return 0
    fi
    echo "Deployment failed. Restoring the previous AI image." >&2
    docker image tag "${ROLLBACK_IMAGE}" "${IMAGE_NAME}"
    docker compose --file "${COMPOSE_FILE}" up \
        --detach --force-recreate --remove-orphans
    wait_for_health || true
}

docker compose --file "${COMPOSE_FILE}" config --quiet
docker compose --file "${COMPOSE_FILE}" build --pull

if ! docker compose --file "${COMPOSE_FILE}" up \
    --detach --remove-orphans; then
    rollback
    exit 1
fi

if ! wait_for_health; then
    docker compose --file "${COMPOSE_FILE}" logs --tail 200 ai >&2
    rollback
    exit 1
fi

docker compose --file "${COMPOSE_FILE}" ps
echo "Tattoo Hysteria AI deployment is healthy."

