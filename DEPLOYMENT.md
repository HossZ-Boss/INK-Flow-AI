# Production Deployment on the Shared EC2 Instance

The FastAPI service runs in its own container under
`/opt/tattoo-hysteria-ai`. It joins the backend's existing external Docker
network and does not publish port 8001 to the EC2 host.

## 1. EC2 prerequisites

Docker and the Docker Compose plugin should already exist because the Django
stack uses them. Confirm that the `ubuntu` user can run Docker:

```bash
docker --version
docker compose version
```

GitHub Actions creates `/opt/tattoo-hysteria-ai`, transfers the exact Git commit
as a bundle, and initializes or updates the checkout automatically. The EC2
instance does not need a GitHub deploy key, personal access token, or manual
repository clone.

The scoped deployment script also creates `tattoo_hysteria_net` when it does not
already exist. Docker and the Compose plugin remain server prerequisites because
the existing backend stack already provides them.

## 2. Production environment file

GitHub Actions creates the server-only environment file automatically during
every production deployment. Do not create it manually and do not commit it.

The workflow generates this file at `/opt/tattoo-hysteria-ai/.env`:

```env
OPENAI_API_KEY=value_from_the_github_production_secret
OPENAI_MODEL=gpt-6-astra
OPENAI_REASONING_EFFORT=low
OPENAI_TEMPERATURE=0.0
OPENAI_TIMEOUT_SECONDS=30
OPENAI_MAX_RETRIES=2
```

The workflow transfers the key through encrypted SSH, writes a temporary file
with restrictive permissions, and atomically replaces `.env`. It does not print
the key. The Compose file loads it only at container runtime.

The service does not send Telegram messages itself, so it does not need a
Telegram bot token. It returns a Telegram-ready summary to Django.

## 3. First deployment

Push to `main`, or manually run the production workflow from GitHub Actions. The
same workflow handles both the first deployment and later updates.

The script performs these operations only for the AI service:

- Validates the production Compose configuration.
- Creates the shared network only when it does not exist.
- Builds the AI image from the current checkout.
- Starts `tattoo_hysteria_ai` without publishing port 8001.
- Waits for the readiness health check.
- Keeps the prior AI image under `tattoo-hysteria-ai:rollback`.

It does not remove networks, volumes, backend containers, or global images.

## 4. GitHub Actions production environment

In GitHub, open `Settings`, then `Environments`, and create an environment
named `production`. Add a required reviewer if your GitHub plan supports it.

Add these environment secrets:

- `EC2_HOST`: Public EC2 IP address or DNS name.
- `EC2_USER`: Usually `ubuntu`.
- `EC2_SSH_PORT`: Usually `22`.
- `EC2_SSH_PRIVATE_KEY`: Private deployment key used only by Actions.
- `EC2_SSH_KNOWN_HOSTS`: Verified SSH known-hosts entry for the EC2 host.
- `OPENAI_API_KEY`: Project-scoped production OpenAI API key.
- `OPENAI_MODEL`: Model used for extraction, vision, routing, and natural replies.
- `OPENAI_REASONING_EFFORT`: Reasoning level for supported OpenAI models.

Generate a dedicated Actions deployment key on a trusted machine:

```bash
ssh-keygen -t ed25519 \
  -C "github-actions-tattoo-hysteria-ai" \
  -f github-actions-tattoo-hysteria-ai
```

Add the public key to `/home/ubuntu/.ssh/authorized_keys` on EC2. Store the
entire private key in `EC2_SSH_PRIVATE_KEY`.

Generate the known-hosts line only after verifying the EC2 SSH host-key
fingerprint through a trusted channel:

```bash
ssh-keyscan -H YOUR_EC2_HOST
```

Store the complete output line in `EC2_SSH_KNOWN_HOSTS`.

The workflow does not need AWS access keys, database credentials, Django
secrets, or Telegram credentials. GitHub's automatic `GITHUB_TOKEN` is
sufficient for checkout.

## 5. CI/CD behavior

Pull requests to `main` run the complete tests and validate the Docker build.
A push to `main`, or a manual workflow dispatch from `main`, additionally:

1. Connects to EC2 over verified SSH.
2. Creates the missing AI directory with ownership assigned to the SSH user.
3. Transfers the exact commit as a Git bundle over the SSH connection.
4. Initializes or updates the EC2 checkout without an EC2-side GitHub key.
5. Creates the protected production `.env` from the GitHub environment secret.
6. Runs `scripts/deploy-production.sh`.
7. Rolls back to the previous AI image if the new container is unhealthy.

## 6. Backend connectivity checks

Enter the Django container:

```bash
cd /opt/tattoo-hysteria-backend
docker compose -f docker-compose.prod.yml exec backend sh
```

Verify private service discovery and liveness:

```bash
curl -fsS http://tattoo_hysteria_ai:8001/health
```

Verify the analysis endpoint with a proper POST body:

```bash
curl -fsS -X POST \
  http://tattoo_hysteria_ai:8001/api/v1/inquiries/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "current_message": "I want a 10cm floral tattoo on my arm.",
    "new_image_urls": [],
    "existing_db_state": {},
    "recent_chat_history": []
  }'
```

Leave the backend container with `exit`.

Confirm that the AI service has no published host port:

```bash
docker port tattoo_hysteria_ai
```

The command should print nothing. Do not add EC2 security-group access for
port 8001.

## 7. Operations

Check status and logs:

```bash
cd /opt/tattoo-hysteria-ai
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs --tail 200 ai
```

Manually restore the retained AI rollback image if necessary:

```bash
cd /opt/tattoo-hysteria-ai
docker image tag tattoo-hysteria-ai:rollback tattoo-hysteria-ai:latest
docker compose -f docker-compose.prod.yml up -d --force-recreate
```

Never run `docker system prune -af` on this shared EC2 instance.

## 8. Learning layer persistence

The service loads the learning index from `LEARNING_INDEX_PATH` (default
`data/learning.faiss`, i.e. `/app/data` in the container) at startup and saves
it after every `POST /api/v1/learning/feedback`. Compose mounts the named
volume `tattoo_hysteria_ai_learning_data` at `/app/data`, so the index
survives rebuilds and restarts even though the root filesystem is read-only.
Never delete that volume unless you intend to discard everything the AI has
learned. Back it up with:

```bash
docker run --rm -v tattoo_hysteria_ai_learning_data:/data -v "$PWD":/backup \
  alpine tar czf /backup/learning-data.tgz -C /data .
```

If the index files are unreadable at startup, they are renamed with a
`.broken-<timestamp>` suffix and a new empty index is started. Set
`LEARNING_ENABLED=false` to turn learning off entirely.

Keep one Uvicorn worker: index writes are coordinated only inside one process.

When validating Compose, use the quiet form below because the normal `config`
output expands environment values and can expose secrets in logs:

```bash
docker compose -f docker-compose.prod.yml config --quiet
```
