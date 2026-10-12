# INK Flow AI

INK Flow AI is a small FastAPI service for analyzing tattoo inquiries. It uses
the latest message, up to 30 recent chat messages, saved inquiry details, and
optional reference images to suggest an artist, classify risk, and draft a
reply.

## Requirements

- Python 3.12
- An OpenAI API key

## Run locally

Open PowerShell in the project folder and create a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Create a `.env` file in the project folder:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-6-astra
OPENAI_REASONING_EFFORT=low
```

`gpt-6-astra` is the default production model. Set `OPENAI_MODEL` only when a
deployment needs a different cost or latency profile.

Start the API with:

```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8001 --reload
```

The interactive API documentation is available at
http://127.0.0.1:8001/docs.

## Using the API

Send a `POST` request to `/api/v1/inquiries/analyze` with a payload like this:

```json
{
  "current_message": "I want a 10cm black floral tattoo on my arm.",
  "message_source": "whatsapp",
  "new_image_urls": [],
  "existing_db_state": {},
  "recent_chat_history": [
    {"role": "user", "content": "I am thinking about a floral design."}
  ]
}
```

Set `message_source` to `outlook` to receive a professional email draft that
uses natural prose and asks only the next one or two useful questions. When
omitted, it defaults to the `source` stored in `existing_db_state.intake` or
`existing_db_state.lead`, then falls back to `whatsapp`. WhatsApp keeps the
concise conversational reply format. Outlook and Gmail quoted reply threads,
including common signatures, are removed before extraction so previous studio
questions and sign-offs cannot be mistaken for new client answers.

Multiple tattoos remain in the flat response contract. `tattoo_idea` preserves
all designs, while placement and shared details remain top-level fields. The
public response does not contain a nested `projects` field. Qualitative answers
such as `hand-sized` are stored in `size_description`; an explicit `not sure`
is accepted and flagged for staff help instead of being asked repeatedly.
`artist_preference_mode` distinguishes a named artist from requests such as
"please recommend the best fit". A recommendation request does not populate
`preferred_artist`; that field changes only when the client selects an artist
or explicitly says that they have no preference.

Every response includes the exact JSON key `Auto-reply`. For WhatsApp it
equals `auto_reply_allowed`, so it is true whenever the draft is safe to send
and false for high-risk or staff-review inquiries. For Outlook it is true only
for a recognized tattoo inquiry or an established tattoo-intake follow-up
that is also safe to send automatically. It is false for unrelated mail,
automated notifications, newsletters, courtesy-only replies, other sources
such as `vcita`, high-risk inquiries, and no-reply senders. The backend should provide
the real sender address as `existing_db_state.sender_email` or
`existing_db_state.lead.email`, and may provide the subject as
`existing_db_state.email_subject`.

The structured `appointment_type` response field is either `online`,
`studio_visit`, or an empty string while it is still unknown. Client-facing
replies display these choices naturally as “online” and “studio visit”.

`risk_level` remains `low` while required intake information is missing and
becomes `high` only when the required intake is complete. Separately,
`staff_review_required`, `review_reasons`, and `intake_status` allow complex
but incomplete cases—such as group requests or an unknown size—to reach staff
without falsely marking them complete.

Tongue tattoo requests keep collecting every intake detail like any other
request. The client is told once that the studio must approve the placement
before an artist, price, or booking is confirmed, and no artist is recommended.
When the intake is complete, the request becomes high risk and reaches staff
with the `specialised_placement_requires_approval` review reason. Status-update
questions pause intake immediately: they set `staff_review_required` and
`telegram_review_required` to true, set `intake_status` to
`needs_staff_review`, and disable both automatic-reply controls when no
verified backend status is available. Image-led style wording is kept as
`reference-led design` in the
notes unless text or vision provides a supported named style.

Use `/api/v1/inquiries/telegram-summary` whenever
`telegram_review_required` is true. It returns a concise staff summary, the
generated draft reply, review reasons, and `reference_image_urls`. The calling
backend must send those image URLs as Telegram media; this service only
returns them. Telegram consumers must not merge prices, schedules, service
codes, or draft replies from older intake records. Request and active-intake
association remains the calling backend's responsibility.

## Staff-only price suggestions

Every high-risk analysis includes `price_suggestion`, both in
`/api/v1/inquiries/analyze` and `/api/v1/inquiries/telegram-summary`. The
model estimates a EUR range for an Amsterdam studio from the tattoo details
and up to three reference images. When staff-approved prices for similar
requests exist in the learning index, they are passed in as anchors. The
Telegram summary ends with a line such as
`Suggested price (AI estimate, not approved): €350-500, about 3-4 hours,
medium confidence.` The field is `null` for low-risk requests or when no
estimate could be made. It is marked `internal_only` and must never be sent
to the client.

## Learning layer

Learning is active by default (`LEARNING_ENABLED=true`). After staff finish
reviewing a request, the backend posts the outcome to
`POST /api/v1/learning/feedback`:

```json
{
  "inquiry_id": "129",
  "channel": "whatsapp",
  "style_tags": ["blackwork"],
  "placement": "forearm",
  "size_estimate_cm": "15 cm",
  "color_preference": "black-and-grey",
  "ai_suggested_artist": "Sandra",
  "final_artist": "Hossam",
  "ai_suggestion_outcome": "corrected",
  "correction_reason": "Hossam had availability.",
  "approved_price_range": {"currency": "EUR", "minimum": "400", "maximum": "550"}
}
```

Artists may be given by display name or key. Re-sending the same
`inquiry_id` with the same content is a no-op; different content for an
existing `inquiry_id` returns 409. The example is embedded, added to the
FAISS index, and saved to `LEARNING_INDEX_PATH` (default
`data/learning.faiss`). Future artist routing, next-action decisions, and
price suggestions use it immediately; there is no minimum record count.
`GET /api/v1/learning/status` reports the record counts.

## Run the tests

```powershell
python -m pytest -q
```

Keep the `.env` file private and do not commit it to source control.

## Production deployment

The production container listens on `0.0.0.0:8001` inside Docker and joins the
external `tattoo_hysteria_net` network. It does not publish port 8001 to the
EC2 host. See [DEPLOYMENT.md](DEPLOYMENT.md) for the complete EC2, Compose,
CI/CD, secret, rollback, and backend-verification instructions.
