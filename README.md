# Social Media OCR Translation Pipeline

This project extracts chat conversations from screenshots, reconstructs the conversation structure, and renders a translated chat-style output image.

It is currently optimized for Facebook Messenger-style conversations, but the pipeline is being kept modular so other chat platforms can be supported later.

## What It Does

Given a folder of input chat screenshots, the pipeline:

1. Cleans the images and prepares them for analysis.
2. Uses Gemini vision to transcribe the conversation structure from the screenshots.
3. Uses OCR hints to refine the source-language message text.
4. Runs one Gemini pass for reference resolution, final English, **and** status-bar/header fields (using a crop of the first screenshot’s top bar).
5. Renders the final conversation as a clean chat image, along with debug comparison outputs.

## Current Pass Structure

1. `Pass 1`: source transcription and conversation structure
2. `Pass 2`: OCR-guided source-text polishing
3. `Pass 3`: reference resolution, final English translation, **and** header/status-bar extraction (first image’s status bar attached to the same request; independent JSON fields so it does not alter the transcript task)

System metadata such as timestamps and call notices is handled separately from normal chat bubbles and merged back in only at the final rendering stage.

## Main Files

- `main.py`: pipeline entry point and orchestration
- `ocr_translate.py`: Gemini/OCR pipeline logic
- `chat_renderer.py`: rendered chat output
- `config.py`: paths, environment, and model configuration
- `pass1_bubble_input.txt`: manual bubble-count/order guidance for Pass 1

## Expected Local Structure

- `input_images/`: place source screenshots here
- `rendered_chat/`: generated images
- `result/`: prompt/debug text outputs
- `result_json/`: structured JSON debug outputs

The generated runtime folders are git-ignored so the repository stays clean. `input_images/` is kept as an empty tracked directory, but actual user screenshots should remain local and should not be committed.

## Running

Typical flow:

1. Copy `.env.example` to `.env` in the project root and set **`GEMINI_API_KEY`** (required). **`GOOGLE_VISION_API_KEY`** is optional but improves hints. You do **not** need to `export` variables in the shell — `python main.py` loads `.env` automatically (same as `web_app.py`).
2. Put screenshots into `input_images/` (only image files are used; `.gitkeep` is ignored).
3. Update `pass1_bubble_input.txt` if needed (or answer prompts when the CLI asks for bubble counts).
4. From the project root, run `python main.py`
5. Check `rendered_chat/`, `result/`, and `result_json/`

The full vision pipeline **cannot** run without a Gemini key; there is no offline translation mode.

## Goal

The main goal of the project is accurate conversation reconstruction and translation, especially in difficult cases where OCR is noisy, chat UI artifacts are present, or subject/reference resolution is ambiguous.

## HTTP API & billing (Paddle)

The FastAPI app (`web_app.py`) exposes translation jobs plus optional **[Paddle Billing](https://www.paddle.com/billing)** (merchant of record). Paddle can pay out to sellers in Israel and handles tax and payment methods; you configure products/prices in Paddle, not a card processor directly.

### Entitlements (PostgreSQL)

Stored in the same database as users. Set **`DATABASE_URL`** (PostgreSQL connection string). On Render, use the **Internal Database URL** from **translate-chat-db** on the **web** service. Tables are created on first use (`users`, `billing_entitlements`, `billing_guest_entitlements`, etc.). **`GET /health`** includes **`database_url_configured`** (boolean; URL is never logged).

- `free_runs_used` / cap of **1** single-image free run per signed-in account (when not subscribed and no credits)
- `paid_job_credits` (one-time “full run” purchases)
- `access_until` (ISO timestamp) for active subscription billing period; monthly run quota is separate (`BILLING_SUBSCRIPTION_RUNS_PER_MONTH`)
- `paddle_customer_id`, `paddle_address_id`, `paddle_subscription_id` (legacy `stripe_*` columns may still exist from older installs and are migrated/read for compatibility)

**Guest (anonymous) rows** in `billing_guest_entitlements`: **1** free single-image run per `X-Guest-Billing-Id` (8–64 hex); `paid_job_credits` from one-time checkout via `POST /billing/guest-checkout-session` (webhook `custom_data.guest_key`, and **`POST /billing/guest-claim-transaction`** after checkout so credits apply even if webhooks are slow); multi-image allowed when credits > 0.

### Environment variables

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | PostgreSQL URL for users + billing (`postgresql://…`; `postgres://` is normalized). Also checks `POSTGRES_URL`, or `DATABASE_URL_FILE` (first line = URL). |
| `BILLING_ENFORCE` | Set to `1` to enforce quotas on `POST /jobs`: signed-in users use Bearer; guests send **X-Guest-Billing-Id** (free tier or guest-paid credits). |
| `REQUIRE_AUTH_FOR_JOBS` | `1` = user jobs need Bearer; **guest** jobs need the same **X-Guest-Billing-Id** on poll/artifact requests. |
| `PADDLE_API_KEY` | Paddle server API key (Dashboard → Developer tools) |
| `PADDLE_WEBHOOK_SECRET` | Webhook signing secret from Paddle (notifications) |
| `PADDLE_SANDBOX` | Leave **unset** or `0` for **live** (`api.paddle.com`). Set `1` only if you use a separate Paddle **sandbox** account (optional). |
| `PADDLE_API_BASE` | Optional override of production API host (default `https://api.paddle.com`) |
| `PADDLE_PRICE_SINGLE` | Price ID for one-time full-run credit |
| `PADDLE_PRICE_DEBUG` | Optional one-time **debug** price (≥ **$0.70 USD** in Paddle — minimum charge); grants **1 job credit** like `single` |
| `PADDLE_PRICE_MONTH` | Price ID for monthly subscription |
| `PADDLE_PRICE_SIXMO` | Price ID for every-6-month subscription |
| `PADDLE_PRICE_YEAR` | Price ID for annual subscription (bill yearly) |
| `BILLING_SUBSCRIPTION_RUNS_PER_MONTH` | Max jobs per calendar month while subscribed (default `7`) |
| `PADDLE_CHECKOUT_COUNTRY` | ISO country for billing address on checkout (default `IL`) |
| `PADDLE_CHECKOUT_POSTAL_CODE` | Postal code for that address (default `6100001`) |
| `PADDLE_CHECKOUT_REGION` / `PADDLE_CHECKOUT_CITY` | Optional region and city |
| `PADDLE_WEBHOOK_TOLERANCE_SEC` | Max age of webhook timestamp for signature verification (default `300`) |
| `FRONTEND_URL` | SPA origin for checkout success/cancel redirects (no trailing slash) |

### Endpoints

- `GET /billing/status` — whether Paddle API key, webhook secret, and price env vars are set (no secrets)
- `GET /billing/guest-status` — guest free-tier counts (**header: X-Guest-Billing-Id**)
- `GET /billing/me` — current entitlements (**Authorization: Bearer** required)
- `POST /billing/checkout-session` — JSON `{ "plan": "single" \| "debug" \| "month" \| "sixmo" \| "year" }` → `{ "url" }` (**Bearer**; signed-in users)
- `POST /billing/guest-checkout-session` — JSON `{ "plan": "single" \| "debug", "email": "…" }` + **X-Guest-Billing-Id** → one-time Paddle checkout without an account
- `POST /billing/guest-claim-transaction` — JSON `{ "transaction_id": "txn_…" }` + **X-Guest-Billing-Id** → verifies payment with Paddle and grants guest credits (webhook backup)
- `POST /billing/user-claim-transaction` — same JSON + **Bearer** → grants **single** / **debug** job credits to the signed-in user (webhook backup for `/pay` overlay)
- `POST /billing/portal-session` — `{ "url" }` for **Paddle customer portal** (subscriptions, etc.); requires a prior successful checkout so a Paddle customer exists
- `POST /billing/webhook` — Paddle notifications (raw body; header `Paddle-Signature`)

After a successful pipeline run, the server decrements credits or increments free usage according to `billing_consumption` (`free`, `credit`, `sub_quota`, `guest_free`, `guest_credit`).

### Paddle setup (short)

1. In Paddle, create catalog **prices**: one-time **single**; optional one-time **debug** (≥ **$0.70 USD** — Paddle minimum) for `PADDLE_PRICE_DEBUG`; recurring **month**, **every 6 months**, **every 12 months** (year). Copy each **Price ID** into the `PADDLE_PRICE_*` env vars.
2. Under **Developer tools** → **Notifications**, add destination URL `https://<your-api>/billing/webhook` and subscribe to at least: **`transaction.completed`** (or **`transaction.paid`** for faster provisioning — the server handles both idempotently), **`subscription.created`**, **`subscription.updated`**, **`subscription.activated`**, **`subscription.canceled`**. Copy the signing secret into `PADDLE_WEBHOOK_SECRET`.
3. **Live only (typical):** use your **live** API key, **live** client token on the frontend, leave `PADDLE_SANDBOX` unset or `0`, and **live** price IDs. Sandbox is optional and only if you explicitly create a sandbox seller account.

### Production hardening (API)

- **CORS:** Set `FRONTEND_URL` (single origin) or `CORS_ORIGINS` (comma-separated, no trailing slashes). If both are empty, the API allows `*` (fine for local dev only). **Custom domain:** After moving the SPA (e.g. from a default Netlify URL to `https://chatreconstruct.com`), set `CORS_ORIGINS` on the API host to **every** origin visitors use — often both apex and `https://www.…` if you serve both. If `FRONTEND_URL` / `CORS_ORIGINS` does not match the `Origin` header, the browser hides the response from JavaScript as **“Failed to fetch”** (e.g. auth **“Could not load provider config”**). A `CORS_ORIGINS` value that parses to no origins (such as a lone comma) has the same effect; fix the env var and redeploy / restart the API.
- **Rate limits:** In-memory per IP on `POST` (`RATE_LIMIT_*` env vars; disable with `RATE_LIMIT_ENABLED=0`). `/billing/webhook` has a high limit; tune if Paddle shares egress IPs.
- **Job caps:** `MAX_JOB_FILES` (default 30), `MAX_JOB_UPLOAD_MB` (default 80 total per job).
- **PostgreSQL:** Use Render Postgres (**translate-chat-db**) or another host; set `DATABASE_URL` on the API. Backups: provider snapshots or `pg_dump`.
- **Monitoring:** Optional `SENTRY_DSN` + `SENTRY_TRACES_SAMPLE_RATE`. Set `MONITOR_READ_TOKEN` to enable `GET /monitor/activity` and `GET /monitor/usage`. The usage report is stored in its own PostgreSQL table and tracks live counters such as total algorithm runs, free-trial attempts, users signed up today, and total users. Paddle webhooks log to stdout at `INFO` (`translate_chat.billing`).
- **Windows fetch command:** after setting `MONITOR_READ_TOKEN`, run:
```powershell
Invoke-RestMethod -Uri "https://YOUR_BACKEND_URL/monitor/usage" -Headers @{ "X-Monitor-Token" = "YOUR_MONITOR_READ_TOKEN" } | ConvertTo-Json -Depth 10
```
- **Legal:** `GET /legal/terms`, `GET /legal/privacy` (set `PUBLIC_CONTACT_EMAIL`).
