# Social Media OCR Translation Pipeline

This project extracts chat conversations from screenshots, reconstructs the conversation structure, and renders a translated chat-style output image.

It is currently optimized for Facebook Messenger-style conversations, but the pipeline is being kept modular so other chat platforms can be supported later.

## What It Does

Given a folder of input chat screenshots, the pipeline:

1. Cleans the images and prepares them for analysis.
2. Uses Gemini vision to transcribe the conversation structure from the screenshots.
3. Uses OCR hints to refine the source-language message text.
4. Runs a reference-resolution pass to improve who is speaking about whom before final English translation.
5. Extracts status-bar/header information separately.
6. Renders the final conversation as a clean chat image, along with debug comparison outputs.

## Current Pass Structure

1. `Pass 1`: source transcription and conversation structure
2. `Pass 2`: OCR-guided source-text polishing
3. `Pass 3`: reference resolution plus final English translation
4. `Pass 4`: status bar extraction for the rendered header

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

### Entitlements (SQLite)

Stored in the same database file as users (`data/users.sqlite3` by default, or `USER_DB_PATH`):

- `free_runs_used` / cap of **1** single-image free run per signed-in account (when not subscribed and no credits)
- `paid_job_credits` (one-time “full run” purchases)
- `access_until` (ISO timestamp) for active subscription billing period; monthly run quota is separate (`BILLING_SUBSCRIPTION_RUNS_PER_MONTH`)
- `paddle_customer_id`, `paddle_address_id`, `paddle_subscription_id` (legacy `stripe_*` columns may still exist from older installs and are migrated/read for compatibility)

**Guest (anonymous) rows** in `billing_guest_entitlements`: **1** free single-image run per `X-Guest-Billing-Id` (8–64 hex); `paid_job_credits` from one-time checkout via `POST /billing/guest-checkout-session` (webhook `custom_data.guest_key`); multi-image allowed when credits > 0.

### Environment variables

| Variable | Purpose |
|----------|---------|
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
- `POST /billing/portal-session` — `{ "url" }` for **Paddle customer portal** (subscriptions, etc.); requires a prior successful checkout so a Paddle customer exists
- `POST /billing/webhook` — Paddle notifications (raw body; header `Paddle-Signature`)

After a successful pipeline run, the server decrements credits or increments free usage according to `billing_consumption` (`free`, `credit`, `sub_quota`, `guest_free`, `guest_credit`).

### Paddle setup (short)

1. In Paddle, create catalog **prices**: one-time **single**; optional one-time **debug** (≥ **$0.70 USD** — Paddle minimum) for `PADDLE_PRICE_DEBUG`; recurring **month**, **every 6 months**, **every 12 months** (year). Copy each **Price ID** into the `PADDLE_PRICE_*` env vars.
2. Under **Developer tools** → **Notifications**, add destination URL `https://<your-api>/billing/webhook` and subscribe to at least: **`transaction.completed`**, **`subscription.created`**, **`subscription.updated`**, **`subscription.activated`**, **`subscription.canceled`**. Copy the signing secret into `PADDLE_WEBHOOK_SECRET`.
3. **Live only (typical):** use your **live** API key, **live** client token on the frontend, leave `PADDLE_SANDBOX` unset or `0`, and **live** price IDs. Sandbox is optional and only if you explicitly create a sandbox seller account.
