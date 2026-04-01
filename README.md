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

1. Put screenshots into `input_images/`
2. Update `pass1_bubble_input.txt` if needed
3. Run `python main.py`
4. Check `rendered_chat/`, `result/`, and `result_json/`

## Goal

The main goal of the project is accurate conversation reconstruction and translation, especially in difficult cases where OCR is noisy, chat UI artifacts are present, or subject/reference resolution is ambiguous.

## HTTP API & billing (Stripe)

The FastAPI app (`web_app.py`) exposes translation jobs plus optional **Stripe** billing.

### Entitlements (SQLite)

Stored in the same database file as users (`data/users.sqlite3` by default, or `USER_DB_PATH`):

- `free_runs_used` / cap of 3 single-image runs (when not on a pass and no credits)
- `paid_job_credits` (one-time “full run” purchases)
- `access_until` (ISO timestamp) for active subscription or day-pass style access
- `stripe_customer_id`, `stripe_subscription_id`

**Guest (anonymous) rows** in `billing_guest_entitlements`: same 3-run / 1-image free cap, keyed by `X-Guest-Billing-Id` (8–64 hex chars).

### Environment variables

| Variable | Purpose |
|----------|---------|
| `BILLING_ENFORCE` | Set to `1` to enforce quotas on `POST /jobs`: signed-in users use Bearer; guests send **X-Guest-Billing-Id** (free tier only). |
| `REQUIRE_AUTH_FOR_JOBS` | `1` = user jobs need Bearer; **guest** jobs need the same **X-Guest-Billing-Id** on poll/artifact requests. |
| `STRIPE_SECRET_KEY` | Stripe secret API key |
| `STRIPE_WEBHOOK_SECRET` | Signing secret from Stripe Dashboard → Webhooks |
| `STRIPE_PRICE_SINGLE` | Price ID for one-time full-run credit |
| `STRIPE_PRICE_DAY` | Price ID for 24h pass (one-time payment) |
| `STRIPE_PRICE_MONTH` | Price ID for monthly subscription |
| `STRIPE_PRICE_SIXMO` | Price ID for every-6-month subscription |
| `FRONTEND_URL` | SPA origin for Checkout success/cancel redirects (no trailing slash) |

### Endpoints

- `GET /billing/status` — whether Stripe/prices are configured (no secrets)
- `GET /billing/guest-status` — guest free-tier counts (**header: X-Guest-Billing-Id**)
- `GET /billing/me` — current entitlements (**Authorization: Bearer** required)
- `POST /billing/checkout-session` — JSON `{ "plan": "single" \| "day" \| "month" \| "sixmo" }` → `{ "url" }` for Stripe Checkout
- `POST /billing/portal-session` — `{ "url" }` for **Stripe Customer Portal** (manage card, cancel subscription); requires prior checkout so a Stripe Customer exists
- `POST /billing/webhook` — raw Stripe webhook (configure URL in Dashboard)

After a successful pipeline run, the server decrements credits or increments free usage according to the `billing_consumption` value stored on the job (or guest free runs for `guest_free`).

### Stripe setup (short)

1. Create Products/Prices in Stripe for the four plans; copy each **Price ID** into the env vars above.
2. In Stripe Dashboard → **Customer portal**, enable the features you want (cancel, update payment method, etc.).
3. Add webhook endpoint `https://<your-api>/billing/webhook` and subscribe to: `checkout.session.completed`, `customer.subscription.updated`, `customer.subscription.deleted`, **`invoice.paid`** (keeps `access_until` aligned on subscription renewals).
4. Local testing: `stripe listen --forward-to localhost:8000/billing/webhook` and use the printed signing secret as `STRIPE_WEBHOOK_SECRET`.
