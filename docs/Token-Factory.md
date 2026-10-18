# AMD Token Factory — Documentation

## Overview

Token Factory is the model API catalog of AMD Radeon Cloud. It gives developers free, OpenAI- and Anthropic-compatible LLM endpoints running on AMD GPUs, plus one-click dedicated model deployments paid for with credits.

The catalog:

### Public Free Model APIs

| | |
|---|---|
| **What it is** | Shared endpoints, always on, run by AMD |
| **Cost** | Free (a daily usage cap applies) |
| **Models** | AMD chooses them, and the list changes over time |
| **Base URL** | `https://developer.amd.com.cn/radeon/api/v1` |
| **Endpoints** | Chat completions, Messages (Anthropic), models list, token counting, OCR |
| **Limits** | Per-key rate, concurrency and spend caps |

One API key (format `rc-` + 48 hex characters) works for every shared model. Any OpenAI client (curl, the openai SDK, LangChain, Cherry Studio) works by changing the base URL and key. Anthropic clients such as Claude Code also work through `/v1/messages`.

## Getting started — redeeming your team API key

Your team has been pre-provisioned with an API key for this event. To reveal it, every member of your team must sign in and request redemption. The key is only shown once all members have redeemed.

1. **Sign in.** Open `notebooks.amd.com/tokenfactory`. You will be redirected to the AMD SSO login page. Sign in with your AMD AI Developer Program account (the same email your team was registered with).
2. **Check your team dashboard.** After sign-in you will see your team name, your email, the team size, and a list of all team members with their redemption status (redeemed or waiting).
3. **Click "Redeem API Key".** This marks your email as redeemed. If other team members have not redeemed yet, the page shows who is still pending.
4. **Wait for all members.** The API key is not revealed until every team member has signed in and clicked "Redeem API Key". Coordinate with your teammates to ensure everyone completes this step. Click "Check Status" to refresh the redemption list.
5. **Copy your API key.** Once all members have redeemed, the key and this documentation are displayed. Click "Copy to clipboard" to copy the key.

> **Important — shared rate limits.** Your entire team shares a single API key. All rate limits — RPM (Requests Per Minute), concurrent request cap, and the daily spend cap — are enforced per key, not per person. That means every call from every teammate counts toward the same limits (see the Rate limits table below). Coordinate with your team: avoid burst-sending requests at the same time, stagger your calls, and use exponential backoff on 429 responses. If one member saturates the RPM limit, everyone else on the team gets throttled too.

6. **Make a call.**

```bash
curl https://developer.amd.com.cn/radeon/api/v1/chat/completions \
  -H "Authorization: Bearer $RADEON_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"DeepSeek-V4-Flash","messages":[{"role":"user","content":"Say hello in one sentence."}]}'
```

From Python with the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="https://developer.amd.com.cn/radeon/api/v1", api_key="rc-...")
resp = client.chat.completions.create(
    model="DeepSeek-V4-Flash",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(resp.choices[0].message.content)
```

From the Anthropic SDK, set `base_url` to `https://developer.amd.com.cn/radeon/api`, without `/v1`, because the SDK appends `/v1/messages` itself.

7. **Check your remaining usage.** Every API response includes `X-RateLimit-*` headers showing your remaining requests and spend. You can also check explicitly:

```bash
curl https://developer.amd.com.cn/radeon/api/v1/usage?include_recent=true \
  -H "Authorization: Bearer $RADEON_API_KEY"
```

This returns your total quota, used amount, remaining amount, reset time, RPM limit, and recent request details. Since the whole team shares one key, this reflects combined usage across all teammates.

**Account review.** Some new accounts are held for verification. While an account is in review, it is not issued an API key and cannot launch instances. If your key is missing, check your account's review status.

## Models and pricing

All 10 shared models are free and marked experimental. The usage figures below are in points per 1M tokens and only show relative usage: "Free to use. Points show relative usage—not a charge." They count against a daily spend cap, which the API reports in USD.

### Public Free Model APIs

| Model ID | Publisher | Type | Context (tokens) | Input | Thinks by default | Input / Output / Cache read (pts per 1M) |
|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash (weights: V4-Flash-0731) | DeepSeek | LLM | 1,048,576 | Text | No | 0.14 / 0.28 / 0.0028 |
| DeepSeek-V4.1-Flash | DeepSeek | VLM | 1,048,576 | Text, image | No | 0.14 / 0.28 / 0.0028 |
| DeepSeek-V4-Flash-Vision-Exp | DeepSeek | VLM | 1,048,576 | Text, image | No | 0.14 / 0.28 / 0.0028 |
| MiMo-V2.6-Flash | Xiaomi | VLM | 1,048,576 | Text, image, audio | Yes | 0.14 / 0.28 / 0.0028 |
| GLM-5.3-Flash | Z.ai | LLM | 262,144 | Text | Yes | 0.15 / 0.50 / 0.03 |
| Qwen3.8-Flash-Next | Qwen | VLM | 262,144 | Text, image | Yes | 0.15 / 0.47 / 0.016 |
| Qwen3.8-27B | Qwen | VLM | 131,072 | Text, image | Yes | 0.50 / 3.00 / 0.05 |
| MiniCPM5-2B | OpenBMB | LLM | 131,072 | Text | No (no thinking) | 0.124 / 0.7425 / 0.124 |
| Gemma-4-31B-IT (experimental) | Google | VLM | 262,144 | Text, image | No | 0.50 / 3.00 / 0.05 |
| MinerU2.5-Pro | Radeon Cloud | Document OCR | n/a | PDF or image | n/a | "Limited Free", billed per page |

Every chat model supports streaming, tool calling (`tools`, `tool_choice`) and `response_format: {"type":"json_object"}`. MiMo-V2.6-Flash is the only one that also accepts strict `json_schema` and audio input. MinerU2.5-Pro uses its own endpoint, `POST /v1/ocr`.

The catalog changes as models are added and retired, so call `GET /v1/models` at runtime instead of hard-coding names.

## API reference

The shared endpoints sit behind a gateway. The gateway validates each request against a fixed set of fields and silently drops anything else. Every path also works under `/api/v1/...`.

### Endpoints (base URL `https://developer.amd.com.cn/radeon/api`)

| Method and path | Purpose | Auth |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible chat (the main endpoint) | `Authorization: Bearer rc-...` |
| `POST /v1/messages` | Anthropic Messages API, for Claude Code and the anthropic SDK | Bearer, or `x-api-key` + `anthropic-version: 2023-06-01` |
| `POST /v1/messages/count_tokens` | Token count estimate (for sizing only, not billing) | Bearer or `x-api-key` |
| `GET /v1/models` | Catalog with pricing, context length and capability flags | Bearer or session |
| `POST /v1/ocr` | MinerU2.5-Pro: a PDF or image goes in and Markdown per page comes out | Bearer |

The shared endpoints do not serve legacy completions, embeddings, image, audio, rerank or `/v1/responses`. Those paths return 404.

### Chat completions parameters

- **Accepted:** `model`, `messages`, `stream`, `temperature`, `top_p`, `max_tokens`, `presence_penalty`, `frequency_penalty`, `response_format`, `tools`, `tool_choice`, `parallel_tool_calls`, `reasoning_effort` (or `reasoning.effort`).
- **Silently dropped:** `stop`, `seed`, `logit_bias`, `logprobs`, `top_logprobs`, `top_k`, `min_p`, `repetition_penalty`. If you need these, use a dedicated endpoint.
- **Rejected with 400:** `thinking: {...}` and `reasoning.enabled`. Use `reasoning_effort` instead. On `/v1/messages`, use `output_config: {"effort": "high"}`.
- **Timeout:** 10 minutes for a non-streaming request. When streaming, the 10 minutes is the maximum gap between chunks. Streams end with `data: [DONE]`.

### Thinking (reasoning) controls

Thinking text comes back in `choices[0].message.reasoning`, and its token count in `usage.completion_tokens_details.reasoning_tokens` (except Qwen3.8-27B). Thinking tokens count toward `max_tokens`, so a low limit can leave `content` empty.

| Model | Accepted `reasoning_effort` | If omitted | `system` message rules |
|---|---|---|---|
| DeepSeek-V4-Flash, V4.1-Flash, V4-Flash-Vision-Exp | none, minimal, low, medium, high, xhigh, max | No thinking | Any position, several allowed; `system` or `developer` |
| MiMo-V2.6-Flash | none + usual tiers (none turns it off) | Thinks | Any position; `developer` accepted |
| Qwen3.8-Flash-Next | none, low, medium, xhigh | Thinks (xhigh) | Exactly one, first; `system` only |
| Qwen3.8-27B | low, medium, xhigh (high = 400) | Thinks (xhigh) | At most one, first |
| GLM-5.3-Flash | low, medium, high (xhigh = 422) | Thinks | Any position; `system` only (`developer` = 422) |
| MiniCPM5-2B | Not applicable | No thinking | Any position; `system` |
| Gemma-4-31B-IT | On / Off only | No thinking | Use `system` as the first message |

One request shape that works on every chat model: a single `system` message at index 0, `reasoning_effort` of `low` or `medium` (or omitted), and text-only content unless the model takes images. Size `max_tokens` to the model's context window.

### Image, audio and OCR input

- **Images:** add an `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}` part to `content`. This works on DeepSeek-V4.1-Flash, V4-Flash-Vision-Exp, MiMo-V2.6-Flash and both Qwen3.8 models. GLM-5.3-Flash rejects images.
- **Audio (MiMo-V2.6-Flash only):** `{"type":"input_audio","input_audio":{"data":"<base64>","format":"wav"}}`.
- **OCR:** send `{"model":"MinerU2.5-Pro","document":{"type":"document_url","document_url":"https://..."},"pages":"0-2"}` to `/v1/ocr`. The response returns `pages[].markdown` and `usage_info.pages_processed`. It does not stream, so set a long client timeout.

### Rate limits (shared endpoints)

| Limit | Scope | Typical value |
|---|---|---|
| Requests per minute | Per API key (platform) | 30 |
| Requests per minute | Per account (gateway, 60 s sliding window) | 20 |
| Requests per minute | Per IP | 120 |
| Concurrent requests | Per API key | 8 |
| Spend cap | Per account, rolling daily period | Read it from `/v1/usage` |

Every response carries `X-RateLimit-*` headers with your remaining requests and spend. When you are limited, you get a 429 with `Retry-After` (60 s for per-minute limits, 1 s for concurrency limits). Retry only 429s, with exponential backoff and jitter. Dedicated endpoints have none of these limits.

### Usage and account endpoints

These Platform API paths use the same host as the model API. They accept a Bearer key.

- `GET /v1/usage?include_recent=true`: total quota, used amount, remaining, reset time, RPM limit and recent requests.
- `POST /api/profile/api-token`: rotate your key. The old key stops working immediately.
- `POST /api/credits/redeem` with `{"coupon": "..."}`: add credits.
- Instances: `POST /api/notebook/request`, `GET /api/notebook/status`, `DELETE /api/notebook/current`. Templates: `/api/profile/templates`.

### Errors

| Status | Meaning |
|---|---|
| 400 | Bad parameter, unknown model (`Requested model <name> not supported`), or not enough credits |
| 401 | Missing or invalid key |
| 403 | Account still in review (`account_not_verified`), or not your instance |
| 404 | The path is not served |
| 422 | Value rejected while the body was being parsed (for example `xhigh` on GLM-5.3-Flash) |
| 429 | Rate limited. Honour `Retry-After` |
| 502 / 503 | Gateway unreachable or saturated. Retry with backoff |

Gateway errors use the OpenAI shape (`{"error":{...}}`). Errors from the platform layer are wrapped in `{"detail":{"error":{...}}}`, so check both.

## FAQ, terms and support

### FAQ

- **Do I need an instance to use the models?** No. Shared endpoints are free and always on. You only need an instance for a shell, a notebook, or a model the catalog doesn't offer.
- **Can I share my key or endpoint with a teammate?** For event/hackathon teams using Token Factory, yes — each team shares one API key and the shared rate limits apply to the whole team. Outside of event-provisioned keys, the general Radeon Cloud policy is one key per person. A dedicated endpoint only accepts its owner's key.
- **Are embeddings available?** Not on the shared endpoints. A vLLM dedicated endpoint usually serves `/v1/embeddings`.
- **A model that used to work now returns 400 "not supported".** The catalog changed. Re-check `GET /v1/models`.
- **My response has empty content and `finish_reason: length`.** Thinking used up `max_tokens`. Raise the limit, or send `reasoning_effort: "low"` (or `"none"` on models that allow it).
- **Does a dedicated endpoint URL stay the same?** No. Each launch creates a new instance ID and a new base URL.

### API terms of use

- **Trial and development only.** The service has no SLA, uptime commitment or support obligation, and models can change or be withdrawn with little notice. Do not run production workloads on it.
- **Upstream models, unmodified.** The models are served from the original published weights, and each model's own licence applies. AMD gives no warranty for outputs.
- **Free.** Credits, quotas and points have no monetary value.
- **Shared team key.** Each event/hackathon team is provisioned one API key shared by all team members. Keys are disabled for selling, automated sign-ups, and use by multiple accounts outside the registered team.
- **No reselling.** You may not proxy, resell or re-badge the API, or tell a model to misreport which model it is.
- **Acceptable use.** No illegal content, malware, attacks, bypassing limits, CSAM, harassment or impersonation. Do not send sensitive personal, health or financial data.
- **Logging.** Prompts and outputs are kept for about 2 days, and metadata is kept longer. Data is not used for training and is not sold.

Contact and appeals: member-service@mail.developer.amd.com.cn