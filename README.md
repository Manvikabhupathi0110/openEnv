---
title: openenv-electrician-dispatch
emoji: ⚡
colorFrom: yellow
colorTo: blue
sdk: docker
pinned: false
---

# openEnv — Electrician Scheduling

An [OpenEnv](https://github.com/huggingface/openenv) environment that simulates an electrician scheduling/dispatch domain. An LLM agent interacts with a ticket queue, a roster of electricians, and an appointment calendar to optimally schedule maintenance work.

---

## Environment Overview

| Property | Value |
|---|---|
| **Name** | `electrician_scheduling` |
| **Runtime** | FastAPI (HTTP + WebSocket) |
| **Observation type** | `ElectricianObservation` |
| **Action type** | `ElectricianActionWrapper` (flat JSON) |
| **Episode ends when** | `finalize` action or max steps reached |

### Electricians

5 electricians with varied skills, ratings, hourly costs, and availability windows:

| ID | Name | Skills | Rating | Cost/hr |
|---|---|---|---|---|
| E001 | Raj Kumar | wiring, panel | 4.8 | $45 |
| E002 | Priya Sharma | lighting, wiring | 4.2 | $38 |
| E003 | Amit Singh | panel, generator | 3.9 | $35 |
| E004 | Deepa Nair | lighting, sockets | 4.5 | $42 |
| E005 | Vikram Rao | wiring, panel, generator | 4.7 | $50 |

### Tickets

8 maintenance tickets across categories: `wiring`, `lighting`, `panel`, `generator`, `sockets`.

---

## Action Space

Send a JSON object with a `type` field:

```json
{"type": "list_tickets"}
{"type": "view_ticket", "ticket_id": "T005"}
{"type": "list_electricians"}
{"type": "propose_appointment", "ticket_id": "T005", "electrician_id": "E001", "start_time": "2024-01-15 09:00"}
{"type": "confirm_appointment", "appointment_id": "PA001"}
{"type": "reschedule_appointment", "appointment_id": "A001", "new_start_time": "2024-01-16 09:00"}
{"type": "finalize"}
{"type": "noop"}
```

## Observation Space

```json
{
  "done": false,
  "reward": -0.05,
  "task_name": "easy",
  "step_number": 1,
  "tickets": [...],
  "electricians": [...],
  "appointments": [...],
  "last_action_result": "...",
  "last_action_error": null,
  "available_actions": [...],
  "score": -0.05
}
```

---

## Tasks

### Easy (`max_steps=12`)
Schedule an electrician for **T005** (electrical panel sparking — URGENT, urgency 5). Requires: skill match (`panel`), valid availability slot, propose + confirm.

**Full score (1.0):** Confirmed appointment with panel-skilled electrician in a valid slot.

### Medium (`max_steps=15`)
An appointment for **T001** (wiring, urgency 5) already exists but needs rescheduling to a better slot.

**Full score (1.0):** Successfully rescheduled to a valid slot with a `wiring`-skilled electrician.

### Hard (`max_steps=20`)
Optimally dispatch electricians to **T003** (generator), **T005** (panel), and **T008** (panel/kitchen) simultaneously. Score is a weighted combination of skill match, urgency, rating, proximity, and slot validity.

---

## Reward Shaping

| Event | Reward |
|---|---|
| Per step | −0.05 |
| Valid info action | +0.01–0.02 |
| Skill match in proposal | +0.10 |
| Valid slot in proposal | +0.10 |
| Proposal placed | +0.05 |
| Confirmation (base) | +0.30 |
| Confirmation with skill match | +0.10 bonus |
| Confirmation with valid slot | +0.10 bonus |
| Reschedule | +0.20 |
| Invalid action | −0.20 |
| Repeated noop (>2) | −0.15 |
| Terminal (finalize/max steps) | grader_score × 1.0 |

---

## Setup & Running

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Or run directly
python server/app.py
```

### Validate before submission
```bash
bash scripts/validate-submission.sh
```

### Run inference (requires LLM API key)
```bash
export OPENAI_API_KEY=your_key_here
python3 inference.py
```

### Docker
```bash
docker build -t electrician-env .
docker run -p 8000:8000 electrician-env
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/reset` | Start new episode (pass `task_name` in body) |
| POST | `/step` | Execute action |
| GET | `/state` | Inspect current state |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |

---

## Baseline Scores

| Task | Random agent | Optimal agent |
|---|---|---|
| easy | ~0.00 | 1.00 |
| medium | ~0.00 | 1.00 |
| hard | ~0.20 | 1.00 |

# Required (for HF router)
export HF_TOKEN=...
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python3 inference.py

#!/usr/bin/env bash
set -euo pipefail

SPACE_URL="${SPACE_URL:-https://manvikabhupathi-openenv-electrician-dispatch.hf.space}"

echo "Health:"
curl -s -o /dev/null -w "%{http_code}\n" "$SPACE_URL/health"

echo "Reset:"
curl -s -o /dev/null -w "%{http_code}\n" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$SPACE_URL/reset"

  
#Example output
[START] task=easy env=electrician_scheduling model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=null reward=-0.05 done=false error=APIStatusError
[END] success=false steps=1 score=0.000001 rewards=-0.05
[START] task=medium env=electrician_scheduling model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=null reward=-0.05 done=false error=APIStatusError
[END] success=false steps=1 score=0.400000 rewards=-0.05
[START] task=hard env=electrician_scheduling model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=null reward=-0.05 done=false error=APIStatusError
[END] success=false steps=1 score=0.000001 rewards=-0.05
Final scores: {'easy': 1e-06, 'medium': 0.4, 'hard': 1e-06}
(.venv)
