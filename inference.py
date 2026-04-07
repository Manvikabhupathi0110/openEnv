"""
Inference Script for ElectricianSchedulingEnv

Environment variables:
  API_BASE_URL:            LLM endpoint (default: https://router.huggingface.co/v1)
  MODEL_NAME:              Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN / API_KEY:      API key for the LLM
  LOCAL_IMAGE_NAME / IMAGE_NAME: Docker image name (optional)
"""

import json
import os
import re
import sys
from openai import OpenAI
from openai import APIConnectionError, APIStatusError, RateLimitError

from openenv_electrician.environment import ElectricianSchedulingEnv
from openenv_electrician.tasks import GRADERS

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "")
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME") or os.environ.get("IMAGE_NAME", "")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

SYSTEM_PROMPT = """You are an electrician dispatch agent. Your job is to schedule electricians for maintenance tickets.

Available actions (respond with ONLY valid JSON):
- {"type": "list_tickets"} - list all tickets
- {"type": "view_ticket", "ticket_id": "T001"} - view ticket details
- {"type": "list_electricians"} - list all electricians
- {"type": "propose_appointment", "ticket_id": "T001", "electrician_id": "E001", "start_time": "2024-01-15 09:00"} - propose appointment
- {"type": "confirm_appointment", "appointment_id": "PA001"} - confirm proposed appointment
- {"type": "reschedule_appointment", "appointment_id": "A001", "new_start_time": "2024-01-16 09:00", "new_electrician_id": "E001"} - reschedule
- {"type": "finalize"} - complete the task

Rules:
1. Match electrician skills to ticket category
2. Only use time slots from electrician's availability
3. Avoid double-booking
4. Always propose then confirm appointments
5. Call finalize when done
"""

MAX_STEPS = {"easy": 12, "medium": 15, "hard": 20}


def run_task(task_name: str) -> float:
    env = ElectricianSchedulingEnv()
    obs = env.reset(task_name=task_name)

    print(f"[START] task={task_name} env=electrician_scheduling model={MODEL_NAME}")
    sys.stdout.flush()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Task: {obs.last_action_result}\n\n"
                f"Current state:\n{obs.model_dump_json(indent=2)}\n\n"
                "Choose your next action as JSON:"
            ),
        },
    ]

    step = 0
    rewards = []
    max_steps = MAX_STEPS[task_name]

    while not obs.done and step < max_steps:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=256,
                temperature=0.1,
            )
            action_str = response.choices[0].message.content.strip()

            try:
                json_match = re.search(r"\{.*\}", action_str, re.DOTALL)
                action_dict = json.loads(json_match.group()) if json_match else {"type": "noop"}
            except (json.JSONDecodeError, AttributeError):
                action_dict = {"type": "noop"}
                action_str = '{"type": "noop"}'

            obs = env.step(action_dict)
            step += 1
            rewards.append(obs.reward or 0.0)

            error_str = obs.last_action_error if obs.last_action_error else "null"
            print(
                f"[STEP] step={step} action={json.dumps(action_dict)} "
                f"reward={obs.reward:.2f} done={str(obs.done).lower()} error={error_str}"
            )
            sys.stdout.flush()

            messages.append({"role": "assistant", "content": action_str})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Result: {obs.last_action_result}\n\n"
                        f"State:\n{obs.model_dump_json(indent=2)}\n\n"
                        "Next action:"
                    ),
                }
            )

        except (APIConnectionError, APIStatusError, RateLimitError, json.JSONDecodeError, ValueError) as exc:
            step += 1
            rewards.append(-0.05)
            error_msg = type(exc).__name__
            print(
                f"[STEP] step={step} action=null reward={-0.05:.2f} done=false error={error_msg}"
            )
            sys.stdout.flush()
            break

    grader = GRADERS[task_name]
    final_score = grader(env._get_state_dict(), [])

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success = final_score >= 0.5
    print(
        f"[END] success={str(success).lower()} steps={step} "
        f"score={final_score:.2f} rewards={rewards_str}"
    )
    sys.stdout.flush()

    return final_score


if __name__ == "__main__":
    scores = {}
    for task in ["easy", "medium", "hard"]:
        scores[task] = run_task(task)
    print(f"Final scores: {scores}", file=sys.stderr)
