"""FastAPI server for the ElectricianSchedulingEnv (single-instance, deterministic).

This avoids session mismatch issues where /reset and /step hit different env instances.
"""

import os
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import Body, FastAPI
from pydantic import BaseModel, Field

from openenv_electrician.environment import ElectricianSchedulingEnv


class ResetBody(BaseModel):
    task_name: str = Field(default="easy")
    seed: Optional[int] = Field(default=None)


class StepBody(BaseModel):
    action: Dict[str, Any]


app = FastAPI()

# Single global environment instance (persists across HTTP calls)
ENV = ElectricianSchedulingEnv()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(body: ResetBody | None = Body(default=None)) -> Dict[str, Any]:
    # Validator may send POST with no body at all.
    task_name = body.task_name if body else "easy"
    seed = body.seed if body else None

    obs = ENV.reset(task_name=task_name, seed=seed)
    return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}


@app.post("/step")
def step(body: Any = Body(default=None)) -> Dict[str, Any]:
    # Support both shapes:
    # 1) {"action": {...}}  (your current API)
    # 2) {...}             (raw action dict some validators use)
    action: Any
    if isinstance(body, dict) and "action" in body and isinstance(body["action"], dict):
        action = body["action"]
    else:
        action = body if body is not None else {"type": "noop"}

    obs = ENV.step(action)
    return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}


@app.get("/state")
def state() -> Dict[str, Any]:
    st = ENV.state
    try:
        return st.model_dump()
    except Exception:
        return {
            "episode_id": getattr(st, "episode_id", None),
            "step_count": getattr(st, "step_count", None),
        }


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    return {
        "name": "electrician_scheduling",
        "runtime": "fastapi",
        "version": "1.0",
        "tasks": ["easy", "medium", "hard"],
    }


@app.get("/schema")
def schema() -> Dict[str, Any]:
    return {
        "action": {"type": "object", "description": "Action JSON dict (flat)."},
        "observation": {"type": "ElectricianObservation"},
    }


@app.get("/mcp")
def mcp() -> Dict[str, Any]:
    return {"status": "ok", "detail": "mcp not implemented in this environment"}


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
