"""FastAPI server for the ElectricianSchedulingEnv (single-instance, deterministic).

This avoids session mismatch issues where /reset and /step hit different env instances.
"""

import os
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
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
def reset(body: ResetBody) -> Dict[str, Any]:
    obs = ENV.reset(task_name=body.task_name, seed=body.seed)
    # Keep HTTP shape compatible with what you already have:
    return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}


@app.post("/step")
def step(body: StepBody) -> Dict[str, Any]:
    obs = ENV.step(body.action)
    return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}


@app.get("/state")
def state() -> Dict[str, Any]:
    # ElectricianSchedulingEnv currently exposes state as a property returning ElectricianState (pydantic model)
    st = ENV.state
    try:
        return st.model_dump()
    except Exception:
        # fallback
        return {"episode_id": getattr(st, "episode_id", None), "step_count": getattr(st, "step_count", None)}

@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    # Minimal OpenEnv-style metadata
    return {
        "name": "electrician_scheduling",
        "runtime": "fastapi",
        "version": "1.0",
        "tasks": ["easy", "medium", "hard"],
    }


@app.get("/schema")
def schema() -> Dict[str, Any]:
    # Minimal schema endpoint for validators/clients that expect it.
    # We keep it simple: action is arbitrary JSON dict; observation matches ElectricianObservation.
    return {
        "action": {"type": "object", "description": "Action JSON dict (flat)."},
        "observation": {"type": "ElectricianObservation"},
    }


@app.get("/mcp")
def mcp() -> Dict[str, Any]:
    # Stub endpoint (some OpenEnv servers expose it)
    return {"status": "ok", "detail": "mcp not implemented in this environment"}
def main() -> None:
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
