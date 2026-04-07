"""Pydantic models for the ElectricianSchedulingEnv."""

from typing import Optional, List, Dict, Any, Literal, Union

from pydantic import BaseModel, Field, ConfigDict
from openenv_core import Observation, Action, State


class Ticket(BaseModel):
    ticket_id: str
    location: tuple
    category: str
    urgency: int  # 1-5
    created_at: str
    sla_deadline: str
    description: str
    status: str = "open"  # open, scheduled, resolved


class Electrician(BaseModel):
    id: str
    name: str
    home_base: tuple
    skills: List[str]
    rating: float
    hourly_cost: float
    availability: List[str]


class Appointment(BaseModel):
    appointment_id: str
    ticket_id: str
    electrician_id: str
    start_time: str
    duration_hours: int = 2
    status: str = "confirmed"  # confirmed, proposed, rescheduled, cancelled

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Action sub-types
# ---------------------------------------------------------------------------

class ListTicketsAction(Action):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
    type: Literal["list_tickets"] = "list_tickets"


class ViewTicketAction(Action):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
    type: Literal["view_ticket"] = "view_ticket"
    ticket_id: str


class ListElectriciansAction(Action):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
    type: Literal["list_electricians"] = "list_electricians"


class ProposeAppointmentAction(Action):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
    type: Literal["propose_appointment"] = "propose_appointment"
    ticket_id: str
    electrician_id: str
    start_time: str  # "YYYY-MM-DD HH:MM"


class ConfirmAppointmentAction(Action):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
    type: Literal["confirm_appointment"] = "confirm_appointment"
    appointment_id: str


class RescheduleAppointmentAction(Action):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
    type: Literal["reschedule_appointment"] = "reschedule_appointment"
    appointment_id: str
    new_start_time: str
    new_electrician_id: Optional[str] = None


class FinalizeAction(Action):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
    type: Literal["finalize"] = "finalize"


class NoopAction(Action):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
    type: Literal["noop"] = "noop"


# Discriminated union of all action types
ElectricianAction = Union[
    ListTicketsAction,
    ViewTicketAction,
    ListElectriciansAction,
    ProposeAppointmentAction,
    ConfirmAppointmentAction,
    RescheduleAppointmentAction,
    FinalizeAction,
    NoopAction,
]


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class ElectricianObservation(Observation):
    """Observation returned by ElectricianSchedulingEnv.step() and reset()."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    task_name: str = ""
    step_number: int = 0
    tickets: List[Ticket] = Field(default_factory=list)
    electricians: List[Electrician] = Field(default_factory=list)
    appointments: List[Appointment] = Field(default_factory=list)
    last_action_result: str = ""
    last_action_error: Optional[str] = None
    available_actions: List[str] = Field(default_factory=list)
    score: float = 0.0


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ElectricianState(State):
    """Internal mutable state for ElectricianSchedulingEnv."""

    task_name: str = ""
    tickets: List[Dict[str, Any]] = Field(default_factory=list)
    electricians: List[Dict[str, Any]] = Field(default_factory=list)
    appointments: List[Dict[str, Any]] = Field(default_factory=list)
    reschedule_count: int = 0
    noop_count: int = 0
    invalid_action_count: int = 0
    cumulative_reward: float = 0.0
    finalized: bool = False
