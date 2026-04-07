"""Main environment for electrician scheduling/dispatch."""

import math
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv_core import Environment

from .data import ELECTRICIANS_DATA, TICKETS_DATA
from .models import (
    Appointment,
    ConfirmAppointmentAction,
    Electrician,
    ElectricianObservation,
    ElectricianState,
    FinalizeAction,
    ListElectriciansAction,
    ListTicketsAction,
    NoopAction,
    ProposeAppointmentAction,
    RescheduleAppointmentAction,
    Ticket,
    ViewTicketAction,
)
from .tasks import GRADERS, TASKS

AVAILABLE_ACTIONS_TEXT = [
    "list_tickets",
    "view_ticket(ticket_id)",
    "list_electricians",
    "propose_appointment(ticket_id, electrician_id, start_time)",
    "confirm_appointment(appointment_id)",
    "reschedule_appointment(appointment_id, new_start_time, new_electrician_id?)",
    "finalize()",
    "noop()",
]


def _haversine_km(loc1: tuple, loc2: tuple) -> float:
    R = 6371.0
    lat1, lon1 = math.radians(loc1[0]), math.radians(loc1[1])
    lat2, lon2 = math.radians(loc2[0]), math.radians(loc2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


class ElectricianSchedulingEnv(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._task_name: str = "easy"
        self._tickets: List[Dict] = []
        self._electricians: List[Dict] = []
        self._appointments: List[Dict] = []
        self._pending_appointments: List[Dict] = []
        self._appt_counter: int = 0
        self._max_steps: int = 12
        self._reschedule_count: int = 0
        self._noop_count: int = 0
        self._invalid_count: int = 0
        self._cumulative_reward: float = 0.0
        self._finalized: bool = False
        self._internal_state = ElectricianState(episode_id=str(uuid4()), step_count=0)

        # Track parsed actions for graders/debugging (especially useful in "hard")
        self._action_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: str = "easy",
        **kwargs: Any,
    ) -> ElectricianObservation:
        random.seed(seed or 42)
        self._task_name = task_name
        task = TASKS.get(task_name, TASKS["easy"])
        self._max_steps = task["max_steps"]

        self._tickets = deepcopy(TICKETS_DATA)
        self._electricians = deepcopy(ELECTRICIANS_DATA)
        self._appointments = []
        self._pending_appointments = []
        self._appt_counter = 0
        self._reschedule_count = 0
        self._noop_count = 0
        self._invalid_count = 0
        self._cumulative_reward = 0.0
        self._finalized = False
        self._action_history = []

        # Medium task: pre-create an appointment for T001 that needs rescheduling
        if task_name == "medium":
            existing_appt = {
                "appointment_id": "A001",
                "ticket_id": "T001",
                "electrician_id": "E002",
                "start_time": "2024-01-15 09:00",
                "duration_hours": 2,
                "status": "confirmed",
                "rescheduled": False,
            }
            self._appointments.append(existing_appt)
            for t in self._tickets:
                if t["ticket_id"] == "T001":
                    t["status"] = "scheduled"

        self._internal_state = ElectricianState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_name=task_name,
            tickets=self._tickets,
            electricians=self._electricians,
            appointments=self._appointments,
        )

        return self._make_observation(
            result=f"Environment reset for task: {task_name}. {task['description']}",
            error=None,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: Any, **kwargs: Any) -> ElectricianObservation:
        self._internal_state.step_count += 1
        step = self._internal_state.step_count

        # Unwrap HTTP payload shape used by openenv_core FastAPI app: {"action": {...}}
        if isinstance(action, dict) and "action" in action and isinstance(action["action"], dict):
            action = action["action"]

        if isinstance(action, dict):
            action = self._parse_action(action)
        elif not isinstance(
            action,
            (
                ListTicketsAction,
                ViewTicketAction,
                ListElectriciansAction,
                ProposeAppointmentAction,
                ConfirmAppointmentAction,
                RescheduleAppointmentAction,
                FinalizeAction,
                NoopAction,
            ),
        ):
            # Handle pydantic model wrappers from the HTTP server (e.g. ElectricianActionWrapper)
            try:
                action = self._parse_action(action.model_dump(exclude_none=False))
            except Exception:
                action = NoopAction()

        # Record action history (best effort)
        try:
            self._action_history.append(action.model_dump())
        except Exception:
            pass

        if self._finalized:
            obs = self._make_observation(result="Episode already finalized.", error="Episode done")
            obs.done = True
            obs.reward = 0.0
            return obs

        reward = -0.05  # per-step penalty
        error: Optional[str] = None
        result = ""

        try:
            if isinstance(action, ListTicketsAction):
                result = "Tickets: " + str(
                    [
                        f"{t['ticket_id']} [{t['category']}, urgency={t['urgency']}, status={t['status']}]"
                        for t in self._tickets
                    ]
                )
                reward += 0.01

            elif isinstance(action, ViewTicketAction):
                ticket = next((t for t in self._tickets if t["ticket_id"] == action.ticket_id), None)
                if not ticket:
                    error = f"Ticket {action.ticket_id} not found"
                    reward -= 0.1
                else:
                    result = (
                        f"Ticket {ticket['ticket_id']}: {ticket['description']} | "
                        f"Category: {ticket['category']} | Urgency: {ticket['urgency']} | "
                        f"Status: {ticket['status']} | Location: {ticket['location']} | "
                        f"SLA: {ticket['sla_deadline']}"
                    )
                    reward += 0.02

            elif isinstance(action, ListElectriciansAction):
                result = "Electricians: " + str(
                    [
                        f"{e['id']} [{','.join(e['skills'])}, rating={e['rating']}]"
                        for e in self._electricians
                    ]
                )
                reward += 0.01

            elif isinstance(action, ProposeAppointmentAction):
                r, result, error = self._handle_propose(action)
                reward += r

            elif isinstance(action, ConfirmAppointmentAction):
                r, result, error = self._handle_confirm(action)
                reward += r

            elif isinstance(action, RescheduleAppointmentAction):
                r, result, error = self._handle_reschedule(action)
                reward += r

            elif isinstance(action, FinalizeAction):
                r, result, error = self._handle_finalize()
                reward += r

            elif isinstance(action, NoopAction):
                self._noop_count += 1
                result = "No operation performed."
                if self._noop_count > 2:
                    reward -= 0.15

            else:
                self._invalid_count += 1
                error = f"Unknown action type: {type(action)}"
                reward -= 0.2

        except (ValueError, KeyError, AttributeError, TypeError) as exc:
            self._invalid_count += 1
            error = str(exc)
            reward -= 0.2

        self._cumulative_reward += reward

        done = self._finalized or step >= self._max_steps
        if step >= self._max_steps and not self._finalized:
            grader = GRADERS.get(self._task_name, GRADERS["easy"])
            final_score = grader(self._get_state_dict(), self._action_history)
            reward += final_score * 0.5
            self._finalized = True
            result = f"Max steps reached. Final score: {final_score:.2f}"
            done = True

        self._update_internal_state()

        obs = self._make_observation(result=result, error=error)
        obs.done = done
        obs.reward = reward
        obs.score = self._cumulative_reward
        return obs

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_propose(self, action: ProposeAppointmentAction):
        ticket = next((t for t in self._tickets if t["ticket_id"] == action.ticket_id), None)
        elec = next((e for e in self._electricians if e["id"] == action.electrician_id), None)

        if not ticket:
            return -0.2, "", f"Ticket {action.ticket_id} not found"
        if not elec:
            return -0.2, "", f"Electrician {action.electrician_id} not found"
        if ticket["status"] == "resolved":
            return -0.1, "", f"Ticket {action.ticket_id} is already resolved"

        reward = 0.0
        warnings: List[str] = []

        if ticket["category"] not in elec["skills"]:
            warnings.append(f"Electrician lacks required skill '{ticket['category']}'")
            reward -= 0.1
        else:
            reward += 0.1

        if action.start_time not in elec["availability"]:
            warnings.append(f"Start time {action.start_time} not in electrician's availability")
            reward -= 0.05
        else:
            reward += 0.1

        for appt in self._appointments:
            if (
                appt["electrician_id"] == elec["id"]
                and appt["start_time"] == action.start_time
                and appt["status"] == "confirmed"
            ):
                return -0.15, "", f"Electrician {elec['id']} already booked at {action.start_time}"

        self._appt_counter += 1
        appt_id = f"PA{self._appt_counter:03d}"
        pending: Dict[str, Any] = {
            "appointment_id": appt_id,
            "ticket_id": action.ticket_id,
            "electrician_id": action.electrician_id,
            "start_time": action.start_time,
            "duration_hours": 2,
            "status": "proposed",
            "rescheduled": False,
        }
        self._pending_appointments.append(pending)

        msg = (
            f"Appointment proposed: {appt_id} for {action.ticket_id} "
            f"with {action.electrician_id} at {action.start_time}."
        )
        if warnings:
            msg += " Warnings: " + "; ".join(warnings)

        return reward + 0.05, msg, None

    def _handle_confirm(self, action: ConfirmAppointmentAction):
        pending = next(
            (a for a in self._pending_appointments if a["appointment_id"] == action.appointment_id),
            None,
        )
        if not pending:
            return (
                -0.15,
                "",
                f"Proposed appointment {action.appointment_id} not found. "
                "Use propose_appointment first.",
            )

        pending["status"] = "confirmed"
        self._appointments.append(pending)
        self._pending_appointments = [
            a for a in self._pending_appointments if a["appointment_id"] != action.appointment_id
        ]

        for t in self._tickets:
            if t["ticket_id"] == pending["ticket_id"]:
                t["status"] = "scheduled"

        elec = next((e for e in self._electricians if e["id"] == pending["electrician_id"]), None)
        ticket = next((t for t in self._tickets if t["ticket_id"] == pending["ticket_id"]), None)
        reward = 0.3
        if elec and ticket and ticket["category"] in elec["skills"]:
            reward += 0.1
        if elec and pending["start_time"] in elec["availability"]:
            reward += 0.1

        return reward, f"Appointment {action.appointment_id} confirmed for ticket {pending['ticket_id']}.", None

    def _handle_reschedule(self, action: RescheduleAppointmentAction):
        appt = next(
            (
                a
                for a in self._appointments
                if a["appointment_id"] == action.appointment_id
                and a["status"] in ("confirmed", "rescheduled")
            ),
            None,
        )
        if not appt:
            return -0.15, "", f"Confirmed appointment {action.appointment_id} not found"

        if self._reschedule_count >= 3:
            return -0.2, "", "Maximum reschedules (3) reached"

        new_elec_id = action.new_electrician_id or appt["electrician_id"]
        elec = next((e for e in self._electricians if e["id"] == new_elec_id), None)
        if not elec:
            return -0.2, "", f"Electrician {new_elec_id} not found"

        if action.new_start_time not in elec["availability"]:
            return -0.1, "", f"New time slot {action.new_start_time} not in electrician's availability"

        for other in self._appointments:
            if (
                other["appointment_id"] != action.appointment_id
                and other["electrician_id"] == new_elec_id
                and other["start_time"] == action.new_start_time
                and other["status"] in ("confirmed", "rescheduled")
            ):
                return -0.15, "", f"Conflict: {new_elec_id} already booked at {action.new_start_time}"

        old_time = appt["start_time"]
        appt["start_time"] = action.new_start_time
        appt["electrician_id"] = new_elec_id
        appt["rescheduled"] = True
        appt["status"] = "rescheduled"
        self._reschedule_count += 1

        reward = 0.2
        ticket = next((t for t in self._tickets if t["ticket_id"] == appt["ticket_id"]), None)
        if ticket and ticket["category"] in elec["skills"]:
            reward += 0.1

        return (
            reward,
            f"Appointment {action.appointment_id} rescheduled from {old_time} to {action.new_start_time}.",
            None,
        )

    def _handle_finalize(self):
        grader = GRADERS.get(self._task_name, GRADERS["easy"])
        final_score = grader(self._get_state_dict(), self._action_history)
        reward = final_score * 1.0
        self._finalized = True
        return reward, f"Finalized. Task score: {final_score:.2f}", None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_action(self, d: dict):
        action_type = d.get("type", "noop")
        fields = {k: v for k, v in d.items() if k not in ("metadata",)}
        if action_type == "list_tickets":
            return ListTicketsAction()
        if action_type == "view_ticket":
            return ViewTicketAction(ticket_id=fields.get("ticket_id", ""))
        if action_type == "list_electricians":
            return ListElectriciansAction()
        if action_type == "propose_appointment":
            return ProposeAppointmentAction(
                ticket_id=fields.get("ticket_id", ""),
                electrician_id=fields.get("electrician_id", ""),
                start_time=fields.get("start_time", ""),
            )
        if action_type == "confirm_appointment":
            return ConfirmAppointmentAction(appointment_id=fields.get("appointment_id", ""))
        if action_type == "reschedule_appointment":
            return RescheduleAppointmentAction(
                appointment_id=fields.get("appointment_id", ""),
                new_start_time=fields.get("new_start_time", ""),
                new_electrician_id=fields.get("new_electrician_id"),
            )
        if action_type == "finalize":
            return FinalizeAction()
        return NoopAction()

    def _get_state_dict(self) -> dict:
        return {
            "task_name": self._task_name,
            "tickets": self._tickets,
            "electricians": self._electricians,
            "appointments": self._appointments,
            "pending_appointments": self._pending_appointments,
            "reschedule_count": self._reschedule_count,
            "noop_count": self._noop_count,
            "invalid_count": self._invalid_count,
            "cumulative_reward": self._cumulative_reward,
            "finalized": self._finalized,
        }

    def _update_internal_state(self) -> None:
        self._internal_state.task_name = self._task_name
        self._internal_state.tickets = self._tickets
        self._internal_state.electricians = self._electricians
        self._internal_state.appointments = self._appointments
        self._internal_state.reschedule_count = self._reschedule_count
        self._internal_state.noop_count = self._noop_count
        self._internal_state.invalid_action_count = self._invalid_count
        self._internal_state.cumulative_reward = self._cumulative_reward
        self._internal_state.finalized = self._finalized

    def _make_observation(self, result: str = "", error: Optional[str] = None) -> ElectricianObservation:
        return ElectricianObservation(
            task_name=self._task_name,
            step_number=self._internal_state.step_count,
            tickets=[Ticket(**t) for t in self._tickets],
            electricians=[Electrician(**e) for e in self._electricians],
            appointments=[Appointment(**a) for a in self._appointments],
            last_action_result=result,
            last_action_error=error,
            available_actions=AVAILABLE_ACTIONS_TEXT,
            score=self._cumulative_reward,
            done=self._finalized,
            reward=0.0,
        )

    @property
    def state(self) -> ElectricianState:
        return self._internal_state
