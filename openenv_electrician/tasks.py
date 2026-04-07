"""Task definitions and graders for the electrician scheduling environment."""

import math

EPS = 1e-6


def _strict01(x: float) -> float:
    """Clamp to the open interval (0, 1) to satisfy validators requiring strict bounds."""
    if x <= 0.0:
        return EPS
    if x >= 1.0:
        return 1.0 - EPS
    return float(x)


TASKS = {
    "easy": {
        "name": "easy",
        "description": (
            "Schedule an electrician for ticket T005 (URGENT panel issue). "
            "Must match skills, availability, and propose+confirm appointment."
        ),
        "target_ticket": "T005",
        "max_steps": 12,
    },
    "medium": {
        "name": "medium",
        "description": (
            "Reschedule an existing appointment to a better time slot. "
            "Handle conflicts, minimize distance, respect policy."
        ),
        "target_ticket": "T001",
        "max_steps": 15,
    },
    "hard": {
        "name": "hard",
        "description": (
            "Dispatch 3 electricians to resolve 3 urgent tickets optimally "
            "(T003, T005, T008). Maximize combined score: urgency*rating/distance*cost."
        ),
        "target_tickets": ["T003", "T005", "T008"],
        "max_steps": 20,
    },
}


def grade_easy(state: dict, history: list) -> float:
    """Grade: T005 must have a confirmed appointment with a panel-skilled electrician."""
    from openenv_electrician.data import ELECTRICIANS_DATA

    appointments = state.get("appointments", [])
    t005_appts = [
        a
        for a in appointments
        if a.get("ticket_id") == "T005" and a.get("status") in ("confirmed", "rescheduled")
    ]
    if not t005_appts:
        return _strict01(0.0)

    appt = t005_appts[-1]
    elec_id = appt.get("electrician_id")
    elec = next((e for e in ELECTRICIANS_DATA if e["id"] == elec_id), None)

    if not elec or "panel" not in elec["skills"]:
        return _strict01(0.3)  # scheduled but wrong skill

    if appt.get("start_time") not in elec["availability"]:
        return _strict01(0.5)  # right skill, but invalid slot

    return _strict01(1.0)


def grade_medium(state: dict, history: list) -> float:
    """Grade: T001 must have been rescheduled at least once."""
    from openenv_electrician.data import ELECTRICIANS_DATA

    appointments = [a for a in state.get("appointments", []) if a.get("ticket_id") == "T001"]

    rescheduled = [a for a in appointments if a.get("rescheduled", False) or a.get("status") == "rescheduled"]

    if not rescheduled:
        active = [a for a in appointments if a.get("status") in ("confirmed", "rescheduled")]
        if not active:
            return _strict01(0.0)
        return _strict01(0.4)  # confirmed but not rescheduled

    final_appt = rescheduled[-1]
    elec = next((e for e in ELECTRICIANS_DATA if e["id"] == final_appt.get("electrician_id")), None)

    score = 0.6  # base for successful reschedule
    if elec and "wiring" in elec["skills"]:
        score += 0.2
    if elec and final_appt.get("start_time") in elec["availability"]:
        score += 0.2

    score = min(score, 1.0)
    return _strict01(score)


def grade_hard(state: dict, history: list) -> float:
    """Grade: all three urgent tickets must be optimally dispatched."""
    from openenv_electrician.data import ELECTRICIANS_DATA, TICKETS_DATA

    target_tickets = ["T003", "T005", "T008"]
    appointments = state.get("appointments", [])

    def dist(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    total_score = 0.0
    for tid in target_tickets:
        appts = [
            a
            for a in appointments
            if a.get("ticket_id") == tid and a.get("status") in ("confirmed", "rescheduled")
        ]
        if not appts:
            continue

        appt = appts[-1]
        elec = next((e for e in ELECTRICIANS_DATA if e["id"] == appt.get("electrician_id")), None)
        ticket = next((t for t in TICKETS_DATA if t["ticket_id"] == tid), None)
        if not elec or not ticket:
            continue

        skill_match = 1.0 if ticket["category"] in elec["skills"] else 0.3
        urgency_score = ticket["urgency"] / 5.0
        rating_score = elec["rating"] / 5.0
        d = dist(elec["home_base"], ticket["location"])
        dist_score = max(0.0, 1.0 - d * 100)
        slot_valid = 1.0 if appt.get("start_time") in elec["availability"] else 0.0

        ticket_score = (
            skill_match * 0.3
            + urgency_score * 0.2
            + rating_score * 0.2
            + dist_score * 0.1
            + slot_valid * 0.2
        )
        total_score += ticket_score

    avg = total_score / len(target_tickets)
    avg = min(avg, 1.0)
    return _strict01(avg)


GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}
