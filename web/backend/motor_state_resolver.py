"""Resolve recorded descending-neuron rates into causal motor states.

Behavior intent and flight state are deliberately separate. DN activity says
what the brain is requesting; the flight state constrains which controller the
body can safely execute. This prevents an airborne fly from snapping directly
into feeding or grooming when those recorded channels become active.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MotorDecision:
    behavior_intent: str
    flight_state: str
    controller: str
    reason: str
    steering: float
    queued_intent: str | None
    state_elapsed_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


class MotorStateResolver:
    """Small deterministic state machine driven only by DN rates and time."""

    ENTER_ESCAPE = 0.08
    EXIT_ESCAPE = 0.03
    TAKEOFF_MS = 400.0
    ESCAPE_RELEASE_MS = 250.0
    LANDING_MS = 500.0
    GROOM_RECOVERY_MS = 233.0

    INTENT_THRESHOLDS = {
        "forward": 0.01,
        "backward": 0.02,
        "groom": 0.02,
        "feed": 0.05,
    }

    GROUND_CONTROLLERS = {
        "idle": "idle",
        "walking": "walk",
        "backward": "backward",
        "grooming": "groom",
        "feeding": "feed",
        "escape": "flight_takeoff",
    }

    def __init__(self):
        self.flight_state = "GROUNDED"
        self.state_elapsed_ms = 0.0
        self.escape_release_ms = 0.0
        self.escape_latched = False
        self.queued_intent = None
        self.last_ground_controller = "idle"
        self.ground_recovery_elapsed_ms = None

    def _intent(self, dn: dict) -> str:
        escape = max(0.0, float(dn.get("escape", 0.0)))
        if self.escape_latched:
            if escape <= self.EXIT_ESCAPE:
                self.escape_latched = False
        elif escape >= self.ENTER_ESCAPE:
            self.escape_latched = True
        if self.escape_latched:
            return "escape"

        candidates = {
            "walking": float(dn.get("forward", 0.0))
            / self.INTENT_THRESHOLDS["forward"],
            "backward": float(dn.get("backward", 0.0))
            / self.INTENT_THRESHOLDS["backward"],
            "grooming": float(dn.get("groom", 0.0))
            / self.INTENT_THRESHOLDS["groom"],
            "feeding": float(dn.get("feed", 0.0))
            / self.INTENT_THRESHOLDS["feed"],
        }
        intent, score = max(candidates.items(), key=lambda item: item[1])
        return intent if score >= 1.0 else "idle"

    def _set_flight_state(self, state: str) -> None:
        if state != self.flight_state:
            self.flight_state = state
            self.state_elapsed_ms = 0.0
            self.escape_release_ms = 0.0

    def update(self, dn: dict, dt_ms: float) -> MotorDecision:
        dt_ms = max(0.0, float(dt_ms))
        self.state_elapsed_ms += dt_ms
        intent = self._intent(dn)

        if self.flight_state == "GROUNDED":
            if intent == "escape":
                self._set_flight_state("TAKEOFF")
        elif self.flight_state == "TAKEOFF":
            if self.state_elapsed_ms >= self.TAKEOFF_MS:
                self._set_flight_state("FLYING")
        elif self.flight_state == "FLYING":
            if intent == "escape":
                self.escape_release_ms = 0.0
            else:
                self.escape_release_ms += dt_ms
                if self.escape_release_ms >= self.ESCAPE_RELEASE_MS:
                    self._set_flight_state("LANDING")
        elif self.flight_state == "LANDING":
            if intent == "escape":
                self._set_flight_state("TAKEOFF")
            elif self.state_elapsed_ms >= self.LANDING_MS:
                self._set_flight_state("GROUNDED")

        if self.flight_state == "TAKEOFF":
            controller = "flight_takeoff"
            reason = "flight state gates ground behaviors during takeoff"
        elif self.flight_state == "FLYING":
            controller = "flight"
            reason = "flight state keeps the aerodynamic controller active"
        elif self.flight_state == "LANDING":
            controller = "flight_landing"
            reason = "landing completes before queued ground behavior"
        else:
            if self.ground_recovery_elapsed_ms is not None:
                self.ground_recovery_elapsed_ms += dt_ms
                if self.ground_recovery_elapsed_ms < self.GROOM_RECOVERY_MS:
                    controller = "groom_recover"
                    self.queued_intent = intent if intent != "idle" else None
                    reason = "front legs return to grounded neutral pose before next behavior"
                else:
                    self.ground_recovery_elapsed_ms = None
                    controller = self.GROUND_CONTROLLERS[intent]
                    self.last_ground_controller = controller
                    self.queued_intent = None
                    reason = f"groom recovery complete; grounded intent selects {controller}"
            elif self.last_ground_controller == "groom" and intent != "grooming":
                self.ground_recovery_elapsed_ms = 0.0
                controller = "groom_recover"
                self.queued_intent = intent if intent != "idle" else None
                reason = "front legs must leave the eyes before the next ground behavior"
            else:
                controller = self.GROUND_CONTROLLERS[intent]
                self.last_ground_controller = controller
                self.queued_intent = None
                reason = f"grounded DN intent selects {controller}"

        if self.flight_state != "GROUNDED":
            self.last_ground_controller = None
            self.ground_recovery_elapsed_ms = None
            self.queued_intent = intent if intent not in {"idle", "escape"} else None

        steering = max(
            -1.0,
            min(1.0, float(dn.get("turn_L", 0.0)) - float(dn.get("turn_R", 0.0))),
        )
        return MotorDecision(
            behavior_intent=intent,
            flight_state=self.flight_state,
            controller=controller,
            reason=reason,
            steering=round(steering, 4),
            queued_intent=self.queued_intent,
            state_elapsed_ms=round(self.state_elapsed_ms, 1),
        )
