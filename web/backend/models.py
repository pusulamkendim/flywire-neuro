"""Pydantic models for the simulation WebSocket protocol."""

from pydantic import BaseModel


class SimFrame(BaseModel):
    """One frame of simulation state, sent to frontend at ~30Hz."""
    t_ms: float
    phase: str                    # hunger, looming, free
    flight_state: str             # GROUNDED, TAKEOFF, FLYING, LANDING
    pos: list[float]              # [x, y, z] mm
    alt_mm: float

    # DN rates (0-1 normalized)
    dn_escape: float
    dn_forward: float
    dn_backward: float
    dn_turn_L: float
    dn_turn_R: float
    dn_groom: float
    dn_feed: float

    # Drive
    drive: list[float]            # [left, right]

    # NT populations (spike counts per interval)
    pam: int
    ppl1: int
    mbon_approach: int
    mbon_avoidance: int
    mbon_suppress: int
    serotonin: int
    octopamine: int
    gaba: int
    ach: int
    glut: int

    # Flight
    wing_freq: float
    behavior_mode: str            # walking, escape, grooming, feeding, flight
    total_spikes: int


class Scenario(BaseModel):
    """A scenario the user can select."""
    name: str
    label: str
    description: str
    phases: list[dict]            # [{name, start_ms, end_ms, stimulus, rate_hz}]


class SimConfig(BaseModel):
    """Configuration sent from frontend to start a simulation."""
    scenario: str = "escape"
    duration_s: float = 1.5
    use_real_brain: bool = False   # False = mock data
