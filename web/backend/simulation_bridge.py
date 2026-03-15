"""
Simulation bridge: runs simulation (mock or real) in a background thread,
pushes SimFrame objects to an asyncio queue for WebSocket broadcast.

Real brain mode: loads 138,639 LIF neurons from FlyWire v783 connectome,
runs through fly-brain-embodied's BrainEngine, and streams actual DN/NT data.
"""

import sys
import math
import time
import asyncio
import threading
from pathlib import Path
from models import SimFrame

# Paths for real brain imports
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
FLY_BRAIN_DIR = PROJECT_DIR / 'fly-brain-embodied'
CODE_DIR = FLY_BRAIN_DIR / 'code'
DATA_DIR = FLY_BRAIN_DIR / 'data'

# Add to path for imports (only when needed)
for p in [str(CODE_DIR), str(FLY_BRAIN_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Scenario definitions
SCENARIOS = {
    "escape": {
        "name": "escape",
        "label": "Hunger → Escape Flight",
        "description": "Hunger drives walking, looming threat triggers Giant Fiber escape.",
        "duration_s": 1.5,
        "phases": [
            {"name": "hunger",  "start_ms": 0,   "end_ms": 300,  "stimulus": "p9",  "rate_hz": 96},
            {"name": "looming", "start_ms": 300,  "end_ms": 600,  "stimulus": "lc4", "rate_hz": 200},
            {"name": "free",    "start_ms": 600,  "end_ms": 1500, "stimulus": None,  "rate_hz": 0},
        ],
    },
    "foraging": {
        "name": "foraging",
        "label": "Sugar Foraging",
        "description": "Fly detects sugar, approaches food source.",
        "duration_s": 1.5,
        "phases": [
            {"name": "search",  "start_ms": 0,   "end_ms": 500,  "stimulus": "p9",    "rate_hz": 60},
            {"name": "sugar",   "start_ms": 500,  "end_ms": 1200, "stimulus": "sugar", "rate_hz": 200},
            {"name": "feeding", "start_ms": 1200, "end_ms": 1500, "stimulus": None,    "rate_hz": 0},
        ],
    },
    "grooming": {
        "name": "grooming",
        "label": "Touch → Grooming",
        "description": "Antennal touch triggers grooming behavior via JO neurons.",
        "duration_s": 1.2,
        "phases": [
            {"name": "walking", "start_ms": 0,   "end_ms": 300,  "stimulus": "p9", "rate_hz": 80},
            {"name": "touch",   "start_ms": 300,  "end_ms": 900,  "stimulus": "jo", "rate_hz": 150},
            {"name": "free",    "start_ms": 900,  "end_ms": 1200, "stimulus": None, "rate_hz": 0},
        ],
    },
}


def _mock_sim_frame(t_ms: float, scenario_name: str) -> SimFrame:
    """Generate a mock SimFrame with plausible sin-wave dynamics."""
    t_s = t_ms / 1000.0
    scenario = SCENARIOS[scenario_name]
    phases = scenario["phases"]

    # Determine current phase
    phase = phases[-1]["name"]
    stimulus = None
    for p in phases:
        if p["start_ms"] <= t_ms < p["end_ms"]:
            phase = p["name"]
            stimulus = p["stimulus"]
            break

    # Base oscillations
    osc1 = math.sin(2 * math.pi * 0.8 * t_s)
    osc2 = math.sin(2 * math.pi * 1.3 * t_s + 0.5)
    noise = math.sin(2 * math.pi * 7.1 * t_s) * 0.05

    # DN rates depend on phase
    dn_escape = 0.0
    dn_forward = 0.0
    dn_backward = 0.0
    dn_groom = 0.0
    dn_feed = 0.0
    behavior = "walking"
    flight_state = "GROUNDED"
    alt = 0.0
    wing_freq = 0.0

    if scenario_name == "escape":
        if phase == "hunger":
            dn_forward = 0.4 + 0.1 * osc1
            behavior = "walking"
        elif phase == "looming":
            progress = (t_ms - 300) / 300.0
            dn_escape = min(1.0, 0.1 + progress * 0.9 + noise)
            dn_forward = max(0, 0.3 - progress * 0.3)
            if progress > 0.3:
                flight_state = "TAKEOFF" if progress < 0.5 else "FLYING"
                alt = min(5.0, progress * 8.0)
                wing_freq = 200.0
                behavior = "flight"
        elif phase == "free":
            decay = math.exp(-(t_ms - 600) / 200.0)
            dn_escape = 0.05 * decay
            dn_forward = 0.1 + 0.2 * (1 - decay)
            progress_land = (t_ms - 600) / 400.0
            if progress_land < 0.5:
                flight_state = "FLYING"
                alt = max(1.0, 5.0 - progress_land * 8.0)
                wing_freq = 200.0
                behavior = "flight"
            elif progress_land < 1.0:
                flight_state = "LANDING"
                alt = max(0.0, 2.0 - (progress_land - 0.5) * 4.0)
                wing_freq = 160.0 * max(0, 1.0 - progress_land)
                behavior = "flight"
            else:
                flight_state = "GROUNDED"
                behavior = "walking"

    elif scenario_name == "foraging":
        if phase == "search":
            dn_forward = 0.3 + 0.1 * osc1
            behavior = "walking"
        elif phase == "sugar":
            progress = (t_ms - 500) / 700.0
            dn_forward = max(0, 0.3 - progress * 0.3)
            dn_feed = min(1.0, progress * 1.2)
            behavior = "feeding" if progress > 0.3 else "walking"
        elif phase == "feeding":
            dn_feed = 0.8 + 0.1 * osc2
            behavior = "feeding"

    elif scenario_name == "grooming":
        if phase == "walking":
            dn_forward = 0.35 + 0.08 * osc1
            behavior = "walking"
        elif phase == "touch":
            progress = (t_ms - 300) / 600.0
            dn_groom = min(0.9, 0.2 + progress * 0.7 + noise)
            dn_forward = max(0, 0.2 - progress * 0.2)
            behavior = "grooming"
        elif phase == "free":
            decay = math.exp(-(t_ms - 900) / 150.0)
            dn_groom = 0.3 * decay
            dn_forward = 0.2 * (1 - decay)
            behavior = "walking" if decay < 0.3 else "grooming"

    # Turn rates (small asymmetric noise)
    dn_turn_L = 0.05 + 0.03 * osc2 + noise
    dn_turn_R = 0.05 - 0.02 * osc2 + noise

    # Position (simple walk + flight trajectory)
    walk_speed = 2.0  # mm/s
    px = walk_speed * t_s * (1.0 + 0.3 * osc1)
    py = 0.5 * math.sin(2 * math.pi * 0.3 * t_s)
    pz = alt

    # Drive
    turn = dn_turn_L - dn_turn_R
    drive_L = dn_forward * (1 + turn) - dn_backward
    drive_R = dn_forward * (1 - turn) - dn_backward

    # NT populations (mock spike counts with phase-dependent modulation)
    base_activity = 50 + 30 * osc1
    threat_boost = max(0, dn_escape * 200)
    feed_boost = max(0, dn_feed * 150)
    groom_boost = max(0, dn_groom * 100)

    pam = int(max(0, 20 + feed_boost + 10 * osc2))
    ppl1 = int(max(0, 8 + threat_boost * 0.3 + 5 * osc1))
    mbon_approach = int(max(0, 15 + feed_boost * 0.5 + 8 * osc2))
    mbon_avoidance = int(max(0, 10 + threat_boost * 0.4 + 6 * osc1))
    mbon_suppress = int(max(0, 5 + 3 * osc2))
    serotonin = int(max(0, 12 + groom_boost * 0.3 + 8 * osc2))
    octopamine = int(max(0, 8 + threat_boost * 0.2 + 4 * osc1))
    gaba = int(max(0, base_activity * 2.5 + threat_boost * 0.5))
    ach = int(max(0, base_activity * 4.0 + feed_boost + threat_boost))
    glut = int(max(0, base_activity * 1.8 + threat_boost * 0.3))
    total_spikes = gaba + ach + glut + pam + ppl1 + serotonin + octopamine

    return SimFrame(
        t_ms=round(t_ms, 1),
        phase=phase,
        flight_state=flight_state,
        pos=[round(px, 2), round(py, 2), round(pz, 2)],
        alt_mm=round(alt, 2),
        dn_escape=round(max(0, dn_escape), 4),
        dn_forward=round(max(0, dn_forward), 4),
        dn_backward=round(max(0, dn_backward), 4),
        dn_turn_L=round(max(0, dn_turn_L), 4),
        dn_turn_R=round(max(0, dn_turn_R), 4),
        dn_groom=round(max(0, dn_groom), 4),
        dn_feed=round(max(0, dn_feed), 4),
        drive=[round(drive_L, 4), round(drive_R, 4)],
        pam=pam, ppl1=ppl1,
        mbon_approach=mbon_approach, mbon_avoidance=mbon_avoidance,
        mbon_suppress=mbon_suppress,
        serotonin=serotonin, octopamine=octopamine,
        gaba=gaba, ach=ach, glut=glut,
        wing_freq=round(wing_freq, 1),
        behavior_mode=behavior,
        total_spikes=total_spikes,
    )


class SimulationBridge:
    """Manages simulation lifecycle — runs in background thread, feeds async queue."""

    def __init__(self):
        self.queue: asyncio.Queue | None = None
        self.running = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.scenario = "escape"
        self.duration_s = 1.5

    def start(self, scenario: str, duration_s: float, loop: asyncio.AbstractEventLoop,
              queue: asyncio.Queue, use_real_brain: bool = False):
        if self.running:
            return
        self.scenario = scenario
        self.duration_s = duration_s
        self.queue = queue
        self._loop = loop
        self.running = True

        if use_real_brain:
            self._thread = threading.Thread(target=self._run_real, daemon=True)
        else:
            self._thread = threading.Thread(target=self._run_mock, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def _emit(self, frame):
        """Thread-safe put into asyncio queue."""
        asyncio.run_coroutine_threadsafe(self.queue.put(frame), self._loop)

    def _run_mock(self):
        """Generate mock frames at ~30Hz."""
        fps = 30
        dt_ms = 1000.0 / fps
        t_ms = 0.0
        end_ms = self.duration_s * 1000.0

        while self.running and t_ms <= end_ms:
            frame = _mock_sim_frame(t_ms, self.scenario)
            self._emit(frame)
            t_ms += dt_ms
            time.sleep(1.0 / fps)

        self.running = False
        self._emit(None)

    def _run_real(self):
        """Run real 138,639-neuron brain simulation."""
        import torch

        # Fix benchmark paths before importing
        import benchmark
        benchmark.COMP_PATH = str(DATA_DIR / '2025_Completeness_783.csv')
        benchmark.CONN_PATH = str(DATA_DIR / '2025_Connectivity_783.parquet')
        benchmark.DATA_DIR = str(DATA_DIR)

        from run_pytorch import TorchModel, MODEL_PARAMS, DT, get_weights, get_hash_tables
        from brain_body_bridge import DN_NEURONS, DN_GROUPS, STIMULI, DNRateDecoder

        torch.set_num_threads(10)

        # ── Load brain ──
        comp_path = str(DATA_DIR / '2025_Completeness_783.csv')
        conn_path = str(DATA_DIR / '2025_Connectivity_783.parquet')
        flyid2i, _ = get_hash_tables(comp_path)
        num_neurons = len(flyid2i)

        weights = get_weights(conn_path, comp_path, str(DATA_DIR))
        device = 'cpu'
        weights = weights.to(device=device)
        model = TorchModel(
            batch=1, size=num_neurons, dt=DT,
            params=MODEL_PARAMS, weights=weights, device=device,
        )
        state = model.state_init()
        rates = torch.zeros(1, num_neurons, device=device)

        # Stimulus mapping
        stim_map = {}
        for sn, si in STIMULI.items():
            stim_map[sn] = [flyid2i[nid] for nid in si['neurons'] if nid in flyid2i]

        # DN mapping
        dn_map = {name: flyid2i[fid] for name, fid in DN_NEURONS.items() if fid in flyid2i}

        decoder = DNRateDecoder(window_ms=50.0, dt_ms=DT, max_rate=200.0)

        # Population tensors for NT logging
        import pandas as pd
        ann_path = PROJECT_DIR / "data" / "neuron_annotations.tsv"
        ann = pd.read_csv(ann_path, sep="\t", low_memory=False)

        def ids_to_tensor(root_ids):
            return torch.tensor([flyid2i[nid] for nid in root_ids if nid in flyid2i],
                                dtype=torch.long, device=device)

        mbon = ann[ann['cell_class'] == 'MBON']
        ser = ann[ann['top_nt'] == 'serotonin']
        ser = ser[~ser['cell_class'].isin(['olfactory', 'visual', 'mechanosensory',
                                            'unknown_sensory', 'gustatory'])]
        octo = ann[ann['top_nt'] == 'octopamine']
        octo = octo[~octo['cell_class'].isin(['olfactory', 'visual', 'mechanosensory',
                                               'unknown_sensory', 'gustatory'])]

        pop_tensors = {
            'pam': ids_to_tensor(ann[ann['cell_type'].str.startswith('PAM', na=False)]['root_id']),
            'ppl1': ids_to_tensor(ann[ann['cell_type'].str.startswith('PPL1', na=False)]['root_id']),
            'mbon_approach': ids_to_tensor(mbon[mbon['top_nt'] == 'acetylcholine']['root_id']),
            'mbon_avoidance': ids_to_tensor(mbon[mbon['top_nt'] == 'glutamate']['root_id']),
            'mbon_suppress': ids_to_tensor(mbon[mbon['top_nt'] == 'gaba']['root_id']),
            'serotonin': ids_to_tensor(ser['root_id']),
            'octopamine': ids_to_tensor(octo['root_id']),
            'gaba': ids_to_tensor(ann[ann['top_nt'] == 'gaba']['root_id']),
            'ach': ids_to_tensor(ann[ann['top_nt'] == 'acetylcholine']['root_id']),
            'glut': ids_to_tensor(ann[ann['top_nt'] == 'glutamate']['root_id']),
        }

        # ── Simulation loop ──
        scenario = SCENARIOS[self.scenario]
        phases = scenario["phases"]
        end_ms = self.duration_s * 1000.0

        # Emit every 10 brain steps = 1ms sim time.
        # On CPU (no CUDA), 138K-neuron sparse matmul is ~0.5-1s per step,
        # so each emit takes ~5-10s wall-clock. This gives the frontend
        # regular updates while accumulating enough spikes to see activity.
        emit_every_n = 10
        brain_step = 0
        spike_acc = torch.zeros(num_neurons, device=device)
        pos_x, pos_y, pos_z = 0.0, 0.0, 0.0
        last_phase_set = None

        while self.running:
            t_ms = brain_step * DT

            if t_ms > end_ms:
                break

            # Determine phase and set stimulus (only when phase changes)
            current_phase = phases[-1]["name"]
            current_stim = None
            current_rate = 0
            for p in phases:
                if p["start_ms"] <= t_ms < p["end_ms"]:
                    current_phase = p["name"]
                    current_stim = p["stimulus"]
                    current_rate = p["rate_hz"]
                    break

            if current_phase != last_phase_set:
                rates.zero_()
                if current_stim and current_stim in stim_map:
                    for idx in stim_map[current_stim]:
                        rates[0, idx] = current_rate
                last_phase_set = current_phase

            # Brain step
            with torch.no_grad():
                cond, dbuf, spk, v, ref = state
                state = model(rates, cond, dbuf, spk, v, ref)
            spike_acc += state[2][0]

            # DN spikes
            dn_spikes = {name: state[2][0, idx].item() for name, idx in dn_map.items()}
            decoder.update(dn_spikes)

            brain_step += 1

            # Emit frame every emit_every_n brain steps (= 15ms sim time)
            if brain_step % emit_every_n == 0:
                dn_rates = {g: decoder.get_group_rate(g) for g in DN_GROUPS}

                # Simple behavior mode detection
                gf = dn_rates.get('escape', 0)
                fwd = dn_rates.get('forward', 0)
                grm = dn_rates.get('groom', 0)
                fed = dn_rates.get('feed', 0)

                if gf > 0.06:
                    behavior = 'flight'
                    flight_state = 'FLYING'
                    pos_z = min(5.0, pos_z + gf * 0.5)
                    wing_freq = 200.0
                elif grm > 0.05:
                    behavior = 'grooming'
                    flight_state = 'GROUNDED'
                    wing_freq = 0.0
                    pos_z = max(0.0, pos_z - 0.3)
                elif fed > 0.05:
                    behavior = 'feeding'
                    flight_state = 'GROUNDED'
                    wing_freq = 0.0
                    pos_z = max(0.0, pos_z - 0.3)
                else:
                    behavior = 'walking'
                    flight_state = 'GROUNDED' if pos_z < 0.5 else 'LANDING'
                    wing_freq = 0.0
                    pos_z = max(0.0, pos_z - 0.2)

                # Position update
                turn = dn_rates.get('turn_L', 0) - dn_rates.get('turn_R', 0)
                pos_x += fwd * 0.3
                pos_y += turn * 0.1

                # Population spikes
                sync_pop = {name: spike_acc[idx].sum().item()
                            for name, idx in pop_tensors.items()}
                total = spike_acc.sum().item()
                spike_acc.zero_()

                drive_l = fwd * (1 + turn) - dn_rates.get('backward', 0)
                drive_r = fwd * (1 - turn) - dn_rates.get('backward', 0)

                frame = SimFrame(
                    t_ms=round(t_ms, 1),
                    phase=phase,
                    flight_state=flight_state,
                    pos=[round(pos_x, 2), round(pos_y, 2), round(pos_z, 2)],
                    alt_mm=round(pos_z, 2),
                    dn_escape=round(gf, 4),
                    dn_forward=round(fwd, 4),
                    dn_backward=round(dn_rates.get('backward', 0), 4),
                    dn_turn_L=round(dn_rates.get('turn_L', 0), 4),
                    dn_turn_R=round(dn_rates.get('turn_R', 0), 4),
                    dn_groom=round(grm, 4),
                    dn_feed=round(fed, 4),
                    drive=[round(drive_l, 4), round(drive_r, 4)],
                    pam=int(sync_pop.get('pam', 0)),
                    ppl1=int(sync_pop.get('ppl1', 0)),
                    mbon_approach=int(sync_pop.get('mbon_approach', 0)),
                    mbon_avoidance=int(sync_pop.get('mbon_avoidance', 0)),
                    mbon_suppress=int(sync_pop.get('mbon_suppress', 0)),
                    serotonin=int(sync_pop.get('serotonin', 0)),
                    octopamine=int(sync_pop.get('octopamine', 0)),
                    gaba=int(sync_pop.get('gaba', 0)),
                    ach=int(sync_pop.get('ach', 0)),
                    glut=int(sync_pop.get('glut', 0)),
                    wing_freq=round(wing_freq, 1),
                    behavior_mode=behavior,
                    total_spikes=int(total),
                )
                self._emit(frame)

        self.running = False
        self._emit(None)
