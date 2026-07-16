"""
Analysis 21: Embodied Brain — Connectome-Driven Walking

Connects the Eon Systems LIF brain model (138,639 neurons) to the
NeuroMechFly v2 biomechanical body (MuJoCo) via descending neuron decoding.

Pipeline:
  1. Stimulate sugar GRN neurons → brain processes signal
  2. Descending neurons (P9, DNa01, MDN) fire in response
  3. DN firing rates → CPG speed/direction modulation
  4. CPG → 42 joint angles → MuJoCo physics → fly walks

This is a simplified version of Eon's full embodied emulation.
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path
from collections import deque
import time

# Add fly-brain code to path
FLY_BRAIN_DIR = Path(__file__).parent / 'fly-brain-embodied'
CODE_DIR = FLY_BRAIN_DIR / 'code'
DATA_DIR = FLY_BRAIN_DIR / 'data'
sys.path.insert(0, str(CODE_DIR))

RESULTS_DIR = Path(__file__).parent / 'results'

# ── Override paths before importing run_pytorch ──
import benchmark
benchmark.COMP_PATH = str(DATA_DIR / '2025_Completeness_783.csv')
benchmark.CONN_PATH = str(DATA_DIR / '2025_Connectivity_783.parquet')
benchmark.DATA_DIR = str(DATA_DIR)

from run_pytorch import TorchModel, MODEL_PARAMS, DT, get_weights, get_hash_tables

# NeuroMechFly
from flygym import Fly, SingleFlySimulation, Camera
from flygym.preprogrammed import get_cpg_biases
from flygym.examples.locomotion import PreprogrammedSteps, CPGNetwork

# =====================================================================
# 1. DESCENDING NEURON DEFINITIONS
# =====================================================================

DN_NEURONS = {
    'P9_left':       720575940627652358,
    'P9_right':      720575940635872101,
    'P9_oDN1_left':  720575940626730883,
    'P9_oDN1_right': 720575940620300308,
    'DNa01_left':    720575940644438551,
    'DNa01_right':   720575940627787609,
    'DNa02_left':    720575940604737708,
    'DNa02_right':   720575940629327659,
    'MDN_1':         720575940616026939,
    'MDN_2':         720575940631082808,
    'MDN_3':         720575940640331472,
    'MDN_4':         720575940610236514,
}

DN_GROUPS = {
    'forward':  ['P9_left', 'P9_right', 'P9_oDN1_left', 'P9_oDN1_right'],
    'turn_L':   ['DNa01_left', 'DNa02_left'],
    'turn_R':   ['DNa01_right', 'DNa02_right'],
    'backward': ['MDN_1', 'MDN_2', 'MDN_3', 'MDN_4'],
}

# Sugar GRN neurons (21 neurons, gustatory receptor neurons)
SUGAR_GRNS = [
    720575940624963786, 720575940630233916, 720575940637568838,
    720575940638202345, 720575940617000768, 720575940630797113,
    720575940632889389, 720575940621754367, 720575940621502051,
    720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940639259967, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570,
    720575940628853239, 720575940629176663, 720575940611875570,
]

# =====================================================================
# 2. BRAIN ENGINE (simplified from brain_body_bridge.py)
# =====================================================================

class BrainEngine:
    """Wraps LIF model for step-by-step execution."""

    def __init__(self, device='cpu'):
        self.device = device
        self.dt = DT  # 0.1 ms

        comp_path = str(DATA_DIR / '2025_Completeness_783.csv')
        conn_path = str(DATA_DIR / '2025_Connectivity_783.parquet')

        self.flyid2i, self.i2flyid = get_hash_tables(comp_path)
        self.num_neurons = len(self.flyid2i)

        print(f"Loading weights for {self.num_neurons} neurons...")
        weights = get_weights(conn_path, comp_path, str(DATA_DIR))
        weights = weights.to(device=self.device)

        self.model = TorchModel(
            batch=1, size=self.num_neurons, dt=self.dt,
            params=MODEL_PARAMS, weights=weights, device=self.device,
        )

        self.state = self.model.state_init()
        self.rates = torch.zeros(1, self.num_neurons, device=self.device)

        # Map DN neuron FlyWire IDs → tensor indices
        self.dn_indices = {}
        for name, flyid in DN_NEURONS.items():
            if flyid in self.flyid2i:
                self.dn_indices[name] = self.flyid2i[flyid]

        # Map sugar GRN IDs → tensor indices
        self.sugar_indices = [self.flyid2i[nid] for nid in SUGAR_GRNS
                              if nid in self.flyid2i]

        print(f"[Brain] {self.num_neurons} neurons on {self.device}")
        print(f"[Brain] DN mapped: {len(self.dn_indices)}/{len(DN_NEURONS)}")
        # Map ORN IDs → tensor indices (for broad olfactory stimulus)
        import pandas as pd
        ann_path = DATA_DIR / 'flywire_annotations.tsv'
        if ann_path.exists():
            ann = pd.read_csv(ann_path, sep='\t')
            orn_ids = ann[ann['cell_class'].str.contains('ORN', na=False)]['root_id'].values
            self.orn_indices = [self.flyid2i[int(rid)] for rid in orn_ids
                                if int(rid) in self.flyid2i]
        else:
            self.orn_indices = []

        print(f"[Brain] Sugar GRNs: {len(self.sugar_indices)}/{len(SUGAR_GRNS)}")
        print(f"[Brain] ORNs: {len(self.orn_indices)}")

    def set_sugar_stimulus(self, rate_hz=200.0):
        """Activate sugar taste neurons at given rate."""
        self.rates.zero_()
        if rate_hz > 0:
            self.rates[0, self.sugar_indices] = rate_hz

    def set_p9_stimulus(self, rate_hz=100.0):
        """Directly stimulate P9 forward-walking neurons (like Eon benchmark)."""
        self.rates.zero_()
        p9_ids = [720575940627652358, 720575940635872101]  # P9 left + right
        for fid in p9_ids:
            if fid in self.flyid2i:
                self.rates[0, self.flyid2i[fid]] = rate_hz

    def set_orn_stimulus(self, rate_hz=200.0):
        """Stimulate olfactory receptor neurons (broad sensory input)."""
        self.rates.zero_()
        if rate_hz > 0:
            self.rates[0, self.orn_indices] = rate_hz

    def clear_stimulus(self):
        self.rates.zero_()

    @torch.no_grad()
    def step(self):
        """Advance brain by one timestep (0.1 ms)."""
        cond, dbuf, spk, v, ref = self.state
        self.state = self.model(self.rates, cond, dbuf, spk, v, ref)
        return self.state[2]  # spikes (1, num_neurons)

    def get_dn_spikes(self):
        """Return current DN spike values."""
        spk = self.state[2]
        return {name: spk[0, idx].item()
                for name, idx in self.dn_indices.items()}


# =====================================================================
# 3. DN RATE DECODER
# =====================================================================

class DNDecoder:
    """Computes firing rates from DN spike trains using sliding window."""

    def __init__(self, window_ms=50.0, dt_ms=0.1, max_rate=200.0):
        self.window_steps = int(window_ms / dt_ms)  # 500 steps
        self.dt_s = dt_ms / 1000.0
        self.max_rate = max_rate
        self.buffers = {n: deque(maxlen=self.window_steps)
                        for n in DN_NEURONS.keys()}

    def update(self, dn_spikes):
        for name, val in dn_spikes.items():
            self.buffers[name].append(val)

    def get_rate(self, name):
        buf = self.buffers.get(name)
        if not buf:
            return 0.0
        n = len(buf)
        window_s = n * self.dt_s
        return sum(buf) / window_s if window_s > 0 else 0.0

    def get_normalized(self, name):
        return min(self.get_rate(name) / self.max_rate, 1.0)

    def get_group_rate(self, group_name):
        names = DN_GROUPS.get(group_name, [])
        if not names:
            return 0.0
        return np.mean([self.get_normalized(n) for n in names])


# =====================================================================
# 4. BRAIN-BODY BRIDGE
# =====================================================================

def dn_to_drive(decoder):
    """Convert DN firing rates to [left_drive, right_drive] for CPG."""
    forward = decoder.get_group_rate('forward')
    turn_L = decoder.get_group_rate('turn_L')
    turn_R = decoder.get_group_rate('turn_R')
    backward = decoder.get_group_rate('backward')

    # Net forward drive
    speed = forward - backward
    speed = max(0, speed)  # clip negative

    # Turn: asymmetry → differential drive
    turn = turn_L - turn_R

    left_drive = speed - turn * 0.5
    right_drive = speed + turn * 0.5

    return np.clip(left_drive, 0, 1), np.clip(right_drive, 0, 1)


# =====================================================================
# 5. MAIN SIMULATION
# =====================================================================

def main():
    print("=" * 72)
    print("ANALYSIS 21: EMBODIED BRAIN — CONNECTOME-DRIVEN WALKING")
    print("=" * 72)

    # ── Brain setup ──
    print("\n[1/4] Loading brain model...")
    t0 = time.time()
    # MPS doesn't support sparse tensors, use CPU with all cores
    torch.set_num_threads(10)  # M4 has 10 CPU cores
    brain = BrainEngine(device='cpu')
    print(f"  Brain loaded in {time.time()-t0:.1f}s")

    decoder = DNDecoder(window_ms=50.0)

    # ── Body setup ──
    print("\n[2/4] Creating fly body (NeuroMechFly v2)...")
    body_timestep = 1e-4  # 0.1 ms (matches brain dt)
    fly = Fly(enable_adhesion=True, draw_adhesion=True, init_pose='stretch', control='position')

    cam = Camera(
        attachment_point=fly.model.worldbody,
        camera_name='camera_right',
        targeted_fly_names=[fly.name],
        play_speed=0.2,
        window_size=(1280, 720),
        fps=30,
        timestamp_text=True,
        draw_contacts=True,
    )
    sim = SingleFlySimulation(fly=fly, cameras=[cam], timestep=body_timestep)

    # CPG controller
    cpg = CPGNetwork(
        timestep=body_timestep,
        intrinsic_freqs=np.ones(6) * 12.0,
        intrinsic_amps=np.ones(6),
        coupling_weights=(get_cpg_biases('tripod') > 0).astype(float) * 10.0,
        phase_biases=get_cpg_biases('tripod'),
        convergence_coefs=np.ones(6) * 20.0,
    )
    preprogrammed_steps = PreprogrammedSteps()
    leg_names = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']

    obs, info = sim.reset()

    # ── Simulation ──
    # Brain timestep = 0.1ms, Body timestep = 0.1ms → 1:1
    # But brain is expensive on CPU, so we sync every 15ms (150 brain steps)
    # like Eon does

    sim_duration_s = 0.5  # 500ms
    sync_interval_ms = 15.0  # brain-body sync every 15ms
    brain_steps_per_sync = int(sync_interval_ms / DT)  # 150 (DT is in ms)
    body_steps_per_sync = int(sync_interval_ms / (body_timestep * 1000))  # 150
    n_syncs = int(sim_duration_s * 1000 / sync_interval_ms)  # ~67

    print(f"\n[3/4] Running embodied simulation...")
    print(f"  Duration: {sim_duration_s}s")
    print(f"  Sync interval: {sync_interval_ms}ms")
    print(f"  Brain steps/sync: {brain_steps_per_sync}")
    print(f"  Body steps/sync: {body_steps_per_sync}")
    print(f"  Total syncs: {n_syncs}")

    # Phase 1: No stimulus (0-200ms) — baseline
    # Phase 2: Sugar stimulus (200-800ms) — should trigger walking
    # Phase 3: No stimulus (800-1000ms) — brain settles

    positions = []
    dn_history = {g: [] for g in DN_GROUPS}
    phase_labels = []

    t_start = time.time()

    for sync_i in range(n_syncs):
        t_ms = sync_i * sync_interval_ms
        t_s = t_ms / 1000.0

        # Set stimulus based on phase
        # P9 direct stimulation → forward walking (like Eon benchmark)
        if t_ms < 30:
            brain.clear_stimulus()
            phase = 'baseline'
        elif t_ms < 400:
            brain.set_p9_stimulus(100.0)  # P9 at 100 Hz
            phase = 'P9_stim'
        else:
            brain.clear_stimulus()
            phase = 'settle'

        phase_labels.append(phase)

        # ── Run brain for sync_interval ──
        total_spikes = 0
        dn_spike_count = 0
        for _ in range(brain_steps_per_sync):
            spikes = brain.step()
            total_spikes += spikes.sum().item()
            dn_spikes = brain.get_dn_spikes()
            dn_spike_count += sum(dn_spikes.values())
            decoder.update(dn_spikes)

        # Record DN group rates
        for g in DN_GROUPS:
            dn_history[g].append(decoder.get_group_rate(g))

        # ── Convert DN rates to motor drive ──
        left_drive, right_drive = dn_to_drive(decoder)

        # Modulate CPG frequency based on brain drive
        avg_drive = (left_drive + right_drive) / 2.0
        base_freq = 12.0
        cpg_freq = base_freq * (0.5 + avg_drive)  # 6-18 Hz based on drive

        # Update CPG frequencies
        cpg.intrinsic_freqs[:] = cpg_freq

        # Differential turning: left/right drive difference
        if avg_drive > 0.01:
            turn_bias = (right_drive - left_drive) / avg_drive
        else:
            turn_bias = 0.0

        # ── Run body for sync_interval ──
        for body_step in range(body_steps_per_sync):
            cpg.step()
            all_joint_angles = []
            all_adhesion = []
            for i, leg in enumerate(leg_names):
                angles = preprogrammed_steps.get_joint_angles(
                    leg, cpg.curr_phases[i], cpg.curr_magnitudes[i])
                all_joint_angles.append(angles)
                adhesion = preprogrammed_steps.get_adhesion_onoff(
                    leg, cpg.curr_phases[i])
                all_adhesion.append(adhesion)

            action = {
                'joints': np.concatenate(all_joint_angles),
                'adhesion': np.array(all_adhesion, dtype=np.float64),
            }
            obs, reward, terminated, truncated, info = sim.step(action)
            sim.render()

        pos = obs['fly'][0].copy()
        positions.append(pos)

        if sync_i % 5 == 0:
            elapsed = time.time() - t_start
            print(f"  t={t_ms:6.0f}ms [{phase:>8s}] "
                  f"brain_spikes={total_spikes:.0f} dn_spikes={dn_spike_count:.0f} "
                  f"fwd={dn_history['forward'][-1]:.4f} "
                  f"pos=({pos[0]:.2f}, {pos[1]:.2f}) "
                  f"[{elapsed:.1f}s elapsed]")

    # Save video
    video_path = str(RESULTS_DIR / '21_embodied_brain.mp4')
    cam.save_video(video_path)
    print(f"\nVideo saved: {video_path}")

    sim.close()
    total_time = time.time() - t_start

    # ── Results ──
    print(f"\n[4/4] Results")
    print(f"  Total wall time: {total_time:.1f}s")
    positions = np.array(positions)
    total_dist = np.linalg.norm(positions[-1, :2] - positions[0, :2])
    print(f"  Distance walked: {total_dist:.2f} mm")
    print(f"  Start: ({positions[0, 0]:.2f}, {positions[0, 1]:.2f})")
    print(f"  End:   ({positions[-1, 0]:.2f}, {positions[-1, 1]:.2f})")

    # DN activity summary
    print(f"\n  DN Activity (mean normalized rate):")
    for phase_name in ['baseline', 'sugar', 'settle']:
        mask = [i for i, p in enumerate(phase_labels) if p == phase_name]
        if mask:
            for g in DN_GROUPS:
                rates = [dn_history[g][i] for i in mask]
                print(f"    {phase_name:>8s} | {g:>10s}: {np.mean(rates):.4f}")

    # ── Visualization ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Analysis 21: Embodied Brain — Connectome-Driven Walking\n'
                 '138,639 LIF neurons → Descending neurons → CPG → MuJoCo body',
                 fontsize=12, fontweight='bold')

    t_axis = np.arange(n_syncs) * sync_interval_ms

    # Panel 1: DN firing rates
    ax = axes[0, 0]
    colors = {'forward': '#2ecc71', 'turn_L': '#3498db', 'turn_R': '#e74c3c', 'backward': '#9b59b6'}
    for g, c in colors.items():
        ax.plot(t_axis, dn_history[g], color=c, label=g, linewidth=1.5)
    ax.axvspan(200, 800, alpha=0.15, color='orange', label='Sugar stimulus')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Normalized DN rate')
    ax.set_title('Descending Neuron Activity')
    ax.legend(fontsize=8)
    ax.set_xlim(0, sim_duration_s * 1000)

    # Panel 2: Fly trajectory (top view)
    ax = axes[0, 1]
    # Color by phase
    for i in range(len(positions) - 1):
        c = '#e74c3c' if phase_labels[i] == 'baseline' else ('#2ecc71' if phase_labels[i] == 'sugar' else '#3498db')
        ax.plot(positions[i:i+2, 0], positions[i:i+2, 1], color=c, linewidth=2)
    ax.plot(positions[0, 0], positions[0, 1], 'ko', markersize=10, label='Start')
    ax.plot(positions[-1, 0], positions[-1, 1], 'k*', markersize=15, label='End')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title(f'Fly Trajectory (top view) — {total_dist:.1f}mm total')
    ax.legend()
    ax.set_aspect('equal')

    # Panel 3: Position over time
    ax = axes[1, 0]
    ax.plot(t_axis, positions[:, 0], label='X', color='#2ecc71')
    ax.plot(t_axis, positions[:, 1], label='Y', color='#3498db')
    ax.axvspan(200, 800, alpha=0.15, color='orange')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Position (mm)')
    ax.set_title('Position Over Time')
    ax.legend()

    # Panel 4: Architecture diagram
    ax = axes[1, 1]
    ax.axis('off')
    diagram = """
    ARCHITECTURE

    Sugar GRNs (21 neurons, 200 Hz)
           ↓
    ┌──────────────────────────────┐
    │    LIF Brain Model           │
    │    138,639 neurons           │
    │    ~5M synapses              │
    │    0.1ms timestep            │
    └──────────────────────────────┘
           ↓
    Descending Neurons (18 cells)
    P9 → forward | DNa01/02 → turn
    MDN → backward
           ↓
    DN Rate Decoder (50ms window)
           ↓
    CPG Modulation (6-18 Hz)
           ↓
    ┌──────────────────────────────┐
    │    NeuroMechFly v2 (MuJoCo)  │
    │    42 joint actuators         │
    │    6 legs, tripod gait       │
    └──────────────────────────────┘
           ↓
    3D Physics Simulation
    """
    ax.text(0.05, 0.95, diagram, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    fig.savefig(str(RESULTS_DIR / '21_embodied_brain.png'), dpi=150, bbox_inches='tight')
    print(f"\nSaved: {RESULTS_DIR / '21_embodied_brain.png'}")
    plt.close()

    print("\n" + "=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)


if __name__ == '__main__':
    main()
