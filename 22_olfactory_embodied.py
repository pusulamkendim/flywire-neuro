"""
Analysis 22: Olfactory Navigation — Brain-Controlled Fly Smells and Turns

Optimized brain simulation (scipy spike-aware matmul: ~38x faster)
+ bilateral ORN stimulation for directional olfactory response.

Experiment: Stimulate LEFT ORNs → does the fly turn left toward the odor?
"""

import sys
import os
import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
from pathlib import Path
from collections import deque
import time

# Add fly-brain code to path
FLY_BRAIN_DIR = Path(__file__).parent / 'fly-brain-embodied'
CODE_DIR = FLY_BRAIN_DIR / 'code'
DATA_DIR = FLY_BRAIN_DIR / 'data'
sys.path.insert(0, str(CODE_DIR))

RESULTS_DIR = Path(__file__).parent / 'results'
DATA_LOCAL = Path(__file__).parent / 'data'

# Override paths
import benchmark
benchmark.COMP_PATH = str(DATA_DIR / '2025_Completeness_783.csv')
benchmark.CONN_PATH = str(DATA_DIR / '2025_Connectivity_783.parquet')
benchmark.DATA_DIR = str(DATA_DIR)

from run_pytorch import TorchModel, MODEL_PARAMS, DT, get_weights, get_hash_tables

from flygym import Fly, SingleFlySimulation, Camera
from flygym.preprogrammed import get_cpg_biases
from flygym.examples.locomotion import PreprogrammedSteps, CPGNetwork

# =====================================================================
# DN DEFINITIONS
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

# =====================================================================
# OPTIMIZED BRAIN ENGINE (scipy spike-aware matmul)
# =====================================================================

class FastBrainEngine:
    """LIF brain with scipy sparse matmul — ~38x faster than torch CSR."""

    def __init__(self):
        comp_path = str(DATA_DIR / '2025_Completeness_783.csv')
        conn_path = str(DATA_DIR / '2025_Connectivity_783.parquet')

        self.flyid2i, self.i2flyid = get_hash_tables(comp_path)
        self.num_neurons = len(self.flyid2i)

        print(f"Loading {self.num_neurons} neurons...")
        weights = get_weights(conn_path, comp_path, str(DATA_DIR))

        # Convert torch sparse CSR → scipy CSR (transposed for row-select trick)
        w_csr = weights.to_sparse_csr()
        crow = w_csr.crow_indices().numpy()
        col = w_csr.col_indices().numpy()
        vals = w_csr.values().numpy()
        self.w_scipy = sp.csr_matrix(
            (vals, col, crow), shape=(self.num_neurons, self.num_neurons)
        )
        # We need W^T for the matmul: result = spikes @ W^T = W^T[firing,:].sum()
        self.w_scipy_t = self.w_scipy.T.tocsr()
        print(f"  Scipy CSR: {self.w_scipy_t.nnz:,} non-zeros")

        # Create torch model for Poisson + LIF (but we'll override the matmul)
        self.model = TorchModel(
            batch=1, size=self.num_neurons, dt=DT,
            params=MODEL_PARAMS, weights=weights, device='cpu',
        )
        self.scale = MODEL_PARAMS['wScale']
        self.state = self.model.state_init()
        self.rates = torch.zeros(1, self.num_neurons)

        # DN indices
        self.dn_indices = {}
        for name, flyid in DN_NEURONS.items():
            if flyid in self.flyid2i:
                self.dn_indices[name] = self.flyid2i[flyid]

        # Load ORN neuron IDs (left/right)
        ann = pd.read_csv(DATA_LOCAL / 'neuron_annotations.tsv', sep='\t',
                          low_memory=False)
        orns = ann[ann['cell_class'].str.contains('olfactory', case=False, na=False)]

        self.orn_left = [self.flyid2i[int(r)] for r in
                         orns[orns['side'] == 'left']['root_id'].values
                         if int(r) in self.flyid2i]
        self.orn_right = [self.flyid2i[int(r)] for r in
                          orns[orns['side'] == 'right']['root_id'].values
                          if int(r) in self.flyid2i]

        print(f"  DN mapped: {len(self.dn_indices)}/{len(DN_NEURONS)}")
        print(f"  ORN left: {len(self.orn_left)}, right: {len(self.orn_right)}")

    def set_stimulus(self, left_rate=0.0, right_rate=0.0, p9_rate=0.0):
        """Set bilateral ORN + P9 stimulus rates."""
        self.rates.zero_()
        if left_rate > 0:
            self.rates[0, self.orn_left] = left_rate
        if right_rate > 0:
            self.rates[0, self.orn_right] = right_rate
        if p9_rate > 0:
            # P9 forward walking neurons
            p9_ids = [720575940627652358, 720575940635872101]
            for fid in p9_ids:
                if fid in self.flyid2i:
                    self.rates[0, self.flyid2i[fid]] = p9_rate

    def clear_stimulus(self):
        self.rates.zero_()

    @torch.no_grad()
    def step(self):
        """One brain step with optimized spike-aware matmul."""
        cond, dbuf, spk, v, ref = self.state

        # Poisson input
        spikes_input = self.model.poisson(self.rates)

        # OPTIMIZED: spike-aware row-select matmul
        spk_np = spk.squeeze(0).numpy()
        firing = np.where(spk_np > 0)[0]

        if len(firing) > 0:
            # Only sum rows of W^T corresponding to firing neurons
            weighted = np.array(self.w_scipy_t[firing, :].sum(axis=0)).flatten()
            weighted_spikes = torch.from_numpy(weighted).unsqueeze(0).float()
        else:
            weighted_spikes = torch.zeros(1, self.num_neurons)

        # LIF update
        total_input = self.scale * (spikes_input + weighted_spikes)
        cond, dbuf, spk, v, ref = self.model.neurons(
            total_input, cond, dbuf, spk, v, ref
        )
        self.state = (cond, dbuf, spk, v, ref)
        return spk

    def get_dn_spikes(self):
        spk = self.state[2]
        return {name: spk[0, idx].item()
                for name, idx in self.dn_indices.items()}


# =====================================================================
# DN DECODER
# =====================================================================

class DNDecoder:
    def __init__(self, window_ms=50.0, dt_ms=0.1, max_rate=200.0):
        self.window_steps = int(window_ms / dt_ms)
        self.dt_s = dt_ms / 1000.0
        self.max_rate = max_rate
        self.buffers = {n: deque(maxlen=self.window_steps)
                        for n in DN_NEURONS.keys()}

    def update(self, dn_spikes):
        for name, val in dn_spikes.items():
            self.buffers[name].append(val)

    def get_normalized(self, name):
        buf = self.buffers.get(name)
        if not buf:
            return 0.0
        n = len(buf)
        window_s = n * self.dt_s
        rate = sum(buf) / window_s if window_s > 0 else 0.0
        return min(rate / self.max_rate, 1.0)

    def get_group_rate(self, group_name):
        names = DN_GROUPS.get(group_name, [])
        if not names:
            return 0.0
        return np.mean([self.get_normalized(n) for n in names])


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 72)
    print("ANALYSIS 22: OLFACTORY NAVIGATION — BRAIN-CONTROLLED FLY")
    print("=" * 72)

    torch.set_num_threads(10)

    # ── Brain ──
    print("\n[1/4] Loading optimized brain model...")
    t0 = time.time()
    brain = FastBrainEngine()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    decoder = DNDecoder(window_ms=50.0)

    # ── Body ──
    print("\n[2/4] Creating fly body...")
    body_timestep = 1e-4
    fly = Fly(enable_adhesion=True, draw_adhesion=True,
              init_pose='stretch', control='position')
    cam = Camera(
        attachment_point=fly.model.worldbody,
        camera_name='camera_right',
        targeted_fly_names=[fly.name],
        play_speed=0.2, window_size=(1280, 720), fps=30,
        timestamp_text=True, draw_contacts=True,
    )
    sim = SingleFlySimulation(fly=fly, cameras=[cam], timestep=body_timestep)

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

    # ── Simulation params ──
    sim_duration_s = 1.0
    sync_interval_ms = 15.0
    brain_steps_per_sync = int(sync_interval_ms / DT)  # 150
    body_steps_per_sync = int(sync_interval_ms / (body_timestep * 1000))  # 150
    n_syncs = int(sim_duration_s * 1000 / sync_interval_ms)

    print(f"\n[3/4] Running olfactory navigation...")
    print(f"  Duration: {sim_duration_s}s, Syncs: {n_syncs}")
    print(f"  Brain steps/sync: {brain_steps_per_sync}")

    # Experiment phases (P9 always active for walking):
    # 0-100ms:    walk only (P9 on, no odor)
    # 100-500ms:  walk + LEFT odor
    # 500-700ms:  walk + BOTH odor (straight ahead)
    # 700-1000ms: walk + RIGHT odor

    positions = []
    orientations = []
    dn_history = {g: [] for g in DN_GROUPS}
    phase_labels = []

    t_start = time.time()

    for sync_i in range(n_syncs):
        t_ms = sync_i * sync_interval_ms

        if t_ms < 100:
            brain.set_stimulus(p9_rate=100.0)
            phase = 'walk'
        elif t_ms < 500:
            brain.set_stimulus(left_rate=200.0, right_rate=0.0, p9_rate=100.0)
            phase = 'smell_L'
        elif t_ms < 700:
            brain.set_stimulus(left_rate=200.0, right_rate=200.0, p9_rate=100.0)
            phase = 'smell_both'
        else:
            brain.set_stimulus(left_rate=0.0, right_rate=200.0, p9_rate=100.0)
            phase = 'smell_R'

        phase_labels.append(phase)

        # ── Brain steps ──
        total_spikes = 0
        dn_spike_count = 0
        for _ in range(brain_steps_per_sync):
            spikes = brain.step()
            total_spikes += spikes.sum().item()
            dn_spikes = brain.get_dn_spikes()
            dn_spike_count += sum(dn_spikes.values())
            decoder.update(dn_spikes)

        for g in DN_GROUPS:
            dn_history[g].append(decoder.get_group_rate(g))

        # ── DN → motor drive ──
        fwd = decoder.get_group_rate('forward')
        turn_L = decoder.get_group_rate('turn_L')
        turn_R = decoder.get_group_rate('turn_R')
        bwd = decoder.get_group_rate('backward')

        speed = max(0, fwd - bwd)
        turn = turn_L - turn_R  # positive = turn left

        left_drive = np.clip(speed - turn * 0.5, 0, 1)
        right_drive = np.clip(speed + turn * 0.5, 0, 1)

        # Modulate CPG — differential turning via left/right leg frequencies
        # Legs: LF=0, LM=1, LH=2, RF=3, RM=4, RH=5
        base_freq = 12.0
        turn_gain = 3.0  # amplify small DN turn signals

        # Amplified turn signal
        turn_amp = turn * turn_gain

        # Forward speed from DN
        avg_speed = max(0.3, speed)  # minimum walking speed

        # To turn LEFT (turn > 0): slow LEFT legs, speed up RIGHT legs
        # To turn RIGHT (turn < 0): slow RIGHT legs, speed up LEFT legs
        left_freq = base_freq * (avg_speed - turn_amp * 0.5)
        right_freq = base_freq * (avg_speed + turn_amp * 0.5)

        # Clamp to reasonable range
        cpg.intrinsic_freqs[:3] = np.clip(left_freq, 4.0, 20.0)   # LF, LM, LH
        cpg.intrinsic_freqs[3:] = np.clip(right_freq, 4.0, 20.0)  # RF, RM, RH

        # ── Body steps ──
        for _ in range(body_steps_per_sync):
            cpg.step()
            all_angles = []
            all_adhesion = []
            for i, leg in enumerate(leg_names):
                angles = preprogrammed_steps.get_joint_angles(
                    leg, cpg.curr_phases[i], cpg.curr_magnitudes[i])
                all_angles.append(angles)
                adhesion = preprogrammed_steps.get_adhesion_onoff(
                    leg, cpg.curr_phases[i])
                all_adhesion.append(adhesion)

            action = {
                'joints': np.concatenate(all_angles),
                'adhesion': np.array(all_adhesion, dtype=np.float64),
            }
            obs, reward, terminated, truncated, info = sim.step(action)
            sim.render()

        pos = obs['fly'][0].copy()
        positions.append(pos)
        orientations.append(obs['fly_orientation'].copy())

        if sync_i % 5 == 0:
            elapsed = time.time() - t_start
            print(f"  t={t_ms:6.0f}ms [{phase:>10s}] "
                  f"spikes={total_spikes:>4.0f} dn={dn_spike_count:.0f} "
                  f"fwd={fwd:.3f} tL={turn_L:.3f} tR={turn_R:.3f} "
                  f"[{elapsed:.1f}s]")

    # Save video
    video_path = str(RESULTS_DIR / '22_olfactory_navigation.mp4')
    cam.save_video(video_path)
    print(f"\nVideo: {video_path}")

    sim.close()
    total_time = time.time() - t_start

    # ── Results ──
    print(f"\n[4/4] Results")
    print(f"  Wall time: {total_time:.1f}s")
    positions = np.array(positions)
    print(f"  Start: ({positions[0, 0]:.2f}, {positions[0, 1]:.2f})")
    print(f"  End:   ({positions[-1, 0]:.2f}, {positions[-1, 1]:.2f})")

    # Heading changes per phase
    for phase_name in ['walk', 'smell_L', 'smell_both', 'smell_R']:
        mask = [i for i, p in enumerate(phase_labels) if p == phase_name]
        if len(mask) >= 2:
            start_pos = positions[mask[0], :2]
            end_pos = positions[mask[-1], :2]
            delta_y = end_pos[1] - start_pos[1]
            direction = "LEFT" if delta_y > 0.1 else ("RIGHT" if delta_y < -0.1 else "STRAIGHT")
            fwd_rates = [dn_history['forward'][i] for i in mask]
            tl_rates = [dn_history['turn_L'][i] for i in mask]
            tr_rates = [dn_history['turn_R'][i] for i in mask]
            print(f"  {phase_name:>10s}: ΔY={delta_y:+.2f}mm ({direction}) "
                  f"fwd={np.mean(fwd_rates):.3f} tL={np.mean(tl_rates):.3f} tR={np.mean(tr_rates):.3f}")

    # ── Visualization ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Analysis 22: Olfactory Navigation — Brain-Controlled Fly\n'
                 '138,639 LIF neurons | Bilateral ORN stimulation | Scipy-optimized',
                 fontsize=12, fontweight='bold')

    t_axis = np.arange(n_syncs) * sync_interval_ms

    # DN rates
    ax = axes[0, 0]
    colors = {'forward': '#2ecc71', 'turn_L': '#3498db', 'turn_R': '#e74c3c', 'backward': '#9b59b6'}
    for g, c in colors.items():
        ax.plot(t_axis, dn_history[g], color=c, label=g, linewidth=1.5)
    ax.axvspan(100, 500, alpha=0.15, color='blue', label='Smell LEFT')
    ax.axvspan(500, 700, alpha=0.15, color='green', label='Smell BOTH')
    ax.axvspan(700, 1000, alpha=0.15, color='red', label='Smell RIGHT')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Normalized DN rate')
    ax.set_title('Descending Neuron Activity')
    ax.legend(fontsize=7)

    # Trajectory
    ax = axes[0, 1]
    phase_colors = {'walk': 'gray', 'smell_L': '#3498db', 'smell_both': '#2ecc71', 'smell_R': '#e74c3c'}
    for i in range(len(positions) - 1):
        c = phase_colors.get(phase_labels[i], 'gray')
        ax.plot(positions[i:i+2, 0], positions[i:i+2, 1], color=c, linewidth=2)
    ax.plot(positions[0, 0], positions[0, 1], 'ko', markersize=10, label='Start')
    ax.plot(positions[-1, 0], positions[-1, 1], 'k*', markersize=15, label='End')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Fly Trajectory (top view)')
    ax.legend()
    ax.set_aspect('equal')

    # Y position over time (turning indicator)
    ax = axes[1, 0]
    ax.plot(t_axis, positions[:, 1], color='#2c3e50', linewidth=2)
    ax.axvspan(100, 500, alpha=0.15, color='blue')
    ax.axvspan(500, 700, alpha=0.15, color='green')
    ax.axvspan(700, 1000, alpha=0.15, color='red')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Y position (mm)')
    ax.set_title('Lateral Movement (+ = left, - = right)')

    # Turn asymmetry
    ax = axes[1, 1]
    turn_asym = [dn_history['turn_L'][i] - dn_history['turn_R'][i] for i in range(n_syncs)]
    ax.plot(t_axis, turn_asym, color='#8e44ad', linewidth=1.5)
    ax.axvspan(100, 500, alpha=0.15, color='blue')
    ax.axvspan(500, 700, alpha=0.15, color='green')
    ax.axvspan(700, 1000, alpha=0.15, color='red')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Turn asymmetry (L - R)')
    ax.set_title('Turn Signal: + = left turn, - = right turn')

    plt.tight_layout()
    fig.savefig(str(RESULTS_DIR / '22_olfactory_navigation.png'), dpi=150, bbox_inches='tight')
    print(f"Plot: {RESULTS_DIR / '22_olfactory_navigation.png'}")

    print("\n" + "=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)


if __name__ == '__main__':
    main()
