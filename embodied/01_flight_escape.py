"""
Embodied 01: Hunger → Foraging → Looming Threat → Escape Flight

Scenario:
  1. Hunger drive → P9 forward neurons (0-300ms) → fly walks seeking food
  2. Looming threat appears (LC4, 300-600ms)
  3. LC4 → connectome → Giant Fiber → takeoff
  4. Fly escapes, lands when threat ends
  5. Post-escape: brain settles, residual activity observed

Key: hunger is an internal motivation state that drives P9 rate.
Walking emerges from hunger, escape emerges from looming — both
through the connectome.

Technical:
  - 138,639 LIF neurons (FlyWire v783 connectome)
  - FlightSystem: force computation (flight.py)
  - Wing animation: geom_xmat (render-only, no physics impact)
  - Leg tuck: actuator ctrl (photo-referenced angles)
  - Full NT logging: PAM, PPL1, MBON, serotonin, octopamine, GABA, ACh, Glut
"""
import sys
# Flush output immediately — no buffering
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import mujoco
import torch
import re
import time
import csv
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# Paths
PROJECT_DIR = Path(__file__).parent.parent
FLY_BRAIN_DIR = PROJECT_DIR / 'fly-brain-embodied'
CODE_DIR = FLY_BRAIN_DIR / 'code'
DATA_DIR = FLY_BRAIN_DIR / 'data'
RESULTS_DIR = PROJECT_DIR / 'results'
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(FLY_BRAIN_DIR))

import benchmark
benchmark.COMP_PATH = str(DATA_DIR / '2025_Completeness_783.csv')
benchmark.CONN_PATH = str(DATA_DIR / '2025_Connectivity_783.parquet')
benchmark.DATA_DIR = str(DATA_DIR)

from run_pytorch import TorchModel, MODEL_PARAMS, DT, get_weights, get_hash_tables
from flight import FlightSystem, FlightState
from brain_body_bridge import DN_NEURONS, DN_GROUPS, STIMULI, DNRateDecoder

from flygym import Fly, SingleFlySimulation, Camera
from flygym.preprogrammed import get_cpg_biases
from flygym.examples.locomotion import PreprogrammedSteps, CPGNetwork

# =====================================================================
# HUNGER STATE (from Analysis 24)
# =====================================================================

class HungerState:
    """Internal hunger drive → P9 firing rate.
    hunger=0.8 → P9 ~96Hz → fast walking
    hunger=0.1 → fly stops
    """
    def __init__(self, initial=0.8):
        self.level = initial

    @property
    def p9_rate(self):
        return self.level * 120.0  # 0-120 Hz

# =====================================================================
# CONFIG
# =====================================================================

WING_FREQ = 200.0
WING_AMPLITUDE = np.radians(30)
SIM_DURATION_S = 1.5
BODY_TIMESTEP = 1e-4
SYNC_INTERVAL_MS = 15.0

FLIGHT_LEG_ANGLES = {
    'LF': {'Coxa': 35, 'Coxa_roll': 76, 'Coxa_yaw': -6,
            'Femur': -100, 'Femur_roll': 53, 'Tibia': 80, 'Tarsus1': -20},
    'LM': {'Coxa': 23, 'Coxa_roll': 107, 'Coxa_yaw': 46,
            'Femur': -60, 'Femur_roll': -11, 'Tibia': 70, 'Tarsus1': -20},
    'LH': {'Coxa': 28, 'Coxa_roll': 143, 'Coxa_yaw': 25,
            'Femur': -70, 'Femur_roll': -24, 'Tibia': 50, 'Tarsus1': 0},
    'RF': {'Coxa': 35, 'Coxa_roll': -76, 'Coxa_yaw': 6,
            'Femur': -100, 'Femur_roll': -53, 'Tibia': 80, 'Tarsus1': -20},
    'RM': {'Coxa': 23, 'Coxa_roll': -107, 'Coxa_yaw': -46,
            'Femur': -60, 'Femur_roll': 11, 'Tibia': 70, 'Tarsus1': -20},
    'RH': {'Coxa': 28, 'Coxa_roll': -143, 'Coxa_yaw': -25,
            'Femur': -70, 'Femur_roll': 24, 'Tibia': 50, 'Tarsus1': 0},
}

# =====================================================================
# BRAIN ENGINE
# =====================================================================

class BrainEngine:
    def __init__(self, device='cpu'):
        self.device = device
        self.dt = DT
        comp_path = str(DATA_DIR / '2025_Completeness_783.csv')
        conn_path = str(DATA_DIR / '2025_Connectivity_783.parquet')
        self.flyid2i, self.i2flyid = get_hash_tables(comp_path)
        self.num_neurons = len(self.flyid2i)

        print(f"  Loading weights for {self.num_neurons:,} neurons...")
        weights = get_weights(conn_path, comp_path, str(DATA_DIR))
        weights = weights.to(device=self.device)
        self.model = TorchModel(
            batch=1, size=self.num_neurons, dt=self.dt,
            params=MODEL_PARAMS, weights=weights, device=self.device,
        )
        self.state = self.model.state_init()
        self.rates = torch.zeros(1, self.num_neurons, device=self.device)

        # Stimulus mapping
        self.stim_indices = {}
        for stim_name, stim_info in STIMULI.items():
            indices = [self.flyid2i[nid] for nid in stim_info['neurons'] if nid in self.flyid2i]
            self.stim_indices[stim_name] = indices
            print(f"  Stimulus '{stim_name}': {len(indices)}/{len(stim_info['neurons'])} mapped")

        # DN mapping
        self.dn_indices = {}
        for name, flyid in DN_NEURONS.items():
            if flyid in self.flyid2i:
                self.dn_indices[name] = self.flyid2i[flyid]
        print(f"  DN mapped: {len(self.dn_indices)}/{len(DN_NEURONS)}")

        # Population indices for NT logging
        import pandas as pd
        ann = pd.read_csv(PROJECT_DIR / "data/neuron_annotations.tsv", sep="\t", low_memory=False)

        # Dopamine
        self.pam_indices = [self.flyid2i[nid] for nid in
            ann[ann['cell_type'].str.startswith('PAM', na=False)]['root_id'] if nid in self.flyid2i]
        self.ppl1_indices = [self.flyid2i[nid] for nid in
            ann[ann['cell_type'].str.startswith('PPL1', na=False)]['root_id'] if nid in self.flyid2i]

        # MBON
        mbon = ann[ann['cell_class'] == 'MBON']
        self.mbon_approach_indices = [self.flyid2i[nid] for nid in
            mbon[mbon['top_nt'] == 'acetylcholine']['root_id'] if nid in self.flyid2i]
        self.mbon_avoidance_indices = [self.flyid2i[nid] for nid in
            mbon[mbon['top_nt'] == 'glutamate']['root_id'] if nid in self.flyid2i]
        self.mbon_suppress_indices = [self.flyid2i[nid] for nid in
            mbon[mbon['top_nt'] == 'gaba']['root_id'] if nid in self.flyid2i]

        # Serotonin interneurons (exclude sensory)
        ser = ann[ann['top_nt'] == 'serotonin']
        ser = ser[~ser['cell_class'].isin(['olfactory','visual','mechanosensory','unknown_sensory','gustatory'])]
        self.serotonin_indices = [self.flyid2i[nid] for nid in ser['root_id'] if nid in self.flyid2i]

        # Octopamine interneurons
        octo = ann[ann['top_nt'] == 'octopamine']
        octo = octo[~octo['cell_class'].isin(['olfactory','visual','mechanosensory','unknown_sensory','gustatory'])]
        self.octopamine_indices = [self.flyid2i[nid] for nid in octo['root_id'] if nid in self.flyid2i]

        # Bulk NT populations
        self.gaba_indices = [self.flyid2i[nid] for nid in
            ann[ann['top_nt'] == 'gaba']['root_id'] if nid in self.flyid2i]
        self.ach_indices = [self.flyid2i[nid] for nid in
            ann[ann['top_nt'] == 'acetylcholine']['root_id'] if nid in self.flyid2i]
        self.glut_indices = [self.flyid2i[nid] for nid in
            ann[ann['top_nt'] == 'glutamate']['root_id'] if nid in self.flyid2i]

        # Convert to torch tensors for fast indexing (no Python loops)
        self.pop_tensors = {
            'pam': torch.tensor(self.pam_indices, dtype=torch.long, device=device),
            'ppl1': torch.tensor(self.ppl1_indices, dtype=torch.long, device=device),
            'mbon_approach': torch.tensor(self.mbon_approach_indices, dtype=torch.long, device=device),
            'mbon_avoidance': torch.tensor(self.mbon_avoidance_indices, dtype=torch.long, device=device),
            'mbon_suppress': torch.tensor(self.mbon_suppress_indices, dtype=torch.long, device=device),
            'serotonin': torch.tensor(self.serotonin_indices, dtype=torch.long, device=device),
            'octopamine': torch.tensor(self.octopamine_indices, dtype=torch.long, device=device),
            'gaba': torch.tensor(self.gaba_indices, dtype=torch.long, device=device),
            'ach': torch.tensor(self.ach_indices, dtype=torch.long, device=device),
            'glut': torch.tensor(self.glut_indices, dtype=torch.long, device=device),
        }

        print(f"  PAM: {len(self.pam_indices)}, PPL1: {len(self.ppl1_indices)}")
        print(f"  MBON app/avd/sup: {len(self.mbon_approach_indices)}/{len(self.mbon_avoidance_indices)}/{len(self.mbon_suppress_indices)}")
        print(f"  Serotonin: {len(self.serotonin_indices)}, Octopamine: {len(self.octopamine_indices)}")
        print(f"  GABA: {len(self.gaba_indices)}, ACh: {len(self.ach_indices)}, Glut: {len(self.glut_indices)}")

    def set_stimulus(self, stim_name, rate_hz=None):
        self.rates.zero_()
        if stim_name and stim_name in self.stim_indices:
            rate = rate_hz or STIMULI[stim_name]['rate']
            for idx in self.stim_indices[stim_name]:
                self.rates[0, idx] = rate

    def clear_stimulus(self):
        self.rates.zero_()

    @torch.no_grad()
    def step(self):
        cond, dbuf, spk, v, ref = self.state
        self.state = self.model(self.rates, cond, dbuf, spk, v, ref)
        return self.state[2]

    def get_dn_spikes(self):
        spk = self.state[2]
        return {name: spk[0, idx].item() for name, idx in self.dn_indices.items()}

    def get_population_spikes(self):
        """Fast population spike counts using torch tensor indexing."""
        spk = self.state[2][0]  # (num_neurons,)
        return {name: spk[idx].sum().item() for name, idx in self.pop_tensors.items()}

    def get_total_spikes(self):
        return self.state[2].sum().item()


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 72)
    print("EMBODIED 01: HUNGER → FORAGING → ESCAPE FLIGHT")
    print("Food odor → brain walks → looming → GF → takeoff")
    print("=" * 72)

    # ── Brain ──
    print("\n[1/5] Loading brain...")
    t0 = time.time()
    torch.set_num_threads(10)
    brain = BrainEngine(device='cpu')
    print(f"  Loaded in {time.time()-t0:.1f}s")

    decoder = DNRateDecoder(window_ms=50.0, dt_ms=DT, max_rate=200.0)

    # ── Body ──
    print("\n[2/5] Creating body...")
    fly = Fly(enable_adhesion=True, init_pose='stretch', control='position')
    cam = Camera(
        attachment_point=fly.model.worldbody, camera_name='camera_back_track',
        targeted_fly_names=[fly.name], play_speed=0.2, window_size=(1280, 720),
        fps=30, timestamp_text=True, draw_contacts=True,
    )
    sim = SingleFlySimulation(fly=fly, cameras=[cam], timestep=BODY_TIMESTEP)

    cpg = CPGNetwork(
        timestep=BODY_TIMESTEP, intrinsic_freqs=np.ones(6)*12, intrinsic_amps=np.ones(6),
        coupling_weights=(get_cpg_biases('tripod')>0).astype(float)*10,
        phase_biases=get_cpg_biases('tripod'), convergence_coefs=np.ones(6)*20,
    )
    preprogrammed = PreprogrammedSteps()
    leg_names = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']

    model_ptr = sim.physics.model.ptr
    data_ptr = sim.physics.data.ptr
    fly_name = fly.name

    thorax_id = mujoco.mj_name2id(model_ptr, mujoco.mjtObj.mjOBJ_BODY, f"{fly_name}/Thorax")
    lwing_geom = mujoco.mj_name2id(model_ptr, mujoco.mjtObj.mjOBJ_GEOM, f"{fly_name}/LWing")
    rwing_geom = mujoco.mj_name2id(model_ptr, mujoco.mjtObj.mjOBJ_GEOM, f"{fly_name}/RWing")

    fly_mass = sum(float(model_ptr.body_mass[bid])
                   for bid in range(model_ptr.nbody)
                   if (mujoco.mj_id2name(model_ptr, mujoco.mjtObj.mjOBJ_BODY, bid) or "").startswith(f"{fly_name}/"))
    gravity_mm = float(abs(model_ptr.opt.gravity[2]))
    mg = fly_mass * gravity_mm

    free_qpos_adr = free_dof_adr = None
    for jid in range(model_ptr.njnt):
        if model_ptr.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            jbody = model_ptr.jnt_bodyid[jid]
            if fly_name in (mujoco.mj_id2name(model_ptr, mujoco.mjtObj.mjOBJ_BODY, jbody) or ""):
                free_qpos_adr = int(model_ptr.jnt_qposadr[jid])
                free_dof_adr = int(model_ptr.jnt_dofadr[jid])
                break

    flight = FlightSystem(total_mass=fly_mass, gravity=gravity_mm,
                          takeoff_thresh=0.06, land_thresh=0.06)

    # Tuck angles
    tuck_angles = np.zeros(len(fly.actuators))
    for i, act in enumerate(fly.actuators):
        match = re.search(r'joint_([A-Z]{2})(\w+)', str(act))
        if not match: continue
        leg, joint = match.group(1), match.group(2)
        if leg in FLIGHT_LEG_ANGLES and joint in FLIGHT_LEG_ANGLES[leg]:
            tuck_angles[i] = np.radians(FLIGHT_LEG_ANGLES[leg][joint])

    print(f"  Mass: {fly_mass*1e6:.0f}mg")

    # ── Simulation params ──
    brain_steps_per_sync = int(SYNC_INTERVAL_MS / DT)
    body_steps_per_sync = int(SYNC_INTERVAL_MS / (BODY_TIMESTEP * 1000))
    n_syncs = int(SIM_DURATION_S * 1000 / SYNC_INTERVAL_MS)

    hunger = HungerState(initial=0.8)

    print(f"\n[3/5] Scenario: {SIM_DURATION_S}s, {n_syncs} syncs")
    print(f"  0-300ms:   Hunger drive (P9 @ {hunger.p9_rate:.0f}Hz) → walking")
    print(f"  300-600ms: Looming threat (LC4 104 neurons) → GF → escape flight")
    print(f"  600ms+:    All stimuli off → brain settles")

    obs, _ = sim.reset()

    # ── Data recording ──
    LOG_KEYS = [
        't_ms', 'phase', 'flight_state', 'alt_mm', 'pos_x', 'pos_y', 'pos_z',
        'dn_escape', 'dn_forward', 'dn_backward', 'dn_turn_L', 'dn_turn_R',
        'dn_groom', 'dn_feed',
        'total_spikes', 'pam', 'ppl1',
        'mbon_approach', 'mbon_avoidance', 'mbon_suppress',
        'serotonin', 'octopamine', 'gaba', 'ach', 'glut',
        'wing_freq',
    ]
    log_rows = []
    positions = []
    phase_labels = []

    print(f"\n[4/5] Running...")
    t_start = time.time()
    is_flying = False

    for sync_i in range(n_syncs):
        t_ms = sync_i * SYNC_INTERVAL_MS

        # ── Stimulus phases ──
        if t_ms < 300:
            brain.set_stimulus('p9', rate_hz=hunger.p9_rate)  # hunger → P9
            phase = 'hunger'
        elif t_ms < 600:
            brain.set_stimulus('lc4', rate_hz=200.0)  # looming threat
            phase = 'looming'
        else:
            brain.clear_stimulus()
            phase = 'free'
        phase_labels.append(phase)

        # ── Brain steps — accumulate spikes via tensor addition ──
        sync_spike_acc = torch.zeros(brain.num_neurons, device=brain.device)
        for _ in range(brain_steps_per_sync):
            brain.step()
            decoder.update(brain.get_dn_spikes())
            sync_spike_acc += brain.state[2][0]  # fast tensor add, no Python loop

        # Read accumulated population spikes (one indexing op per population)
        sync_total = sync_spike_acc.sum().item()
        sync_pop = {name: sync_spike_acc[idx].sum().item()
                    for name, idx in brain.pop_tensors.items()}

        # DN rates
        dn_rates = {g: decoder.get_group_rate(g) for g in DN_GROUPS}

        # ── Flight system ──
        fly_pos = obs['fly'][0]
        fly_fwd = obs.get('fly_orientation', np.array([1, 0, 0]))
        flight.update(decoder, fly_pos, fly_fwd, SYNC_INTERVAL_MS / 1000.0)
        is_flying = flight.is_airborne

        # ── Body steps ──
        for body_step in range(body_steps_per_sync):
            t_body_s = (sync_i * body_steps_per_sync + body_step) * BODY_TIMESTEP

            if is_flying:
                data_ptr.xfrc_applied[thorax_id, :] = flight.force_torque
                obs, _, _, _, _ = sim.step({'joints': tuck_angles, 'adhesion': np.zeros(6)})

                if free_qpos_adr is not None:
                    data_ptr.qpos[free_qpos_adr+3:free_qpos_adr+7] = flight.get_desired_quat()
                    data_ptr.qvel[free_dof_adr+3:free_dof_adr+6] = 0
                    if flight.state == FlightState.LANDING:
                        vz = data_ptr.qvel[free_dof_adr+2]
                        if vz > 0: data_ptr.qvel[free_dof_adr+2] = vz * 0.95
                        data_ptr.qvel[free_dof_adr] *= 0.98
                        data_ptr.qvel[free_dof_adr+1] *= 0.98

                # Wing animation (geom_xmat only)
                angle = WING_AMPLITUDE * np.sin(2*np.pi*WING_FREQ*t_body_s)
                for gid, sign in [(lwing_geom, 1), (rwing_geom, -1)]:
                    cur = R.from_matrix(data_ptr.geom_xmat[gid].reshape(3,3))
                    data_ptr.geom_xmat[gid] = (cur * R.from_euler('y', sign*angle)).as_matrix().flatten()
            else:
                data_ptr.xfrc_applied[thorax_id, :] = 0
                avg_drive = dn_rates.get('forward', 0)
                cpg.intrinsic_freqs[:] = 12.0 * (0.5 + max(avg_drive, 0))
                cpg.step()
                aa = [preprogrammed.get_joint_angles(l, cpg.curr_phases[i], cpg.curr_magnitudes[i])
                      for i, l in enumerate(leg_names)]
                ad = [preprogrammed.get_adhesion_onoff(l, cpg.curr_phases[i])
                      for i, l in enumerate(leg_names)]
                obs, _, _, _, _ = sim.step({
                    'joints': np.concatenate(aa),
                    'adhesion': np.array(ad, dtype=np.float64),
                })

            sim.render()

        # ── Record ──
        pos = obs['fly'][0].copy()
        positions.append(pos)

        row = {
            't_ms': t_ms, 'phase': phase,
            'flight_state': FlightState(flight.state).name,
            'alt_mm': f"{flight.altitude:.2f}",
            'pos_x': f"{pos[0]:.2f}", 'pos_y': f"{pos[1]:.2f}", 'pos_z': f"{pos[2]:.2f}",
            'dn_escape': f"{dn_rates['escape']:.4f}",
            'dn_forward': f"{dn_rates['forward']:.4f}",
            'dn_backward': f"{dn_rates['backward']:.4f}",
            'dn_turn_L': f"{dn_rates['turn_L']:.4f}",
            'dn_turn_R': f"{dn_rates['turn_R']:.4f}",
            'dn_groom': f"{dn_rates['groom']:.4f}",
            'dn_feed': f"{dn_rates['feed']:.4f}",
            'total_spikes': f"{sync_total:.0f}",
            'wing_freq': f"{flight.wing_freq:.0f}",
        }
        row.update({k: f"{sync_pop[k]:.0f}" for k in sync_pop})
        log_rows.append(row)

        # Console
        if sync_i % 5 == 0:
            elapsed = time.time() - t_start
            fs = FlightState(flight.state).name
            print(f"  t={t_ms:6.0f}ms [{phase:>7s}] "
                  f"GF={dn_rates['escape']:.3f} fwd={dn_rates['forward']:.3f} "
                  f"PAM={sync_pop['pam']:.0f} PPL1={sync_pop['ppl1']:.0f} "
                  f"5HT={sync_pop['serotonin']:.0f} OA={sync_pop['octopamine']:.0f} "
                  f"| {fs:>8s} alt={flight.altitude:.1f} "
                  f"[{elapsed:.0f}s]")

    # ── Save video ──
    video_path = str(RESULTS_DIR / '28_flight_escape.mp4')
    cam.save_video(video_path)
    sim.close()
    total_time = time.time() - t_start

    # ── Save CSV log ──
    log_path = str(RESULTS_DIR / '28_flight_escape_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=LOG_KEYS)
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\n  Log: {log_path}")

    # ── Results ──
    positions = np.array(positions)
    altitudes = [float(r['alt_mm']) for r in log_rows]
    gf_hist = [float(r['dn_escape']) for r in log_rows]
    fwd_hist = [float(r['dn_forward']) for r in log_rows]
    max_alt = max(altitudes)

    takeoff_ms = None
    for r in log_rows:
        if r['flight_state'] == 'TAKEOFF' and takeoff_ms is None:
            takeoff_ms = float(r['t_ms'])

    hunger_start = 0; loom_start = 300
    reaction_ms = takeoff_ms - loom_start if takeoff_ms else None

    print(f"\n[5/5] Results (wall time: {total_time:.0f}s)")
    print(f"""
  +----------------------------------------------------+
  |  ESCAPE FLIGHT RESULTS                             |
  +----------------------------------------------------+
  |  Hunger drive start:  {hunger_start:>6.0f} ms                    |
  |  Looming start:       {loom_start:>6.0f} ms                    |
  |  Takeoff time:        {str(takeoff_ms) if takeoff_ms else 'N/A':>6} ms                    |
  |  Reaction time:       {f'{reaction_ms:.0f}' if reaction_ms else 'N/A':>6} ms                    |
  |  Max altitude:        {max_alt:>6.1f} mm                    |
  |  GF peak rate:        {max(gf_hist):>6.4f}                      |
  |  P9 post-escape:      {fwd_hist[-1]:>6.4f} (brain-driven?)     |
  +----------------------------------------------------+
""")

    # ── Figure (9 panels, all English) ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Embodied 01: Hunger → Foraging → Escape Flight\n'
                 '138,639 LIF neurons | FlyWire v783 | NeuroMechFly v2',
                 fontsize=13, fontweight='bold')

    t_ax = np.arange(n_syncs) * SYNC_INTERVAL_MS

    def shade(ax):
        ax.axvspan(0, 300, alpha=0.1, color='orange', label='Hunger drive')
        ax.axvspan(300, 600, alpha=0.15, color='red', label='Looming')

    # 1: DN rates
    ax = axes[0, 0]
    ax.plot(t_ax, gf_hist, 'r-', label='GF (escape)', lw=2)
    ax.plot(t_ax, fwd_hist, 'g-', label='P9 (forward)', lw=1.5)
    ax.plot(t_ax, [float(r['dn_backward']) for r in log_rows], 'purple', label='MDN (backward)', lw=1)
    ax.plot(t_ax, [float(r['dn_groom']) for r in log_rows], 'c-', label='aDN1 (groom)', lw=1, alpha=0.7)
    ax.plot(t_ax, [float(r['dn_feed']) for r in log_rows], 'm-', label='MN9 (feed)', lw=1, alpha=0.7)
    shade(ax)
    if takeoff_ms: ax.axvline(takeoff_ms, color='orange', ls='--', label=f'Takeoff ({takeoff_ms:.0f}ms)')
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Normalized DN Rate')
    ax.set_title('Descending Neuron Activity'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    # 2: Altitude
    ax = axes[0, 1]
    ax.plot(t_ax, altitudes, 'b-', lw=2); shade(ax)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Altitude (mm)')
    ax.set_title(f'Flight Altitude (max: {max_alt:.1f}mm)'); ax.grid(True, alpha=0.3)

    # 3: Turn L/R
    ax = axes[0, 2]
    ax.plot(t_ax, [float(r['dn_turn_L']) for r in log_rows], 'b-', label='Turn Left', lw=1.5)
    ax.plot(t_ax, [float(r['dn_turn_R']) for r in log_rows], 'r-', label='Turn Right', lw=1.5)
    shade(ax)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Normalized DN Rate')
    ax.set_title('Turn Commands (L/R)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 4: PAM/PPL1
    ax = axes[1, 0]
    ax.plot(t_ax, [float(r['pam']) for r in log_rows], '#2ecc71', label=f'PAM reward ({len(brain.pam_indices)})', lw=2)
    ax.plot(t_ax, [float(r['ppl1']) for r in log_rows], '#e74c3c', label=f'PPL1 punishment ({len(brain.ppl1_indices)})', lw=2)
    shade(ax)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Spikes / 15ms')
    ax.set_title('Dopamine: PAM vs PPL1'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 5: MBON valence
    ax = axes[1, 1]
    ax.plot(t_ax, [float(r['mbon_approach']) for r in log_rows], 'g-', label=f'Approach ACh ({len(brain.mbon_approach_indices)})', lw=2)
    ax.plot(t_ax, [float(r['mbon_avoidance']) for r in log_rows], 'r-', label=f'Avoidance Glut ({len(brain.mbon_avoidance_indices)})', lw=2)
    ax.plot(t_ax, [float(r['mbon_suppress']) for r in log_rows], 'b-', label=f'Suppress GABA ({len(brain.mbon_suppress_indices)})', lw=1.5)
    shade(ax)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Spikes / 15ms')
    ax.set_title('MBON Valence'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 6: Serotonin + Octopamine
    ax = axes[1, 2]
    ax.plot(t_ax, [float(r['serotonin']) for r in log_rows], '#9b59b6', label=f'Serotonin ({len(brain.serotonin_indices)})', lw=2)
    ax.plot(t_ax, [float(r['octopamine']) for r in log_rows], '#e67e22', label=f'Octopamine ({len(brain.octopamine_indices)})', lw=2)
    shade(ax)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Spikes / 15ms')
    ax.set_title('Neuromodulators: Serotonin & Octopamine'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 7: GABA / ACh / Glut (bulk NT)
    ax = axes[2, 0]
    ax.plot(t_ax, [float(r['gaba']) for r in log_rows], 'b-', label=f'GABA ({len(brain.gaba_indices)})', lw=1.5)
    ax.plot(t_ax, [float(r['ach']) for r in log_rows], 'g-', label=f'ACh ({len(brain.ach_indices)})', lw=1.5)
    ax.plot(t_ax, [float(r['glut']) for r in log_rows], 'r-', label=f'Glut ({len(brain.glut_indices)})', lw=1.5)
    shade(ax)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Spikes / 15ms')
    ax.set_title('Bulk Neurotransmitters'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 8: Trajectory
    ax = axes[2, 1]
    colors_ph = {'hunger': '#f39c12', 'looming': '#e74c3c', 'free': '#3498db'}
    for i in range(len(positions)-1):
        ax.plot(positions[i:i+2, 0], positions[i:i+2, 1],
                color=colors_ph.get(phase_labels[i], '#95a5a6'), lw=2)
    ax.plot(positions[0,0], positions[0,1], 'ko', ms=10, label='Start')
    ax.plot(positions[-1,0], positions[-1,1], 'k*', ms=15, label='End')
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
    ax.set_title('Trajectory (top view)'); ax.legend(fontsize=8); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    # 9: Total brain activity
    ax = axes[2, 2]
    ax.plot(t_ax, [float(r['total_spikes']) for r in log_rows], 'k-', lw=1.5)
    shade(ax)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Total spikes / 15ms')
    ax.set_title('Whole-Brain Activity (138K neurons)'); ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = str(RESULTS_DIR / '28_flight_escape.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Figure: {fig_path}")
    print(f"  Video: {video_path}")
    print(f"\n{'='*72}")
    print("COMPLETE")
    print(f"{'='*72}")


if __name__ == '__main__':
    main()
