"""
Embodied 02: Flight Escape v2 — Eon architecture

Changes from 01:
  - HybridTurningController (amplitude-based, drive [0,0] = stop)
  - BRAIN_RATIO=100 (1 brain step per 100 body steps, like Eon)
  - Accumulated spike logging via tensor addition
  - Wing animation (geom_xmat) + leg tuck (photo-referenced)

Scenario:
  1. Hunger drive → P9 → walking (0-300ms)
  2. Looming threat LC4 → GF → escape flight (300-600ms)
  3. All stimuli off → brain settles (600ms+)
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import mujoco
import torch
import re
import time
import csv
from pathlib import Path
from scipy.spatial.transform import Rotation as R

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

from flygym import Fly, Camera
from flygym.simulation import SingleFlySimulation
from flygym.examples.locomotion.turning_controller import HybridTurningController

# =====================================================================
# CONFIG
# =====================================================================

BODY_TIMESTEP = 1e-4          # 0.1ms
BRAIN_RATIO = 10              # 1 brain step per 10 body steps (1ms interval)
SIM_DURATION_S = 1.5
WING_FREQ = 200.0
WING_AMPLITUDE = np.radians(30)

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
# HUNGER STATE
# =====================================================================

class HungerState:
    def __init__(self, initial=0.8):
        self.level = initial

    @property
    def p9_rate(self):
        return self.level * 120.0

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

        print(f"  Loading {self.num_neurons:,} neurons...")
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
        for sn, si in STIMULI.items():
            idx = [self.flyid2i[nid] for nid in si['neurons'] if nid in self.flyid2i]
            self.stim_indices[sn] = idx

        # DN mapping
        self.dn_indices = {name: self.flyid2i[fid]
                           for name, fid in DN_NEURONS.items() if fid in self.flyid2i}

        # Population indices for NT logging (torch tensors for fast indexing)
        import pandas as pd
        ann = pd.read_csv(PROJECT_DIR / "data/neuron_annotations.tsv", sep="\t", low_memory=False)

        def ids_to_tensor(root_ids):
            return torch.tensor([self.flyid2i[nid] for nid in root_ids if nid in self.flyid2i],
                                dtype=torch.long, device=device)

        mbon = ann[ann['cell_class'] == 'MBON']
        ser = ann[ann['top_nt'] == 'serotonin']
        ser = ser[~ser['cell_class'].isin(['olfactory','visual','mechanosensory','unknown_sensory','gustatory'])]
        octo = ann[ann['top_nt'] == 'octopamine']
        octo = octo[~octo['cell_class'].isin(['olfactory','visual','mechanosensory','unknown_sensory','gustatory'])]

        self.pop_tensors = {
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

        for name, t in self.pop_tensors.items():
            print(f"  {name}: {len(t)}")

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


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 72)
    print("EMBODIED 02: FLIGHT ESCAPE v2 (Eon architecture)")
    print("HybridTurningController + BRAIN_RATIO=100")
    print("=" * 72)

    # ── Brain ──
    print("\n[1/5] Brain...")
    t0 = time.time()
    torch.set_num_threads(10)
    brain = BrainEngine(device='cpu')
    print(f"  Loaded in {time.time()-t0:.1f}s")

    decoder = DNRateDecoder(window_ms=50.0, dt_ms=DT, max_rate=200.0)
    hunger = HungerState(initial=0.8)

    # ── Body (HybridTurningController like Eon) ──
    print("\n[2/5] Body (HybridTurningController)...")
    contact_sensors = [
        f"{leg}{seg}"
        for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
        for seg in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]
    fly = Fly(enable_adhesion=True, init_pose='stretch', control='position',
              contact_sensor_placements=contact_sensors)
    cam = Camera(
        attachment_point=fly.model.worldbody, camera_name='camera_back_track',
        targeted_fly_names=[fly.name], play_speed=0.2, window_size=(1280, 720),
        fps=30, timestamp_text=True, draw_contacts=True,
    )

    sim = HybridTurningController(fly=fly, cameras=[cam], timestep=BODY_TIMESTEP)
    obs, info = sim.reset()

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

    print(f"  Mass: {fly_mass*1e6:.0f}mg, BRAIN_RATIO: {BRAIN_RATIO}")

    # ── Simulation ──
    n_body_steps = int(SIM_DURATION_S / BODY_TIMESTEP)
    brain_interval = BRAIN_RATIO  # body steps between brain steps
    log_interval = int(15.0 / (BODY_TIMESTEP * 1000))  # log every 15ms (150 steps)
    print_interval = int(75.0 / (BODY_TIMESTEP * 1000))  # print every 75ms

    print(f"\n[3/5] Simulation: {SIM_DURATION_S}s, {n_body_steps:,} body steps")
    print(f"  Brain step every {brain_interval} body steps ({brain_interval * BODY_TIMESTEP * 1000:.1f}ms)")
    print(f"  Log every {log_interval} steps ({log_interval * BODY_TIMESTEP * 1000:.0f}ms)")
    print(f"  Phases:")
    print(f"    0-300ms:   Hunger drive (P9 @ {hunger.p9_rate:.0f}Hz)")
    print(f"    300-600ms: Looming (LC4)")
    print(f"    600ms+:    Free")

    # Data recording
    LOG_KEYS = [
        't_ms', 'phase', 'flight_state', 'alt_mm', 'pos_x', 'pos_y', 'pos_z',
        'dn_escape', 'dn_forward', 'dn_backward', 'dn_turn_L', 'dn_turn_R',
        'dn_groom', 'dn_feed', 'drive_L', 'drive_R',
        'total_spikes', 'pam', 'ppl1',
        'mbon_approach', 'mbon_avoidance', 'mbon_suppress',
        'serotonin', 'octopamine', 'gaba', 'ach', 'glut',
        'wing_freq',
    ]
    log_rows = []
    spike_acc = torch.zeros(brain.num_neurons, device=brain.device)
    last_log_step = 0

    is_flying = False
    drive = np.array([0.0, 0.0])

    print(f"\n[4/5] Running...")
    t_start = time.time()

    for body_step in range(n_body_steps):
        t_ms = body_step * BODY_TIMESTEP * 1000
        t_s = body_step * BODY_TIMESTEP

        # ── Brain step (every BRAIN_RATIO body steps) ──
        if body_step % brain_interval == 0:
            # Set stimulus
            if t_ms < 300:
                brain.set_stimulus('p9', rate_hz=hunger.p9_rate)
            elif t_ms < 600:
                brain.set_stimulus('lc4', rate_hz=200.0)
            else:
                brain.clear_stimulus()

            brain.step()
            decoder.update(brain.get_dn_spikes())
            spike_acc += brain.state[2][0]

            # Update flight system
            fly_pos = obs['fly'][0]
            fly_fwd = obs.get('fly_orientation', np.array([1, 0, 0]))
            flight.update(decoder, fly_pos, fly_fwd, brain_interval * BODY_TIMESTEP)
            is_flying = flight.is_airborne

            # Compute drive from DN rates (like Eon's BrainBodyBridge)
            if not is_flying:
                fwd = decoder.get_group_rate('forward')
                turn_L = decoder.get_group_rate('turn_L')
                turn_R = decoder.get_group_rate('turn_R')
                bkw = decoder.get_group_rate('backward')
                turn = turn_L - turn_R
                left = fwd * (1.0 + turn) - bkw
                right = fwd * (1.0 - turn) - bkw
                drive = np.clip([left, right], -0.5, 1.5)
            else:
                drive = np.array([0.0, 0.0])

        # ── Body step ──
        if is_flying:
            # Flight forces
            data_ptr.xfrc_applied[thorax_id, :] = flight.force_torque

            # Tuck legs via SingleFlySimulation.step (bypass HybridTurningController)
            flight_action = {'joints': tuck_angles, 'adhesion': np.zeros(6)}
            obs, _, _, _, _ = SingleFlySimulation.step(sim, flight_action)

            # Orientation lock
            if free_qpos_adr is not None:
                data_ptr.qpos[free_qpos_adr+3:free_qpos_adr+7] = flight.get_desired_quat()
                data_ptr.qvel[free_dof_adr+3:free_dof_adr+6] = 0
                if flight.state == FlightState.LANDING:
                    vz = data_ptr.qvel[free_dof_adr+2]
                    if vz > 0: data_ptr.qvel[free_dof_adr+2] = vz * 0.95
                    data_ptr.qvel[free_dof_adr] *= 0.98
                    data_ptr.qvel[free_dof_adr+1] *= 0.98

            # Wing animation
            angle = WING_AMPLITUDE * np.sin(2*np.pi*WING_FREQ*t_s)
            for gid, sign in [(lwing_geom, 1), (rwing_geom, -1)]:
                cur = R.from_matrix(data_ptr.geom_xmat[gid].reshape(3,3))
                data_ptr.geom_xmat[gid] = (cur * R.from_euler('y', sign*angle)).as_matrix().flatten()
        else:
            # Walking via HybridTurningController (amplitude-based)
            data_ptr.xfrc_applied[thorax_id, :] = 0
            obs, _, _, _, _ = sim.step(drive)

        sim.render()

        # ── Log (every log_interval steps) ──
        if body_step % log_interval == 0 and body_step > 0:
            phase = 'hunger' if t_ms < 300 else 'looming' if t_ms < 600 else 'free'
            pos = obs['fly'][0]
            dn_rates = {g: decoder.get_group_rate(g) for g in DN_GROUPS}

            # Read accumulated spikes since last log
            sync_pop = {name: spike_acc[idx].sum().item()
                        for name, idx in brain.pop_tensors.items()}
            sync_total = spike_acc.sum().item()
            spike_acc.zero_()  # reset accumulator

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
                'drive_L': f"{drive[0]:.4f}",
                'drive_R': f"{drive[1]:.4f}",
                'total_spikes': f"{sync_total:.0f}",
                'wing_freq': f"{flight.wing_freq:.0f}",
            }
            row.update({k: f"{sync_pop[k]:.0f}" for k in sync_pop})
            log_rows.append(row)

        # ── Console print ──
        if body_step % print_interval == 0:
            elapsed = time.time() - t_start
            phase = 'hunger' if t_ms < 300 else 'looming' if t_ms < 600 else 'free'
            fs = FlightState(flight.state).name
            gf = decoder.get_group_rate('escape')
            fwd = decoder.get_group_rate('forward')
            print(f"  t={t_ms:6.0f}ms [{phase:>7s}] "
                  f"GF={gf:.3f} fwd={fwd:.3f} drive=[{drive[0]:.2f},{drive[1]:.2f}] "
                  f"| {fs:>8s} alt={flight.altitude:.1f} "
                  f"[{elapsed:.0f}s]")

    # ── Save ──
    video_path = str(RESULTS_DIR / '29_flight_escape_v2.mp4')
    cam.save_video(video_path)
    sim.close()
    total_time = time.time() - t_start

    log_path = str(RESULTS_DIR / '29_flight_escape_v2_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=LOG_KEYS)
        writer.writeheader()
        writer.writerows(log_rows)

    # Results
    gf_vals = [float(r['dn_escape']) for r in log_rows]
    fwd_vals = [float(r['dn_forward']) for r in log_rows]
    alt_vals = [float(r['alt_mm']) for r in log_rows]

    takeoff_ms = None
    for r in log_rows:
        if r['flight_state'] == 'TAKEOFF' and takeoff_ms is None:
            takeoff_ms = float(r['t_ms'])

    print(f"""
  +----------------------------------------------------+
  |  RESULTS (wall time: {total_time:.0f}s)                       |
  +----------------------------------------------------+
  |  Takeoff:       {str(takeoff_ms) if takeoff_ms else 'N/A':>6} ms                        |
  |  Reaction:      {f'{takeoff_ms-300:.0f}' if takeoff_ms else 'N/A':>6} ms (from looming)          |
  |  Max altitude:  {max(alt_vals):>6.1f} mm                        |
  |  GF peak:       {max(gf_vals):>6.3f}                            |
  |  Post-escape fwd:{fwd_vals[-1]:>6.3f} (brain-driven?)          |
  +----------------------------------------------------+
  Log: {log_path}
  Video: {video_path}
""")

    # ── Figure ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Embodied 02: Flight Escape v2 (Eon architecture)\n'
                 'HybridTurningController + BRAIN_RATIO=100 + NT logging',
                 fontsize=13, fontweight='bold')

    t_ax = np.array([float(r['t_ms']) for r in log_rows])

    def shade(ax):
        ax.axvspan(0, 300, alpha=0.1, color='orange', label='Hunger')
        ax.axvspan(300, 600, alpha=0.15, color='red', label='Looming')

    # 1: DN rates
    ax = axes[0, 0]
    ax.plot(t_ax, gf_vals, 'r-', label='GF (escape)', lw=2)
    ax.plot(t_ax, fwd_vals, 'g-', label='P9 (forward)', lw=1.5)
    ax.plot(t_ax, [float(r['dn_backward']) for r in log_rows], 'purple', label='MDN (backward)', lw=1)
    shade(ax); ax.set_xlabel('Time (ms)'); ax.set_ylabel('DN Rate')
    ax.set_title('Descending Neurons'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 2: Drive
    ax = axes[0, 1]
    ax.plot(t_ax, [float(r['drive_L']) for r in log_rows], 'b-', label='Left drive', lw=1.5)
    ax.plot(t_ax, [float(r['drive_R']) for r in log_rows], 'r-', label='Right drive', lw=1.5)
    shade(ax); ax.set_xlabel('Time (ms)'); ax.set_ylabel('Drive')
    ax.set_title('Motor Drive [L, R]'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 3: Altitude
    ax = axes[0, 2]
    ax.plot(t_ax, alt_vals, 'b-', lw=2); shade(ax)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('mm')
    ax.set_title(f'Altitude (max: {max(alt_vals):.1f}mm)'); ax.grid(True, alpha=0.3)

    # 4: PAM/PPL1
    ax = axes[1, 0]
    ax.plot(t_ax, [float(r['pam']) for r in log_rows], '#2ecc71', label='PAM reward', lw=2)
    ax.plot(t_ax, [float(r['ppl1']) for r in log_rows], '#e74c3c', label='PPL1 punishment', lw=2)
    shade(ax); ax.set_xlabel('Time (ms)'); ax.set_ylabel('Spikes')
    ax.set_title('Dopamine'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 5: MBON
    ax = axes[1, 1]
    ax.plot(t_ax, [float(r['mbon_approach']) for r in log_rows], 'g-', label='Approach', lw=2)
    ax.plot(t_ax, [float(r['mbon_avoidance']) for r in log_rows], 'r-', label='Avoidance', lw=2)
    ax.plot(t_ax, [float(r['mbon_suppress']) for r in log_rows], 'b-', label='Suppress', lw=1.5)
    shade(ax); ax.set_xlabel('Time (ms)'); ax.set_ylabel('Spikes')
    ax.set_title('MBON Valence'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 6: Serotonin + Octopamine
    ax = axes[1, 2]
    ax.plot(t_ax, [float(r['serotonin']) for r in log_rows], '#9b59b6', label='Serotonin', lw=2)
    ax.plot(t_ax, [float(r['octopamine']) for r in log_rows], '#e67e22', label='Octopamine', lw=2)
    shade(ax); ax.set_xlabel('Time (ms)'); ax.set_ylabel('Spikes')
    ax.set_title('Neuromodulators'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 7: GABA/ACh/Glut
    ax = axes[2, 0]
    ax.plot(t_ax, [float(r['gaba']) for r in log_rows], 'b-', label='GABA', lw=1.5)
    ax.plot(t_ax, [float(r['ach']) for r in log_rows], 'g-', label='ACh', lw=1.5)
    ax.plot(t_ax, [float(r['glut']) for r in log_rows], 'r-', label='Glutamate', lw=1.5)
    shade(ax); ax.set_xlabel('Time (ms)'); ax.set_ylabel('Spikes')
    ax.set_title('Bulk Neurotransmitters'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 8: Trajectory
    ax = axes[2, 1]
    positions = np.array([[float(r['pos_x']), float(r['pos_y'])] for r in log_rows])
    phases = [r['phase'] for r in log_rows]
    colors_ph = {'hunger': '#f39c12', 'looming': '#e74c3c', 'free': '#3498db'}
    for i in range(len(positions)-1):
        ax.plot(positions[i:i+2, 0], positions[i:i+2, 1],
                color=colors_ph.get(phases[i], '#95a5a6'), lw=2)
    ax.plot(positions[0,0], positions[0,1], 'ko', ms=10, label='Start')
    ax.plot(positions[-1,0], positions[-1,1], 'k*', ms=15, label='End')
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
    ax.set_title('Trajectory'); ax.legend(fontsize=8); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    # 9: Total activity
    ax = axes[2, 2]
    ax.plot(t_ax, [float(r['total_spikes']) for r in log_rows], 'k-', lw=1.5)
    shade(ax); ax.set_xlabel('Time (ms)'); ax.set_ylabel('Total spikes')
    ax.set_title('Whole-Brain Activity'); ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = str(RESULTS_DIR / '29_flight_escape_v2.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure: {fig_path}")
    print(f"\n{'='*72}\nCOMPLETE\n{'='*72}")


if __name__ == '__main__':
    main()
