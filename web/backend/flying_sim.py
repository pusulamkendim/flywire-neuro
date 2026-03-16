"""
Flying simulation: Walk → Takeoff → Flight → Landing → Walk

Wing beat kinematics from flybody's WingBeatPatternGenerator (218Hz, 3-DOF).
Leg tuck angles from photo-referenced Drosophila flight posture.
Flight forces via xfrc_applied on Thorax (like embodied/00_flight_camera_test.py).

Architecture: same as walking_sim.py (MuJoCo + cache + WebSocket stream).
"""

import os
os.environ.setdefault('MUJOCO_GL', 'disabled')

import sys
import time
import json
import asyncio
import threading
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
FLY_BRAIN_DIR = PROJECT_DIR / 'fly-brain-embodied'
CACHE_DIR = Path(__file__).resolve().parent / 'walk_cache'

for p in [str(FLY_BRAIN_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

LEG_NAMES = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']

# Photo-referenced flight leg angles (from embodied/00_flight_camera_test.py)
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


def _generate_flight():
    """
    Walk 0.3s → Takeoff 0.3s → Fly 1.5s → Land 0.4s → Walk 0.5s = 3.0s
    Returns (geom_names, frames).
    """
    import mujoco
    import re
    from flygym import Fly
    from flygym.simulation import SingleFlySimulation
    from flygym.preprogrammed import get_cpg_biases
    from flygym.examples.locomotion import PreprogrammedSteps, CPGNetwork
    from flybody.tasks.pattern_generators import WingBeatPatternGenerator

    fly = Fly(enable_adhesion=True, init_pose='stretch', control='position')
    sim = SingleFlySimulation(fly=fly, timestep=1e-4)
    sim.reset()

    steps = PreprogrammedSteps()
    cpg = CPGNetwork(
        timestep=1e-4,
        intrinsic_freqs=np.ones(6) * 12.0,
        intrinsic_amps=np.ones(6),
        coupling_weights=(get_cpg_biases('tripod') > 0).astype(float) * 10.0,
        phase_biases=get_cpg_biases('tripod'),
        convergence_coefs=np.ones(6) * 20.0,
    )

    # Wing beat pattern generator from flybody (218Hz, 3-DOF per wing)
    wbpg = WingBeatPatternGenerator(dt_ctrl=1e-4)
    wbpg.reset(ctrl_freq=218.0)

    model_ptr = sim.physics.model.ptr
    data_ptr = sim.physics.data.ptr
    fly_name = fly.name

    # Geom map
    geom_map = {}
    for gid in range(model_ptr.ngeom):
        gname = mujoco.mj_id2name(model_ptr, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
        if gname.startswith(f'{fly_name}/') and model_ptr.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            geom_map[gname.replace(f'{fly_name}/', '')] = gid

    thorax_bid = mujoco.mj_name2id(model_ptr, mujoco.mjtObj.mjOBJ_BODY, f'{fly_name}/Thorax')
    thorax_id = mujoco.mj_name2id(model_ptr, mujoco.mjtObj.mjOBJ_BODY, f'{fly_name}/Thorax')
    geom_names = list(geom_map.keys())
    geom_ids = list(geom_map.values())

    # Wing geom IDs
    lwing_gid = geom_map.get('LWing', -1)
    rwing_gid = geom_map.get('RWing', -1)

    # Flight mass and gravity
    fly_mass = sum(
        float(model_ptr.body_mass[bid])
        for bid in range(model_ptr.nbody)
        if (mujoco.mj_id2name(model_ptr, mujoco.mjtObj.mjOBJ_BODY, bid) or "").startswith(f"{fly_name}/")
    )
    gravity = float(abs(model_ptr.opt.gravity[2]))
    mg = fly_mass * gravity

    # Free joint
    free_qpos_adr = free_dof_adr = None
    for jid in range(model_ptr.njnt):
        if model_ptr.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            jbody = model_ptr.jnt_bodyid[jid]
            if fly_name in (mujoco.mj_id2name(model_ptr, mujoco.mjtObj.mjOBJ_BODY, jbody) or ""):
                free_qpos_adr = int(model_ptr.jnt_qposadr[jid])
                free_dof_adr = int(model_ptr.jnt_dofadr[jid])
                break

    # Tuck angles
    tuck_angles = np.zeros(len(fly.actuators))
    for i, act in enumerate(fly.actuators):
        match = re.search(r'joint_([A-Z]{2})(\w+)', str(act))
        if not match:
            continue
        leg, joint = match.group(1), match.group(2)
        if leg in FLIGHT_LEG_ANGLES and joint in FLIGHT_LEG_ANGLES[leg]:
            tuck_angles[i] = np.radians(FLIGHT_LEG_ANGLES[leg][joint])

    # Phase timing (ms)
    WALK1_END = 300
    TAKEOFF_END = 600
    FLY_END = 2100
    LAND_END = 2500
    TOTAL_MS = 3000

    timestep = 1e-4
    n_steps = int(TOTAL_MS / 1000.0 / timestep)
    emit_every = 333

    frames = []
    fly_start_pos = None
    fly_start_z = float(data_ptr.qpos[free_qpos_adr + 2]) if free_qpos_adr else 0
    is_flying = False

    print(f"[Flight] Generating {TOTAL_MS}ms flight sim ({n_steps} steps)...", flush=True)
    t0 = time.time()

    for step_i in range(n_steps):
        t_ms = step_i * timestep * 1000
        t_s = step_i * timestep

        if t_ms < WALK1_END:
            # Walking
            phase = 'walking'
            is_flying = False
            cpg.step()
            joints = [steps.get_joint_angles(leg, cpg.curr_phases[i], cpg.curr_magnitudes[i])
                      for i, leg in enumerate(LEG_NAMES)]
            adhesion = [steps.get_adhesion_onoff(leg, cpg.curr_phases[i])
                        for i, leg in enumerate(LEG_NAMES)]
            action = {'joints': np.concatenate(joints),
                      'adhesion': np.array(adhesion, dtype=np.float64)}
            data_ptr.xfrc_applied[thorax_id, :] = 0

        elif t_ms < TAKEOFF_END:
            # Takeoff — direct position control (no force fights)
            phase = 'takeoff'
            is_flying = True
            action = {'joints': tuck_angles, 'adhesion': np.zeros(6)}
            # Smooth rise: 0 → 5mm over 300ms
            progress = (t_ms - WALK1_END) / (TAKEOFF_END - WALK1_END)
            target_z = 5.0 * progress
            data_ptr.xfrc_applied[thorax_id, :] = [0, 0, mg, 0, 0, 0]  # hover
            data_ptr.qpos[free_qpos_adr + 2] = fly_start_z + target_z
            data_ptr.qvel[free_dof_adr + 2] = 15.0 * (1 - progress)  # decelerating
            data_ptr.qvel[free_dof_adr + 0] = 5.0  # gentle forward

        elif t_ms < FLY_END:
            # Cruising — hold altitude, drift forward
            phase = 'flying'
            is_flying = True
            action = {'joints': tuck_angles, 'adhesion': np.zeros(6)}
            data_ptr.xfrc_applied[thorax_id, :] = [0, 0, mg, 0, 0, 0]  # hover
            data_ptr.qpos[free_qpos_adr + 2] = fly_start_z + 5.0  # lock altitude
            data_ptr.qvel[free_dof_adr + 2] = 0  # no vertical velocity
            data_ptr.qvel[free_dof_adr + 0] = 3.0  # steady forward

        elif t_ms < LAND_END:
            # Landing — smooth descent
            phase = 'landing'
            is_flying = True
            action = {'joints': tuck_angles, 'adhesion': np.zeros(6)}
            progress = (t_ms - FLY_END) / (LAND_END - FLY_END)
            target_z = 5.0 * (1 - progress)
            data_ptr.xfrc_applied[thorax_id, :] = [0, 0, mg * 0.8, 0, 0, 0]
            data_ptr.qpos[free_qpos_adr + 2] = fly_start_z + target_z
            data_ptr.qvel[free_dof_adr + 2] = -12.0 * (1 - progress)
            data_ptr.qvel[free_dof_adr + 0] = 1.0 * (1 - progress)

        else:
            # Back to walking
            phase = 'walking'
            is_flying = False
            data_ptr.xfrc_applied[thorax_id, :] = 0
            cpg.step()
            joints = [steps.get_joint_angles(leg, cpg.curr_phases[i], cpg.curr_magnitudes[i])
                      for i, leg in enumerate(LEG_NAMES)]
            adhesion = [steps.get_adhesion_onoff(leg, cpg.curr_phases[i])
                        for i, leg in enumerate(LEG_NAMES)]
            action = {'joints': np.concatenate(joints),
                      'adhesion': np.array(adhesion, dtype=np.float64)}

        # Physics step
        sim.step(action)

        # Orientation lock during flight
        if is_flying and free_qpos_adr is not None:
            data_ptr.qpos[free_qpos_adr + 3:free_qpos_adr + 7] = [1, 0, 0, 0]
            data_ptr.qvel[free_dof_adr + 3:free_dof_adr + 6] = 0
            if phase == 'landing':
                vz = data_ptr.qvel[free_dof_adr + 2]
                if vz > 0:
                    data_ptr.qvel[free_dof_adr + 2] = vz * 0.95
                data_ptr.qvel[free_dof_adr] *= 0.98
                data_ptr.qvel[free_dof_adr + 1] *= 0.98

        # Wing beat animation (render-only, via geom_xmat)
        if is_flying and lwing_gid >= 0 and rwing_gid >= 0:
            # Get wing angles from flybody's WBPG
            wing_angles = wbpg.step(218.0)  # [yaw, roll, pitch] × 2 wings
            l_yaw, l_roll, l_pitch = wing_angles[0], wing_angles[1], wing_angles[2]
            r_yaw, r_roll, r_pitch = wing_angles[3], wing_angles[4], wing_angles[5]

            # Apply to geom_xmat (render only)
            for gid, (yaw, roll, pitch) in [(lwing_gid, (l_yaw, l_roll, l_pitch)),
                                             (rwing_gid, (r_yaw, r_roll, r_pitch))]:
                base_rot = R.from_matrix(data_ptr.geom_xmat[gid].reshape(3, 3))
                wing_rot = R.from_euler('yzx', [yaw * 0.3, roll * 0.3, pitch * 0.3])
                new_rot = base_rot * wing_rot
                data_ptr.geom_xmat[gid] = new_rot.as_matrix().flatten()

        # Emit frame
        if step_i % emit_every == 0 and step_i > 0:
            thorax_pos = data_ptr.xpos[thorax_bid].copy()
            if fly_start_pos is None:
                fly_start_pos = thorax_pos.copy()

            poses = []
            for gid in geom_ids:
                xpos = data_ptr.geom_xpos[gid] - thorax_pos
                xmat = data_ptr.geom_xmat[gid].reshape(3, 3)
                poses.extend(xpos.tolist())
                if np.all(np.isfinite(xmat)):
                    q = R.from_matrix(xmat).as_quat()
                else:
                    q = [0, 0, 0, 1]
                poses.extend(q.tolist())

            fly_pos = (thorax_pos - fly_start_pos).tolist()

            frames.append({
                "t_ms": round(t_ms, 1),
                "fly_pos": [round(v, 3) for v in fly_pos],
                "poses": [round(v, 5) for v in poses],
                "phase": phase,
            })

    sim.close()
    elapsed = time.time() - t0
    print(f"[Flight] Generated {len(frames)} frames in {elapsed:.1f}s", flush=True)
    return geom_names, frames


FLIGHT_CACHE = CACHE_DIR / 'flight_3.0s.json'


class FlyingBridge:
    """Streams flight frames from cache (or generates + caches first)."""

    def __init__(self):
        self.queue = None
        self.running = False
        self._thread = None
        self._loop = None

    def start(self, loop, queue):
        if self.running:
            return
        self.queue = queue
        self._loop = loop
        self.running = True
        self._thread = threading.Thread(target=self._run_safe, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def _emit(self, data):
        try:
            future = asyncio.run_coroutine_threadsafe(self.queue.put(data), self._loop)
            future.result(timeout=5)
        except Exception as e:
            print(f"[Flight] emit error: {e}", flush=True)

    def _run_safe(self):
        try:
            self._run()
        except Exception as e:
            import traceback
            print(f"[Flight] ERROR: {e}", flush=True)
            traceback.print_exc()
            self.running = False
            self._emit({"event": "walk_end"})

    def _run(self):
        if FLIGHT_CACHE.exists():
            print(f"[Flight] Loading cache...", flush=True)
            with open(FLIGHT_CACHE) as f:
                cached = json.load(f)
            geom_names = cached['geom_names']
            frames = cached['frames']
            print(f"[Flight] Cache: {len(frames)} frames", flush=True)
        else:
            geom_names, frames = _generate_flight()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(FLIGHT_CACHE, 'w') as f:
                json.dump({'geom_names': geom_names, 'frames': frames}, f)
            print(f"[Flight] Cached: {FLIGHT_CACHE.stat().st_size // 1024}KB", flush=True)

        self._emit({"event": "walk_init", "geom_names": geom_names})

        for frame in frames:
            if not self.running:
                break
            frame["event"] = "walk_frame"
            self._emit(frame)
            time.sleep(1.0 / 30)

        self.running = False
        self._emit({"event": "walk_end"})
