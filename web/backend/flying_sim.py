"""
Flying simulation: Walk → Takeoff → Flight → Landing → Walk

Wing beat kinematics from flybody's WingBeatPatternGenerator (218Hz, 3-DOF).
Leg tuck angles from photo-referenced Drosophila flight posture.
Flight altitude is physically supported while the replay follows a deterministic
straight path, avoiding MuJoCo free-body drift between recordings.

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

# Mesh-fitted in-flight tuck pose. Front and middle tarsi fold under the thorax;
# hind legs remain partly visible behind the body for aerodynamic stability.
FLIGHT_LEG_ANGLES = {
    'LF': {'Coxa': 29.5, 'Coxa_roll': 117.7, 'Coxa_yaw': 27.3,
            'Femur': -75.4, 'Femur_roll': 100.0, 'Tibia': 132.0, 'Tarsus1': 22.2},
    'LM': {'Coxa': 37.7, 'Coxa_roll': 145.0, 'Coxa_yaw': 70.0,
            'Femur': -74.9, 'Femur_roll': 36.8, 'Tibia': 117.5, 'Tarsus1': 79.6},
    'LH': {'Coxa': 41.5, 'Coxa_roll': 138.2, 'Coxa_yaw': 59.4,
            'Femur': -104.9, 'Femur_roll': -3.3, 'Tibia': 101.8, 'Tarsus1': 85.3},
    'RF': {'Coxa': 29.5, 'Coxa_roll': -117.7, 'Coxa_yaw': -27.3,
            'Femur': -75.4, 'Femur_roll': -100.0, 'Tibia': 132.0, 'Tarsus1': 22.2},
    'RM': {'Coxa': 37.7, 'Coxa_roll': -145.0, 'Coxa_yaw': -70.0,
            'Femur': -74.9, 'Femur_roll': -36.8, 'Tibia': 117.5, 'Tarsus1': 79.6},
    'RH': {'Coxa': 41.5, 'Coxa_roll': -138.2, 'Coxa_yaw': -59.4,
            'Femur': -104.9, 'Femur_roll': 3.3, 'Tibia': 101.8, 'Tarsus1': 85.3},
}


def _generate_flight():
    """
    Walk 0.3s → Takeoff 0.4s → Straight flight 2.4s → Land 0.5s
    → Walk 0.6s = 4.2s
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

    # Standing and in-flight leg targets.
    stance_angles = np.concatenate([
        steps.get_joint_angles(leg, np.pi, 0.0) for leg in LEG_NAMES
    ])
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
    TAKEOFF_END = 700
    FLY_END = 3100
    LAND_END = 3600
    TOTAL_MS = 4200
    FLIGHT_ALTITUDE = 5.0
    FLIGHT_DISTANCE = 12.0

    timestep = 1e-4
    n_steps = int(TOTAL_MS / 1000.0 / timestep)
    emit_every = 333

    frames = []
    fly_start_pos = None
    flight_origin_xy = None
    flight_origin_z = None
    is_flying = False

    print(f"[Flight] Generating {TOTAL_MS}ms flight sim ({n_steps} steps)...", flush=True)
    t0 = time.time()

    for step_i in range(n_steps):
        t_ms = step_i * timestep * 1000
        target_position = None
        target_velocity = None
        heading_rad = 0.0
        bank_rad = 0.0

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
            # The middle legs remain extended briefly to push off while the
            # front/hind legs begin folding. All six finish in the tuck pose.
            phase = 'takeoff'
            is_flying = True
            if flight_origin_xy is None:
                flight_origin_xy = data_ptr.qpos[
                    free_qpos_adr:free_qpos_adr + 2].copy()
                flight_origin_z = float(data_ptr.qpos[free_qpos_adr + 2])
            progress = (t_ms - WALK1_END) / (TAKEOFF_END - WALK1_END)
            rise = progress * progress * (3.0 - 2.0 * progress)

            takeoff_joints = tuck_angles.copy()
            for leg_i, leg in enumerate(LEG_NAMES):
                delay = 0.24 if leg[1] == 'M' else 0.0
                fold = np.clip((progress - delay) / (1.0 - delay), 0.0, 1.0)
                fold = fold * fold * (3.0 - 2.0 * fold)
                sl = slice(leg_i * 7, (leg_i + 1) * 7)
                takeoff_joints[sl] = (
                    stance_angles[sl] * (1.0 - fold) + tuck_angles[sl] * fold
                )
            takeoff_adhesion = np.zeros(6)
            if progress < 0.24:
                takeoff_adhesion[[1, 4]] = 1  # middle-leg push-off contact
            action = {'joints': takeoff_joints, 'adhesion': takeoff_adhesion}

            target_position = np.array([
                flight_origin_xy[0],
                flight_origin_xy[1],
                flight_origin_z + FLIGHT_ALTITUDE * rise,
            ])
            target_velocity = np.zeros(3)
            data_ptr.xfrc_applied[thorax_id, :] = [0, 0, mg, 0, 0, 0]  # hover

        elif t_ms < FLY_END:
            # Stable, straight flight. The global path is deliberately simple
            # so the tucked leg configuration and wing motion remain readable.
            phase = 'flying'
            is_flying = True
            action = {'joints': tuck_angles, 'adhesion': np.zeros(6)}
            progress = (t_ms - TAKEOFF_END) / (FLY_END - TAKEOFF_END)
            target_position = np.array([
                flight_origin_xy[0] + FLIGHT_DISTANCE * progress,
                flight_origin_xy[1],
                flight_origin_z + FLIGHT_ALTITUDE + 0.18 * np.sin(2.0 * np.pi * progress),
            ])
            target_velocity = np.array([
                FLIGHT_DISTANCE / ((FLY_END - TAKEOFF_END) / 1000.0),
                0.0,
                0.18 * 2.0 * np.pi / ((FLY_END - TAKEOFF_END) / 1000.0)
                * np.cos(2.0 * np.pi * progress),
            ])
            data_ptr.xfrc_applied[thorax_id, :] = [0, 0, mg, 0, 0, 0]  # hover

        elif t_ms < LAND_END:
            # Extend all legs before touchdown. Front legs reach first, while
            # middle/hind legs follow within the same landing response.
            phase = 'landing'
            is_flying = True
            progress = (t_ms - FLY_END) / (LAND_END - FLY_END)
            descend = 1.0 - progress * progress * (3.0 - 2.0 * progress)

            landing_joints = tuck_angles.copy()
            reach_durations = {'F': 0.58, 'M': 0.76, 'H': 0.92}
            for leg_i, leg in enumerate(LEG_NAMES):
                reach = np.clip(progress / reach_durations[leg[1]], 0.0, 1.0)
                reach = reach * reach * (3.0 - 2.0 * reach)
                sl = slice(leg_i * 7, (leg_i + 1) * 7)
                landing_joints[sl] = (
                    tuck_angles[sl] * (1.0 - reach) + stance_angles[sl] * reach
                )
            action = {'joints': landing_joints, 'adhesion': np.zeros(6)}

            target_position = np.array([
                flight_origin_xy[0] + FLIGHT_DISTANCE,
                flight_origin_xy[1],
                flight_origin_z + FLIGHT_ALTITUDE * descend,
            ])
            target_velocity = np.zeros(3)
            data_ptr.xfrc_applied[thorax_id, :] = [0, 0, mg * 0.8, 0, 0, 0]

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

        # Constrain the replay root after the physics step. The joint/wing
        # dynamics still come from MuJoCo, while the global flight path remains
        # stable and reproducible.
        if is_flying and free_qpos_adr is not None:
            data_ptr.qpos[free_qpos_adr:free_qpos_adr + 3] = target_position
            data_ptr.qpos[free_qpos_adr + 3:free_qpos_adr + 7] = [1, 0, 0, 0]
            data_ptr.qvel[free_dof_adr:free_dof_adr + 3] = target_velocity
            data_ptr.qvel[free_dof_adr + 3:free_dof_adr + 6] = 0
            mujoco.mj_forward(model_ptr, data_ptr)

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
                "heading_rad": round(float(heading_rad), 5),
                "bank_rad": round(float(bank_rad), 5),
            })

    sim.close()
    elapsed = time.time() - t0
    print(f"[Flight] Generated {len(frames)} frames in {elapsed:.1f}s", flush=True)
    return geom_names, frames


FLIGHT_CACHE = CACHE_DIR / 'flight_leg_posture_v3_4.2s.json'


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

        self._emit({
            "event": "walk_init",
            "geom_names": geom_names,
            "render_mode": "flight_path",
        })

        for frame in frames:
            if not self.running:
                break
            frame["event"] = "walk_frame"
            self._emit(frame)
            time.sleep(1.0 / 30)

        self.running = False
        self._emit({"event": "walk_end"})
