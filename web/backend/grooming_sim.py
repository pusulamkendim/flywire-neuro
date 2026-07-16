"""
Grooming simulation: Walk → JO touch → front-leg grooming → walk again.

Same architecture as walking_sim.py:
- First run: MuJoCo computes frames, caches to disk
- Subsequent runs: instant replay from cache
- Streams per-geom transforms via WebSocket
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

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
FLY_BRAIN_DIR = PROJECT_DIR / 'fly-brain-embodied'
CACHE_DIR = Path(__file__).resolve().parent / 'walk_cache'

for p in [str(FLY_BRAIN_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

LEG_NAMES = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']


class GroomingController:
    """Coordinated front-leg eye/antenna grooming controller.

    The key poses were fitted against the FlyGym mesh so that Tarsus5 follows
    a path over the compound eye instead of merely oscillating below the body.
    Right-leg poses are mirrored from the left-leg poses.
    """

    GROOM_DURATION_S = 1.5

    # Actuator order: Coxa, Coxa_roll, Coxa_yaw, Femur, Femur_roll,
    # Tibia, Tarsus1. Values are degrees for readability.
    LEFT_KEY_POSES_DEG = {
        # Tarsus is lifted in front of the head before making contact.
        'ready': [-20.9, 130.6, 13.2, -58.2, 88.8, 146.3, -6.2],
        # Sweep from the lower/front edge of the eye to its upper edge.
        'lower_eye': [-22.5, 135.0, 13.9, -56.5, 95.8, 150.9, -3.8],
        'upper_eye': [-37.3, 130.8, 22.1, -55.1, 101.9, 148.1, -4.7],
        # Pull laterally away before starting the next wiping stroke.
        'outer_eye': [-29.9, 133.7, 17.5, -55.2, 98.4, 146.4, -6.0],
    }

    def __init__(self, preprogrammed_steps, freq_hz=2.4):
        self.steps = preprogrammed_steps
        self.freq = freq_hz
        self.neutral = np.zeros(42)
        for i, leg in enumerate(self.steps.legs):
            self.neutral[i * 7:(i + 1) * 7] = self.steps.get_joint_angles(
                leg, np.pi, 0.0)

        self.left_poses = {
            name: np.deg2rad(values)
            for name, values in self.LEFT_KEY_POSES_DEG.items()
        }
        self.right_poses = {
            name: self._mirror_pose(pose)
            for name, pose in self.left_poses.items()
        }

    @staticmethod
    def _mirror_pose(left_pose):
        """Mirror the lateral rotational DOFs for the right front leg."""
        right_pose = left_pose.copy()
        right_pose[[1, 2, 4]] *= -1
        return right_pose

    @staticmethod
    def _smoothstep(value):
        value = np.clip(value, 0.0, 1.0)
        return value * value * (3.0 - 2.0 * value)

    @classmethod
    def _blend(cls, start, end, amount):
        amount = cls._smoothstep(amount)
        return start + (end - start) * amount

    def _stroke_pose(self, poses, phase):
        """Interpolate a closed ready → eye wipe → release trajectory."""
        names = ('ready', 'lower_eye', 'upper_eye', 'outer_eye', 'ready')
        position = (phase % 1.0) * (len(names) - 1)
        segment = min(int(position), len(names) - 2)
        return self._blend(
            poses[names[segment]],
            poses[names[segment + 1]],
            position - segment,
        )

    def get_action(self, groom_time_s):
        joints = self.neutral.copy()

        entry_duration = 0.22
        exit_duration = 0.20
        active_time = max(0.0, groom_time_s - entry_duration)

        # Introduce the half-cycle offset gradually so both legs lift smoothly,
        # then wipe alternate eyes instead of moving as a rigid pair.
        offset_ramp = self._smoothstep(active_time / 0.18)
        left_phase = active_time * self.freq
        right_phase = left_phase + 0.5 * offset_ramp
        left_target = self._stroke_pose(self.left_poses, left_phase)
        right_target = self._stroke_pose(self.right_poses, right_phase)

        if groom_time_s < entry_duration:
            entry = groom_time_s / entry_duration
            left_target = self._blend(self.neutral[0:7], self.left_poses['ready'], entry)
            right_target = self._blend(self.neutral[21:28], self.right_poses['ready'], entry)
        elif groom_time_s > self.GROOM_DURATION_S - exit_duration:
            exit_amount = (
                groom_time_s - (self.GROOM_DURATION_S - exit_duration)
            ) / exit_duration
            left_target = self._blend(left_target, self.neutral[0:7], exit_amount)
            right_target = self._blend(right_target, self.neutral[21:28], exit_amount)

        joints[0:7] = left_target
        joints[21:28] = right_target
        adhesion = np.array([0, 1, 1, 0, 1, 1])  # front legs free, rest grounded
        return {"joints": joints, "adhesion": adhesion}


def _generate_grooming():
    """
    Run MuJoCo: walk 0.5s → groom 1.5s → walk 1.0s = 3.0s total.
    Returns (geom_names, frames).
    """
    import mujoco
    from flygym import Fly
    from flygym.simulation import SingleFlySimulation
    from flygym.preprogrammed import get_cpg_biases
    from flygym.examples.locomotion import PreprogrammedSteps, CPGNetwork
    from scipy.spatial.transform import Rotation as R

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
    groom_ctrl = GroomingController(steps)

    model_ptr = sim.physics.model.ptr
    data_ptr = sim.physics.data.ptr
    fly_name = fly.name

    geom_map = {}
    for gid in range(model_ptr.ngeom):
        gname = mujoco.mj_id2name(model_ptr, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
        if gname.startswith(f'{fly_name}/') and model_ptr.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            geom_map[gname.replace(f'{fly_name}/', '')] = gid

    thorax_bid = mujoco.mj_name2id(model_ptr, mujoco.mjtObj.mjOBJ_BODY, f'{fly_name}/Thorax')
    geom_names = list(geom_map.keys())
    geom_ids = list(geom_map.values())

    # Phase timing (ms)
    WALK1_END = 500
    GROOM_END = 2000
    TOTAL_MS = 3000

    timestep = 1e-4
    n_steps = int(TOTAL_MS / 1000.0 / timestep)
    emit_every = 333  # ~30fps

    frames = []
    fly_start_pos = None

    print(f"[Groom] Generating {TOTAL_MS}ms grooming sim ({n_steps} steps)...", flush=True)
    t0 = time.time()

    for step_i in range(n_steps):
        t_ms = step_i * timestep * 1000
        t_s = step_i * timestep

        # Phase selection
        if t_ms < WALK1_END:
            # Walking phase
            cpg.step()
            joints = [steps.get_joint_angles(leg, cpg.curr_phases[i], cpg.curr_magnitudes[i])
                      for i, leg in enumerate(LEG_NAMES)]
            adhesion = [steps.get_adhesion_onoff(leg, cpg.curr_phases[i])
                        for i, leg in enumerate(LEG_NAMES)]
            action = {
                'joints': np.concatenate(joints),
                'adhesion': np.array(adhesion, dtype=np.float64),
            }
            phase = 'walking'
        elif t_ms < GROOM_END:
            # Grooming phase — front tarsi sweep across the compound eyes.
            groom_time_s = (t_ms - WALK1_END) / 1000.0
            action = groom_ctrl.get_action(groom_time_s)
            phase = 'grooming'
        else:
            # Return to walking
            cpg.step()
            joints = [steps.get_joint_angles(leg, cpg.curr_phases[i], cpg.curr_magnitudes[i])
                      for i, leg in enumerate(LEG_NAMES)]
            adhesion = [steps.get_adhesion_onoff(leg, cpg.curr_phases[i])
                        for i, leg in enumerate(LEG_NAMES)]
            action = {
                'joints': np.concatenate(joints),
                'adhesion': np.array(adhesion, dtype=np.float64),
            }
            phase = 'walking'

        sim.step(action)

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
    print(f"[Groom] Generated {len(frames)} frames in {elapsed:.1f}s", flush=True)
    return geom_names, frames


GROOM_CACHE = CACHE_DIR / 'groom_eye_clean_v2_3.0s.json'


class GroomingBridge:
    """Streams grooming frames from cache (or generates + caches first)."""

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
            print(f"[Groom] emit error: {e}", flush=True)

    def _run_safe(self):
        try:
            self._run()
        except Exception as e:
            import traceback
            print(f"[Groom] ERROR: {e}", flush=True)
            traceback.print_exc()
            self.running = False
            self._emit({"event": "walk_end"})

    def _run(self):
        if GROOM_CACHE.exists():
            print(f"[Groom] Loading cache...", flush=True)
            with open(GROOM_CACHE) as f:
                cached = json.load(f)
            geom_names = cached['geom_names']
            frames = cached['frames']
            print(f"[Groom] Cache: {len(frames)} frames", flush=True)
        else:
            geom_names, frames = _generate_grooming()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(GROOM_CACHE, 'w') as f:
                json.dump({'geom_names': geom_names, 'frames': frames}, f)
            print(f"[Groom] Cached: {GROOM_CACHE.stat().st_size // 1024}KB", flush=True)

        # Stream
        self._emit({"event": "walk_init", "geom_names": geom_names})

        for frame in frames:
            if not self.running:
                break
            frame["event"] = "walk_frame"
            self._emit(frame)
            time.sleep(1.0 / 30)

        self.running = False
        self._emit({"event": "walk_end"})
