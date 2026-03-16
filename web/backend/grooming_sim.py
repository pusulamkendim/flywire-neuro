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
    """Front-leg oscillation for antennal grooming. From Eon's fly_embodied.py."""

    def __init__(self, preprogrammed_steps, freq_hz=4.0):
        self.steps = preprogrammed_steps
        self.freq = freq_hz
        self.neutral = np.zeros(42)
        for i, leg in enumerate(self.steps.legs):
            self.neutral[i * 7:(i + 1) * 7] = self.steps.get_joint_angles(
                leg, np.pi, 0.0)

    def get_action(self, time_s):
        joints = self.neutral.copy()
        phase = 2 * np.pi * self.freq * time_s
        femur_offset = 0.3 * np.sin(phase)
        tibia_offset = 0.4 * np.sin(phase + np.pi / 2)
        for base in (0, 21):  # LF and RF
            joints[base + 3] += femur_offset
            joints[base + 5] += tibia_offset
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
            # Grooming phase — front legs oscillate
            action = groom_ctrl.get_action(t_s)
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


GROOM_CACHE = CACHE_DIR / 'groom_3.0s.json'


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
