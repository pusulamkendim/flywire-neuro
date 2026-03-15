"""
Walking simulation: runs NeuroMechFly CPG tripod gait in MuJoCo,
streams per-geom transforms for 3D animation in the browser.

First run: computes and caches all frames to disk.
Subsequent runs: replays from cache (instant, no MuJoCo needed).
"""

import os
os.environ['MUJOCO_GL'] = 'disabled'

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


def _cache_path(duration_s):
    """Cache file path for a given duration."""
    return CACHE_DIR / f'walk_{duration_s:.1f}s.json'


def _generate_walk(duration_s):
    """Run MuJoCo CPG walking, return list of frame dicts."""
    import mujoco
    from flygym import Fly
    from flygym.simulation import SingleFlySimulation
    from flygym.preprogrammed import get_cpg_biases
    from flygym.examples.locomotion import PreprogrammedSteps, CPGNetwork
    from scipy.spatial.transform import Rotation as R

    fly = Fly(enable_adhesion=True, init_pose='stretch', control='position')
    sim = SingleFlySimulation(fly=fly, timestep=1e-4)
    sim.reset()

    cpg = CPGNetwork(
        timestep=1e-4,
        intrinsic_freqs=np.ones(6) * 12.0,
        intrinsic_amps=np.ones(6),
        coupling_weights=(get_cpg_biases('tripod') > 0).astype(float) * 10.0,
        phase_biases=get_cpg_biases('tripod'),
        convergence_coefs=np.ones(6) * 20.0,
    )
    steps = PreprogrammedSteps()

    model_ptr = sim.physics.model.ptr
    data_ptr = sim.physics.data.ptr
    fly_name = fly.name

    # Map geom names → IDs
    geom_map = {}
    for gid in range(model_ptr.ngeom):
        gname = mujoco.mj_id2name(model_ptr, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
        if gname.startswith(f'{fly_name}/') and model_ptr.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            geom_map[gname.replace(f'{fly_name}/', '')] = gid

    thorax_bid = mujoco.mj_name2id(model_ptr, mujoco.mjtObj.mjOBJ_BODY, f'{fly_name}/Thorax')
    geom_names = list(geom_map.keys())
    geom_ids = list(geom_map.values())

    timestep = 1e-4
    n_steps = int(duration_s / timestep)
    emit_every = 333  # ~30fps

    frames = []
    fly_start_pos = None

    print(f"[Walk] Generating {duration_s}s walk ({n_steps} steps)...", flush=True)
    t0 = time.time()

    for step_i in range(n_steps):
        cpg.step()
        joints = [steps.get_joint_angles(leg, cpg.curr_phases[i], cpg.curr_magnitudes[i])
                  for i, leg in enumerate(LEG_NAMES)]
        adhesion = [steps.get_adhesion_onoff(leg, cpg.curr_phases[i])
                    for i, leg in enumerate(LEG_NAMES)]

        sim.step({
            'joints': np.concatenate(joints),
            'adhesion': np.array(adhesion, dtype=np.float64),
        })

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
                "t_ms": round(step_i * timestep * 1000, 1),
                "fly_pos": [round(v, 3) for v in fly_pos],
                "poses": [round(v, 5) for v in poses],
            })

    sim.close()
    elapsed = time.time() - t0
    print(f"[Walk] Generated {len(frames)} frames in {elapsed:.1f}s", flush=True)

    return geom_names, frames


class WalkingBridge:
    """Streams walk frames from cache (or generates + caches first)."""

    def __init__(self):
        self.queue = None
        self.running = False
        self._thread = None
        self._loop = None

    def start(self, loop, queue, duration_s=5.0):
        if self.running:
            return
        self.queue = queue
        self._loop = loop
        self.running = True
        self.duration_s = duration_s
        self._thread = threading.Thread(target=self._run_safe, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def _emit(self, data):
        try:
            future = asyncio.run_coroutine_threadsafe(self.queue.put(data), self._loop)
            future.result(timeout=5)
        except Exception as e:
            print(f"[Walk] emit error: {e}", flush=True)

    def _run_safe(self):
        import faulthandler
        faulthandler.enable()
        try:
            self._run()
        except Exception as e:
            import traceback
            print(f"[Walk] ERROR: {e}", flush=True)
            traceback.print_exc()
            self.running = False
            self._emit({"event": "walk_end"})

    def _run(self):
        cache_file = _cache_path(self.duration_s)

        if cache_file.exists():
            # Replay from cache
            print(f"[Walk] Loading cache: {cache_file}", flush=True)
            with open(cache_file) as f:
                cached = json.load(f)
            geom_names = cached['geom_names']
            frames = cached['frames']
            print(f"[Walk] Cache loaded: {len(frames)} frames", flush=True)
        else:
            # Generate + save
            geom_names, frames = _generate_walk(self.duration_s)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump({'geom_names': geom_names, 'frames': frames}, f)
            size_kb = cache_file.stat().st_size // 1024
            print(f"[Walk] Cached: {cache_file} ({size_kb}KB)", flush=True)

        # Stream frames
        self._emit({"event": "walk_init", "geom_names": geom_names})

        fps = 30
        for frame in frames:
            if not self.running:
                break
            frame["event"] = "walk_frame"
            self._emit(frame)
            time.sleep(1.0 / fps)  # real-time playback from cache

        self.running = False
        self._emit({"event": "walk_end"})
