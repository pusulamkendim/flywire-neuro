"""
Escape simulation: Walk → Looming detected → Fast escape run → Recovery

No flight — ground-level escape with directional turning (like cockroach escape).
"""

import os
os.environ.setdefault('MUJOCO_GL', 'disabled')

import sys, time, json, asyncio, threading
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


def _generate_escape():
    """Walk → freeze → fast escape with turn → decelerate → walk. 3s total."""
    import mujoco
    from flygym import Fly
    from flygym.simulation import SingleFlySimulation
    from flygym.preprogrammed import get_cpg_biases
    from flygym.examples.locomotion import PreprogrammedSteps, CPGNetwork

    fly = Fly(enable_adhesion=True, init_pose='stretch', control='position')
    sim = SingleFlySimulation(fly=fly, timestep=1e-4)
    sim.reset()
    steps = PreprogrammedSteps()
    cpg = CPGNetwork(
        timestep=1e-4, intrinsic_freqs=np.ones(6) * 12.0, intrinsic_amps=np.ones(6),
        coupling_weights=(get_cpg_biases('tripod') > 0).astype(float) * 10.0,
        phase_biases=get_cpg_biases('tripod'), convergence_coefs=np.ones(6) * 20.0,
    )

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

    WALK_END = 500
    FREEZE_END = 600       # brief startle freeze
    ESCAPE_END = 1500      # fast asymmetric run (turn away)
    DECEL_END = 2200       # slow down
    TOTAL_MS = 3000

    timestep = 1e-4
    n_steps = int(TOTAL_MS / 1000.0 / timestep)
    emit_every = 333
    frames = []
    fly_start_pos = None

    print(f"[Escape] Generating {TOTAL_MS}ms...", flush=True)
    t0 = time.time()

    for step_i in range(n_steps):
        t_ms = step_i * timestep * 1000

        if t_ms < WALK_END:
            cpg.intrinsic_freqs[:] = 12.0
            phase = 'walking'
        elif t_ms < FREEZE_END:
            cpg.intrinsic_freqs[:] = 0.5  # startle freeze
            phase = 'freeze'
        elif t_ms < ESCAPE_END:
            # Fast escape with right turn (escape LEFT from threat on right)
            progress = (t_ms - FREEZE_END) / (ESCAPE_END - FREEZE_END)
            speed = 20.0 - 5.0 * progress  # start fast, slow down
            turn_bias = 3.0 * max(0, 1.0 - progress * 1.5)  # turn fades
            cpg.intrinsic_freqs[:3] = speed + turn_bias   # left legs faster
            cpg.intrinsic_freqs[3:] = speed - turn_bias   # right legs slower
            phase = 'escape'
        elif t_ms < DECEL_END:
            progress = (t_ms - ESCAPE_END) / (DECEL_END - ESCAPE_END)
            speed = 15.0 - 10.0 * progress
            cpg.intrinsic_freqs[:] = max(5.0, speed)
            phase = 'decelerate'
        else:
            cpg.intrinsic_freqs[:] = 10.0
            phase = 'walking'

        cpg.step()
        joints = [steps.get_joint_angles(leg, cpg.curr_phases[i], cpg.curr_magnitudes[i])
                  for i, leg in enumerate(LEG_NAMES)]
        adhesion = [steps.get_adhesion_onoff(leg, cpg.curr_phases[i])
                    for i, leg in enumerate(LEG_NAMES)]
        sim.step({'joints': np.concatenate(joints),
                  'adhesion': np.array(adhesion, dtype=np.float64)})

        if step_i % emit_every == 0 and step_i > 0:
            thorax_pos = data_ptr.xpos[thorax_bid].copy()
            if fly_start_pos is None: fly_start_pos = thorax_pos.copy()
            poses = []
            for gid in geom_ids:
                xpos = data_ptr.geom_xpos[gid] - thorax_pos
                xmat = data_ptr.geom_xmat[gid].reshape(3, 3)
                poses.extend(xpos.tolist())
                q = R.from_matrix(xmat).as_quat() if np.all(np.isfinite(xmat)) else [0,0,0,1]
                poses.extend(q.tolist())
            fly_pos = (thorax_pos - fly_start_pos).tolist()
            frames.append({"t_ms": round(t_ms, 1), "fly_pos": [round(v,3) for v in fly_pos],
                           "poses": [round(v,5) for v in poses], "phase": phase})

    sim.close()
    print(f"[Escape] {len(frames)} frames in {time.time()-t0:.1f}s", flush=True)
    return geom_names, frames


CACHE = CACHE_DIR / 'escape_3.0s.json'

class EscapeBridge:
    def __init__(self):
        self.queue = self._loop = self._thread = None; self.running = False
    def start(self, loop, queue):
        if self.running: return
        self.queue, self._loop, self.running = queue, loop, True
        self._thread = threading.Thread(target=self._run_safe, daemon=True); self._thread.start()
    def stop(self): self.running = False
    def _emit(self, data):
        try: asyncio.run_coroutine_threadsafe(self.queue.put(data), self._loop).result(timeout=5)
        except Exception as e: print(f"[Escape] emit error: {e}", flush=True)
    def _run_safe(self):
        try: self._run()
        except Exception as e:
            import traceback; print(f"[Escape] ERROR: {e}", flush=True); traceback.print_exc()
            self.running = False; self._emit({"event": "walk_end"})
    def _run(self):
        if CACHE.exists():
            with open(CACHE) as f: cached = json.load(f)
            geom_names, frames = cached['geom_names'], cached['frames']
        else:
            geom_names, frames = _generate_escape()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(CACHE, 'w') as f: json.dump({'geom_names': geom_names, 'frames': frames}, f)
        self._emit({"event": "walk_init", "geom_names": geom_names})
        for frame in frames:
            if not self.running: break
            frame["event"] = "walk_frame"; self._emit(frame); time.sleep(1.0/30)
        self.running = False; self._emit({"event": "walk_end"})
