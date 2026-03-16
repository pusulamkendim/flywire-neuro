"""
Feed simulation using flybody model (full proboscis kinematics).

flybody has 5 proboscis joints:
  - rostrum: hinge, opens the mouth area
  - haustellum_abduct: lateral movement
  - haustellum: main extension (the tube going down)
  - labrum_left/right: labellum pads that spread open

Sequence: stand → rostrum opens → haustellum extends → labrum spreads → pump → retract
"""

import os
os.environ.setdefault('MUJOCO_GL', 'disabled')

import sys, time, json, asyncio, threading
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

CACHE_DIR = Path(__file__).resolve().parent / 'walk_cache'


def _generate_feed():
    """Feeding with full proboscis animation using flybody model. 4s."""
    import mujoco, flybody

    pkg = os.path.dirname(flybody.__file__)
    xml_path = os.path.join(pkg, 'fruitfly', 'assets', 'fruitfly.xml')

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Settle
    for _ in range(200):
        mujoco.mj_step(model, data)

    thorax_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'thorax')

    # Proboscis joint qpos addresses
    joints = {}
    for name in ['rostrum', 'haustellum_abduct', 'haustellum', 'labrum_left', 'labrum_right']:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            joints[name] = int(model.jnt_qposadr[jid])

    # Geom map (mesh geoms only)
    geom_names = []
    geom_ids = []
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f'geom_{gid}'
        geom_names.append(gname)
        geom_ids.append(gid)

    # Find free joint (root body) to lock fly in place
    free_qadr = None
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            free_qadr = int(model.jnt_qposadr[jid])
            break

    # Save initial root position after settling
    init_qpos = data.qpos[:7].copy() if free_qadr is not None else None

    # Phase timing (ms)
    IDLE_END = 500          # standing still
    OPEN_END = 1000         # rostrum opens, haustellum starts extending
    EXTEND_END = 1500       # haustellum fully extended, labrum opens
    PUMP_END = 3000         # pumping (feeding)
    RETRACT_END = 3500      # retract everything
    TOTAL_MS = 4000

    timestep = 2e-4  # flybody uses 2e-4 physics timestep
    n_steps = int(TOTAL_MS / 1000.0 / timestep)
    emit_every = 167  # ~30fps at 2e-4 timestep

    frames = []
    fly_start_pos = None

    print(f"[Feed-flybody] Generating {TOTAL_MS}ms ({n_steps} steps)...", flush=True)
    t0 = time.time()

    for step_i in range(n_steps):
        t_ms = step_i * timestep * 1000

        # Smooth interpolation helper
        def smooth(t, start_ms, dur_ms):
            p = np.clip((t - start_ms) / dur_ms, 0, 1)
            return 3*p**2 - 2*p**3  # smoothstep

        if t_ms < IDLE_END:
            phase = 'standing'
        elif t_ms < OPEN_END:
            # Rostrum opens, haustellum begins
            phase = 'opening'
            s = smooth(t_ms, IDLE_END, OPEN_END - IDLE_END)
            data.qpos[joints['rostrum']] = -1.0 * s       # open (negative = down)
            data.qpos[joints['haustellum']] = -0.5 * s     # start extending
        elif t_ms < EXTEND_END:
            # Full extension + labrum opens
            phase = 'extending'
            s = smooth(t_ms, OPEN_END, EXTEND_END - OPEN_END)
            data.qpos[joints['rostrum']] = -1.0
            data.qpos[joints['haustellum']] = -0.5 - 0.8 * s  # full extend
            data.qpos[joints['labrum_left']] = 0.8 * s     # labellum opens
            data.qpos[joints['labrum_right']] = 0.8 * s
        elif t_ms < PUMP_END:
            # Pumping: haustellum oscillates, labrum pulses
            phase = 'feeding'
            t_pump = (t_ms - EXTEND_END) / 1000.0
            pump = np.sin(2 * np.pi * 3.0 * t_pump)  # 3Hz
            data.qpos[joints['rostrum']] = -1.0
            data.qpos[joints['haustellum']] = -1.3 + 0.15 * pump
            data.qpos[joints['labrum_left']] = 0.8 + 0.15 * pump
            data.qpos[joints['labrum_right']] = 0.8 - 0.15 * pump  # alternating
            data.qpos[joints['haustellum_abduct']] = 0.03 * pump  # slight lateral
        elif t_ms < RETRACT_END:
            # Retract everything
            phase = 'retracting'
            s = 1.0 - smooth(t_ms, PUMP_END, RETRACT_END - PUMP_END)
            data.qpos[joints['rostrum']] = -1.0 * s
            data.qpos[joints['haustellum']] = -1.3 * s
            data.qpos[joints['labrum_left']] = 0.8 * s
            data.qpos[joints['labrum_right']] = 0.8 * s
            data.qpos[joints['haustellum_abduct']] = 0
        else:
            phase = 'standing'
            for j in joints.values():
                data.qpos[j] = 0

        mujoco.mj_step(model, data)

        # Lock fly in place (no falling — only proboscis moves)
        if free_qadr is not None and init_qpos is not None:
            data.qpos[free_qadr:free_qadr + 7] = init_qpos
            data.qvel[:6] = 0  # zero root velocity

        if step_i % emit_every == 0 and step_i > 0:
            thorax_pos = data.xpos[thorax_bid].copy()
            if fly_start_pos is None:
                fly_start_pos = thorax_pos.copy()

            poses = []
            for gid in geom_ids:
                xpos = data.geom_xpos[gid] - thorax_pos
                xmat = data.geom_xmat[gid].reshape(3, 3)
                poses.extend(xpos.tolist())
                q = R.from_matrix(xmat).as_quat() if np.all(np.isfinite(xmat)) else [0, 0, 0, 1]
                poses.extend(q.tolist())

            fly_pos = (thorax_pos - fly_start_pos).tolist()
            frames.append({
                "t_ms": round(t_ms, 1),
                "fly_pos": [round(v, 3) for v in fly_pos],
                "poses": [round(v, 5) for v in poses],
                "phase": phase,
            })

    elapsed = time.time() - t0
    print(f"[Feed-flybody] {len(frames)} frames in {elapsed:.1f}s", flush=True)
    return geom_names, frames


CACHE = CACHE_DIR / 'feed_flybody_4.0s.json'


class FeedBridge:
    def __init__(self):
        self.queue = self._loop = self._thread = None
        self.running = False

    def start(self, loop, queue):
        if self.running:
            return
        self.queue, self._loop, self.running = queue, loop, True
        self._thread = threading.Thread(target=self._run_safe, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def _emit(self, data):
        try:
            asyncio.run_coroutine_threadsafe(self.queue.put(data), self._loop).result(timeout=5)
        except Exception as e:
            print(f"[Feed] emit error: {e}", flush=True)

    def _run_safe(self):
        try:
            self._run()
        except Exception as e:
            import traceback
            print(f"[Feed] ERROR: {e}", flush=True)
            traceback.print_exc()
            self.running = False
            self._emit({"event": "walk_end"})

    def _run(self):
        if CACHE.exists():
            with open(CACHE) as f:
                cached = json.load(f)
            geom_names, frames = cached['geom_names'], cached['frames']
            print(f"[Feed] Cache: {len(frames)} frames", flush=True)
        else:
            geom_names, frames = _generate_feed()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(CACHE, 'w') as f:
                json.dump({'geom_names': geom_names, 'frames': frames}, f)
            print(f"[Feed] Cached: {CACHE.stat().st_size // 1024}KB", flush=True)

        self._emit({"event": "walk_init", "geom_names": geom_names})
        for frame in frames:
            if not self.running:
                break
            frame["event"] = "walk_frame"
            self._emit(frame)
            time.sleep(1.0 / 30)
        self.running = False
        self._emit({"event": "walk_end"})
