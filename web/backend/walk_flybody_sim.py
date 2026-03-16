"""
Walking simulation: flybody model + real motion capture data.

From dataset: XY position + Yaw (real trajectory) + joint angles (real legs)
Fixed by us: Z = 0.1435 (constant ground level), Roll = 0, Pitch = -8.8° (natural tilt)

This gives: real leg movement + real walking path + stable on flat ground.
"""

import os
os.environ.setdefault('MUJOCO_GL', 'disabled')

import sys, time, json, asyncio, threading
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

CACHE_DIR = Path(__file__).resolve().parent / 'walk_cache'
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
WALK_DATA = PROJECT_DIR / 'data' / 'simulation' / 'datasets_walking-imitation' / \
    'walking-dataset_female-only_snippets-16252_trk-files-0-9.hdf5'

CHAIN_TRAJECTORIES = [970, 396, 671, 857]

# Constants from data analysis
FIXED_Z = 0.1435       # mean Z across all trajectories
FIXED_PITCH = -8.8     # degrees, natural body tilt


def _generate_walk_flybody():
    import mujoco, flybody, h5py

    pkg = os.path.dirname(flybody.__file__)
    xml_path = os.path.join(pkg, 'fruitfly', 'assets', 'fruitfly.xml')
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Load trajectories: root_qpos (for XY + yaw) + qpos (joint angles)
    with h5py.File(str(WALK_DATA), 'r') as f:
        timestep_data = f['timestep_seconds'][()]
        all_root = []
        all_joints = []
        xy_offset = np.zeros(2)

        for traj_idx in CHAIN_TRAJECTORIES:
            key = str(traj_idx).zfill(5)
            root = f['trajectories'][key]['root_qpos'][:].copy()
            joints = f['trajectories'][key]['qpos'][:].copy()

            # Center XY at current offset (chain trajectories)
            root[:, :2] -= root[0, :2]
            root[:, :2] += xy_offset
            xy_offset = root[-1, :2].copy()

            all_root.append(root)
            all_joints.append(joints)

    root_qpos = np.concatenate(all_root, axis=0)
    joint_qpos = np.concatenate(all_joints, axis=0)
    n_frames = len(root_qpos)

    print(f"[Walk-FB] {n_frames} frames ({n_frames * timestep_data:.2f}s)", flush=True)

    thorax_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'thorax')

    # Free joint
    free_qadr = None
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            free_qadr = int(model.jnt_qposadr[jid])
            break

    # Non-free joint addresses
    mocap_qadrs = []
    for jid in range(model.njnt):
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            mocap_qadrs.append(int(model.jnt_qposadr[jid]))

    # Geom map
    geom_names, geom_ids = [], []
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f'geom_{gid}'
        geom_names.append(gname)
        geom_ids.append(gid)

    steps_per_frame = max(1, int(timestep_data / model.opt.timestep))
    emit_every = max(1, int(1.0 / 30.0 / timestep_data))

    frames = []
    fly_start_pos = None

    print(f"[Walk-FB] steps/frame={steps_per_frame}, emit_every={emit_every}", flush=True)
    t0 = time.time()

    for fi in range(n_frames):
        # --- Root: use XY + Yaw from data, fix Z + Roll + Pitch ---
        if free_qadr is not None:
            # XY from dataset (real walking path)
            data.qpos[free_qadr + 0] = root_qpos[fi, 0]  # X
            data.qpos[free_qadr + 1] = root_qpos[fi, 1]  # Y
            # Z fixed (flat ground)
            data.qpos[free_qadr + 2] = FIXED_Z

            # Extract yaw from dataset quaternion (MuJoCo: w,x,y,z)
            qw, qx, qy, qz = root_qpos[fi, 3:7]
            # scipy wants x,y,z,w
            rot = R.from_quat([qx, qy, qz, qw])
            roll_data, pitch_data, yaw_data = rot.as_euler('xyz', degrees=True)

            # Build clean quaternion: Roll=0, Pitch=fixed, Yaw=from data
            clean_rot = R.from_euler('xyz', [0, FIXED_PITCH, yaw_data], degrees=True)
            clean_quat = clean_rot.as_quat()  # x,y,z,w
            # MuJoCo wants w,x,y,z
            data.qpos[free_qadr + 3] = clean_quat[3]  # w
            data.qpos[free_qadr + 4] = clean_quat[0]  # x
            data.qpos[free_qadr + 5] = clean_quat[1]  # y
            data.qpos[free_qadr + 6] = clean_quat[2]  # z

        # --- Joints: all from dataset ---
        for ji, qadr in enumerate(mocap_qadrs):
            if ji < joint_qpos.shape[1]:
                data.qpos[qadr] = joint_qpos[fi, ji]

        # Forward kinematics (compute geom positions from qpos)
        mujoco.mj_forward(model, data)

        # Emit
        if fi % emit_every == 0:
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
                "t_ms": round(fi * timestep_data * 1000, 1),
                "fly_pos": [round(v, 4) for v in fly_pos],
                "poses": [round(v, 5) for v in poses],
                "phase": "walking",
            })

    elapsed = time.time() - t0
    print(f"[Walk-FB] {len(frames)} frames in {elapsed:.1f}s", flush=True)

    if frames:
        fp0, fp1 = frames[0]['fly_pos'], frames[-1]['fly_pos']
        disp = ((fp1[0]-fp0[0])**2 + (fp1[1]-fp0[1])**2)**0.5
        zs = [f['fly_pos'][2] for f in frames]
        print(f"[Walk-FB] XY disp: {disp:.3f}, Z: [{min(zs):.4f}, {max(zs):.4f}]", flush=True)

    return geom_names, frames


CACHE = CACHE_DIR / 'walk_flybody_real.json'


class WalkFlybodyBridge:
    def __init__(self):
        self.queue = self._loop = self._thread = None
        self.running = False

    def start(self, loop, queue):
        if self.running: return
        self.queue, self._loop, self.running = queue, loop, True
        self._thread = threading.Thread(target=self._run_safe, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def _emit(self, data):
        try:
            asyncio.run_coroutine_threadsafe(self.queue.put(data), self._loop).result(timeout=5)
        except Exception as e:
            print(f"[Walk-FB] emit error: {e}", flush=True)

    def _run_safe(self):
        try:
            self._run()
        except Exception as e:
            import traceback
            print(f"[Walk-FB] ERROR: {e}", flush=True)
            traceback.print_exc()
            self.running = False
            self._emit({"event": "walk_end"})

    def _run(self):
        if CACHE.exists():
            with open(CACHE) as f:
                cached = json.load(f)
            geom_names, frames = cached['geom_names'], cached['frames']
            print(f"[Walk-FB] Cache: {len(frames)} frames", flush=True)
        else:
            geom_names, frames = _generate_walk_flybody()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(CACHE, 'w') as f:
                json.dump({'geom_names': geom_names, 'frames': frames}, f)
            print(f"[Walk-FB] Cached: {CACHE.stat().st_size // 1024}KB", flush=True)

        self._emit({"event": "walk_init", "geom_names": geom_names})
        for frame in frames:
            if not self.running: break
            frame["event"] = "walk_frame"
            self._emit(frame)
            time.sleep(1.0 / 30)
        self.running = False
        self._emit({"event": "walk_end"})
