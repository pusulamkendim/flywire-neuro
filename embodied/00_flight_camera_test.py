"""
Embodied 00: Flight Camera Test — Manuel uçuş (beyin modeli olmadan)

Sinek yerde yürüyor → havalanıyor → uçuyor → iniyor.
Kanat animasyonu (geom_xmat) + bacak tuck + orientation lock.

Gerçek Drosophila uçuş referansı:
  - Kanat frekansı: ~200 Hz, stroke açısı: ±30° (yatay düzlemde)
  - Bacaklar: dışa doğru sarkık, hafif bükülü (kanca şekli)
  - Kanatlar: flygym'de statik mesh — geom_xmat ile animasyon

Teknik notlar:
  - Kanat animasyonu: data.geom_xmat değiştirilerek yapılır (model değil)
    → fizik motoru etkilenmez, sadece render'da görünür
  - Bacak tuck: actuator ctrl ile, stretch açılarından türetilmiş
    → Sağ bacaklar: Coxa_roll, Coxa_yaw, Femur_roll ters işaretli (aynalama)
  - body_quat DEĞİŞTİRME — fizik motorunu bozar ve bacakları açar
  - Orientation lock: free joint qpos quaternion + qvel sıfırlama
"""
import numpy as np
import mujoco
import re
import time
from scipy.spatial.transform import Rotation as R
from flygym import Fly, SingleFlySimulation, Camera
from flygym.preprogrammed import get_cpg_biases
from flygym.examples.locomotion import PreprogrammedSteps, CPGNetwork
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / 'results'

# =====================================================================
# KONFİGÜRASYON
# =====================================================================

WING_FREQ = 200.0          # Hz — gerçek Drosophila
WING_AMPLITUDE = np.radians(30)  # ±30°
SIM_DURATION_S = 1.5
BODY_TIMESTEP = 1e-4

# Uçuş bacak pozisyonları — gerçek sinek fotoğrafından türetilmiş
# Stretch açılarına yakın ama:
#   Femur: biraz yukarı çekilmiş (daha az negatif)
#   Tibia: daha bükülü (kanca şekli)
# Sağ bacaklar: roll ve yaw açıları ters işaretli (simetri)
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
# SETUP
# =====================================================================

print("=" * 60)
print("EMBODIED 00: FLIGHT CAMERA TEST")
print("=" * 60)

fly = Fly(enable_adhesion=True, init_pose='stretch', control='position')
cam = Camera(
    attachment_point=fly.model.worldbody,
    camera_name='camera_back_track',
    targeted_fly_names=[fly.name],
    play_speed=0.2,
    window_size=(1280, 720),
    fps=30,
    timestamp_text=True,
)
sim = SingleFlySimulation(fly=fly, cameras=[cam], timestep=BODY_TIMESTEP)

# CPG (yürüme fazı için)
cpg = CPGNetwork(
    timestep=BODY_TIMESTEP,
    intrinsic_freqs=np.ones(6) * 12.0,
    intrinsic_amps=np.ones(6),
    coupling_weights=(get_cpg_biases('tripod') > 0).astype(float) * 10.0,
    phase_biases=get_cpg_biases('tripod'),
    convergence_coefs=np.ones(6) * 20.0,
)
preprogrammed = PreprogrammedSteps()
leg_names = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']

# MuJoCo pointers
model_ptr = sim.physics.model.ptr
data_ptr = sim.physics.data.ptr
fly_name = fly.name

# Body IDs
thorax_id = mujoco.mj_name2id(model_ptr, mujoco.mjtObj.mjOBJ_BODY, f"{fly_name}/Thorax")
lwing_geom_id = mujoco.mj_name2id(model_ptr, mujoco.mjtObj.mjOBJ_GEOM, f"{fly_name}/LWing")
rwing_geom_id = mujoco.mj_name2id(model_ptr, mujoco.mjtObj.mjOBJ_GEOM, f"{fly_name}/RWing")

# Mass & gravity
fly_mass = sum(
    float(model_ptr.body_mass[bid])
    for bid in range(model_ptr.nbody)
    if (mujoco.mj_id2name(model_ptr, mujoco.mjtObj.mjOBJ_BODY, bid) or "").startswith(f"{fly_name}/")
)
gravity = float(abs(model_ptr.opt.gravity[2]))
mg = fly_mass * gravity

# Free joint addresses
free_qpos_adr = None
free_dof_adr = None
for jid in range(model_ptr.njnt):
    if model_ptr.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
        jbody = model_ptr.jnt_bodyid[jid]
        bname = mujoco.mj_id2name(model_ptr, mujoco.mjtObj.mjOBJ_BODY, jbody) or ""
        if fly_name in bname:
            free_qpos_adr = int(model_ptr.jnt_qposadr[jid])
            free_dof_adr = int(model_ptr.jnt_dofadr[jid])
            break

print(f"  Mass: {fly_mass*1e6:.0f} mg, Thorax: {thorax_id}")
print(f"  LWing geom: {lwing_geom_id}, RWing geom: {rwing_geom_id}")

# =====================================================================
# TUCK AÇILARI HAZIRLA
# =====================================================================

tuck_angles = np.zeros(len(fly.actuators))
for i, act in enumerate(fly.actuators):
    s = str(act)
    match = re.search(r'joint_([A-Z]{2})(\w+)', s)
    if not match:
        continue
    leg = match.group(1)
    joint = match.group(2)
    if leg in FLIGHT_LEG_ANGLES and joint in FLIGHT_LEG_ANGLES[leg]:
        tuck_angles[i] = np.radians(FLIGHT_LEG_ANGLES[leg][joint])

print(f"  Tuck angles: {np.count_nonzero(tuck_angles)}/{len(tuck_angles)} joints assigned")

# =====================================================================
# KANAT ANİMASYONU
# =====================================================================

def animate_wings(t_seconds):
    """
    Kanat çırpma animasyonu — geom_xmat seviyesinde.
    Model'e dokunmaz, sadece render verisini değiştirir.
    Sonraki mj_step geom_xmat'ı yeniden hesaplar → güvenli.
    """
    angle = WING_AMPLITUDE * np.sin(2 * np.pi * WING_FREQ * t_seconds)
    for geom_id, sign in [(lwing_geom_id, 1), (rwing_geom_id, -1)]:
        current_rot = R.from_matrix(data_ptr.geom_xmat[geom_id].reshape(3, 3))
        wing_rot = R.from_euler('y', sign * angle)
        new_rot = current_rot * wing_rot
        data_ptr.geom_xmat[geom_id] = new_rot.as_matrix().flatten()

# =====================================================================
# SİMÜLASYON
# =====================================================================

obs, _ = sim.reset()
n_steps = int(SIM_DURATION_S / BODY_TIMESTEP)

print(f"\n  Simülasyon: {SIM_DURATION_S}s ({n_steps:,} adım)")
print(f"  Fazlar:")
print(f"    0-200ms:    Yerde yürüme")
print(f"    200-800ms:  Uçuş (sabit kuvvet)")
print(f"    800-1500ms: İniş")
print()

t_start = time.time()
is_flying = False

for step_i in range(n_steps):
    t_ms = step_i * BODY_TIMESTEP * 1000
    t_s = step_i * BODY_TIMESTEP

    if t_ms < 200:
        # ── YÜRÜME ──
        is_flying = False
        cpg.step()
        all_angles = []
        all_adhesion = []
        for i, leg in enumerate(leg_names):
            angles = preprogrammed.get_joint_angles(
                leg, cpg.curr_phases[i], cpg.curr_magnitudes[i])
            all_angles.append(angles)
            adhesion = preprogrammed.get_adhesion_onoff(leg, cpg.curr_phases[i])
            all_adhesion.append(adhesion)
        action = {
            'joints': np.concatenate(all_angles),
            'adhesion': np.array(all_adhesion, dtype=np.float64),
        }
        obs, _, _, _, _ = sim.step(action)

    else:
        # ── UÇUŞ ──
        is_flying = True

        # Kuvvet
        force = np.zeros(6)
        alt = float(data_ptr.qpos[free_qpos_adr + 2])

        if t_ms < 400:
            # Takeoff
            force[2] = mg * 1.15
            force[0] = mg * 0.08
        elif t_ms < 800:
            # Hover
            alt_error = 5.0 - alt
            force[2] = mg * (1.0 + 0.03 * alt_error)
            force[0] = mg * 0.05
        else:
            # İniş
            landing_progress = (t_ms - 800) / 700
            force[2] = mg * max(0.3, 0.9 - 0.6 * landing_progress)
            force[0] = mg * 0.02

        data_ptr.xfrc_applied[thorax_id, :] = force

        # Bacaklar tuck pozisyonunda
        obs, _, _, _, _ = sim.step({
            'joints': tuck_angles,
            'adhesion': np.zeros(6),
        })

        # Orientation lock (dönmeyi engelle)
        data_ptr.qpos[free_qpos_adr + 3:free_qpos_adr + 7] = [1, 0, 0, 0]
        data_ptr.qvel[free_dof_adr + 3:free_dof_adr + 6] = 0

        # İniş sırasında hız sönümleme
        if t_ms > 800:
            vz = data_ptr.qvel[free_dof_adr + 2]
            if vz > 0:
                data_ptr.qvel[free_dof_adr + 2] = vz * 0.95
            data_ptr.qvel[free_dof_adr + 0] *= 0.98
            data_ptr.qvel[free_dof_adr + 1] *= 0.98

    # Kanat animasyonu (render-only)
    if is_flying:
        animate_wings(t_s)

    sim.render()

    # Progress log
    if step_i % 2000 == 0:
        pos = obs['fly'][0]
        elapsed = time.time() - t_start
        phase = 'walk' if t_ms < 200 else 'takeoff' if t_ms < 400 else 'hover' if t_ms < 800 else 'land'
        print(f"  t={t_ms:6.0f}ms [{phase:>7s}] "
              f"pos=({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f})mm "
              f"[{elapsed:.0f}s]")

# Video kaydet
video_path = str(RESULTS_DIR / '00_flight_camera_test.mp4')
cam.save_video(video_path)
sim.close()

total_time = time.time() - t_start
print(f"\n  Video: {video_path}")
print(f"  Süre: {total_time:.0f}s")
print(f"\n{'='*60}")
print("TAMAMLANDI")
print(f"{'='*60}")
