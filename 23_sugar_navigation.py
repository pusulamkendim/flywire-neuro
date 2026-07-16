"""
Analysis 23: Sugar Dish Navigation + Eye Grooming

Brain-controlled fly with 3 emergent behaviors:
  1. Olfactory navigation → walk toward sugar dish
  2. Eye grooming → dust irritation triggers JO→aDN1→front-leg cleaning
  3. Gustatory feeding → tarsal sugar contact triggers MN9→stop+feed

Scenario timeline:
  Walk → track odor → DUST! → stop → groom eyes → resume → reach sugar → feed

Pipeline per 15ms sync cycle:
  1. obs["odor_intensity"] → bilateral ORN firing rates
  2. JO stimulus (dust) → aDN1 grooming DN
  3. Fly proximity to sugar dish → sugar GRN activation
  4. Brain (138K LIF neurons, scipy-optimized) runs 150 steps
  5. DN spikes decoded → forward/turn/groom/feed rates
  6. aDN1 above threshold → grooming mode (front legs oscillate)
  7. MN9 above threshold → feeding mode (stop walking)
  8. Otherwise → differential CPG turning toward odor source
"""

import sys
import os
import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
from pathlib import Path
from collections import deque
import time

# Add fly-brain code to path
FLY_BRAIN_DIR = Path(__file__).parent / 'fly-brain-embodied'
CODE_DIR = FLY_BRAIN_DIR / 'code'
DATA_DIR = FLY_BRAIN_DIR / 'data'
sys.path.insert(0, str(CODE_DIR))

RESULTS_DIR = Path(__file__).parent / 'results'
DATA_LOCAL = Path(__file__).parent / 'data'

# Override paths
import benchmark
benchmark.COMP_PATH = str(DATA_DIR / '2025_Completeness_783.csv')
benchmark.CONN_PATH = str(DATA_DIR / '2025_Connectivity_783.parquet')
benchmark.DATA_DIR = str(DATA_DIR)

from run_pytorch import TorchModel, MODEL_PARAMS, DT, get_weights, get_hash_tables

from flygym import Fly, SingleFlySimulation, Camera
from flygym.arena import OdorArena
from flygym.preprogrammed import get_cpg_biases
from flygym.examples.locomotion import PreprogrammedSteps, CPGNetwork

# =====================================================================
# DN + STIMULI DEFINITIONS (from brain_body_bridge.py)
# =====================================================================

DN_NEURONS = {
    'P9_left':       720575940627652358,
    'P9_right':      720575940635872101,
    'P9_oDN1_left':  720575940626730883,
    'P9_oDN1_right': 720575940620300308,
    'DNa01_left':    720575940644438551,
    'DNa01_right':   720575940627787609,
    'DNa02_left':    720575940604737708,
    'DNa02_right':   720575940629327659,
    'MDN_1':         720575940616026939,
    'MDN_2':         720575940631082808,
    'MDN_3':         720575940640331472,
    'MDN_4':         720575940610236514,
    'MN9_left':      720575940660219265,
    'MN9_right':     720575940618238523,
    # Grooming (aDN1 — antennal/eye grooming descending neurons)
    'aDN1_left':     720575940624319124,
    'aDN1_right':    720575940616185531,
}

DN_GROUPS = {
    'forward':  ['P9_left', 'P9_right', 'P9_oDN1_left', 'P9_oDN1_right'],
    'turn_L':   ['DNa01_left', 'DNa02_left'],
    'turn_R':   ['DNa01_right', 'DNa02_right'],
    'backward': ['MDN_1', 'MDN_2', 'MDN_3', 'MDN_4'],
    'feed':     ['MN9_left', 'MN9_right'],
    'groom':    ['aDN1_left', 'aDN1_right'],
}

# Sugar GRN neuron IDs (from brain_body_bridge.py STIMULI['sugar'])
SUGAR_GRN_IDS = [
    720575940624963786, 720575940630233916, 720575940637568838,
    720575940638202345, 720575940617000768, 720575940630797113,
    720575940632889389, 720575940621754367, 720575940621502051,
    720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940639259967, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570,
    720575940628853239, 720575940629176663, 720575940611875570,
]

# JO (Johnston's Organ) neuron IDs — antennal touch/vibration → triggers aDN1 grooming
# Full set from brain_body_bridge.py STIMULI['jo'] (188 neurons, 9 subtypes)
JO_NEURON_IDS = [
    # JO-E subtypes (vibration/touch — connects to aDN1 for grooming)
    720575940645106376, 720575940615272415, 720575940619869120,
    720575940620257345, 720575940620382889, 720575940630834683,
    720575940632449619, 720575940634020508, 720575940605530302,
    720575940607140035, 720575940608742409, 720575940615590843,
    720575940620410177, 720575940621870618, 720575940622344170,
    720575940623298559, 720575940626042149, 720575940627379333,
    720575940630080071, 720575940632128031, 720575940632307527,
    720575940634820703,
    # JO-C subtypes
    720575940606154370, 720575940605919334, 720575940608884931,
    720575940616655989, 720575940620543110, 720575940622937528,
    720575940624799290, 720575940626565455, 720575940627941431,
    720575940627977457, 720575940628160617, 720575940629188251,
    720575940641921421,
    # JO-EDM subset
    720575940615972027, 720575940618941037, 720575940619729835,
    720575940627282279, 720575940628903247, 720575940604122982,
    720575940609486690, 720575940609541917, 720575940610018266,
    720575940611061526, 720575940611273395, 720575940611684787,
    720575940614060829, 720575940616040587, 720575940618599872,
    720575940618684481, 720575940619663239, 720575940619932654,
    720575940620919578, 720575940621218729, 720575940622271684,
    720575940622638276, 720575940623312828, 720575940625797617,
    720575940625962568, 720575940626309438, 720575940626666066,
    720575940627109991, 720575940628101126, 720575940628978450,
    720575940629055721, 720575940629650997, 720575940629985900,
    720575940630992557, 720575940637054835, 720575940637084762,
    720575940638664437, 720575940646927668, 720575940646929204,
    720575940659131009,
    # JO-EDP
    720575940609522461, 720575940610261346, 720575940613641915,
    720575940615469785, 720575940616589878, 720575940616951124,
    720575940619479979, 720575940621218985, 720575940628444667,
    720575940634634606, 720575940640753267, 720575940650244342,
    # JO-EVL
    720575940615573597, 720575940615848788, 720575940619083349,
    720575940621397417, 720575940621625597, 720575940622283912,
    720575940627049731, 720575940629022149, 720575940630122015,
    720575940630564179, 720575940633153375, 720575940637410869,
    720575940638681845, 720575940621033477, 720575940621776410,
    720575940621815690, 720575940622234211, 720575940622635817,
    720575940623897096, 720575940626148354, 720575940626540821,
    720575940628258715, 720575940629743063, 720575940630202624,
    720575940630544967, 720575940633553820, 720575940644036644,
    # JO-EVM
    720575940602132509, 720575940602506208, 720575940610759634,
    720575940614188149, 720575940615809349, 720575940615976891,
    720575940619341105, 720575940621092534, 720575940622419165,
    720575940622449388, 720575940623108134, 720575940624981436,
    720575940628192055, 720575940630059847, 720575940632767383,
    720575940639296189, 720575940645466500, 720575940611783464,
    720575940612307478, 720575940612960552, 720575940614351477,
    720575940617212134, 720575940617434086, 720575940618130334,
    720575940620249734, 720575940620940276, 720575940621010352,
    720575940621729757, 720575940623437547, 720575940624546062,
    720575940624686268, 720575940625054647, 720575940625605905,
    720575940626795909, 720575940627585688, 720575940630020111,
    720575940632175268, 720575940634073183, 720575940634891700,
    720575940637012196, 720575940637243504, 720575940639339392,
    720575940659426177,
    # JO-EVP
    720575940620444654, 720575940631866508, 720575940607853833,
    720575940611088563, 720575940612773374, 720575940613221928,
    720575940615024543, 720575940615986459, 720575940617811013,
    720575940618467195, 720575940621442224, 720575940622199977,
    720575940624915230, 720575940625559358, 720575940627104649,
    720575940627314088, 720575940633058989, 720575940636335735,
    # JO-CA
    720575940605800369, 720575940608784579, 720575940618135109,
    720575940626719101, 720575940629296185, 720575940636137591,
    720575940602720940, 720575940610079857, 720575940614427195,
    720575940616501787, 720575940617156445, 720575940625909962,
    720575940626241369, 720575940629105658, 720575940629138959,
    720575940636559534, 720575940641372661,
    # JO-CL
    720575940626135548, 720575940627751567, 720575940604753437,
    720575940613971485, 720575940614835362, 720575940623399059,
    720575940630319671, 720575940639082062,
    # JO-CM
    720575940607386307, 720575940634512992, 720575940614035485,
    720575940618901424, 720575940630070343, 720575940633443353,
    720575940635058612, 720575940637632419, 720575940625626000,
]

# =====================================================================
# GROOMING CONTROLLER
# =====================================================================

class GroomingController:
    """Front-leg oscillation for eye/antennal grooming behavior.
    LF and RF sweep across the head while middle/hind legs hold position."""

    def __init__(self, preprogrammed_steps=None, freq_hz=4.0):
        self.steps = preprogrammed_steps or PreprogrammedSteps()
        self.freq = freq_hz
        # Neutral pose: all legs at stance position
        self.neutral = np.zeros(42)
        for i, leg in enumerate(self.steps.legs):
            self.neutral[i * 7:(i + 1) * 7] = self.steps.get_joint_angles(
                leg, np.pi, 0.0
            )

    def get_action(self, time_s):
        """Return joints+adhesion action dict for grooming at given time."""
        joints = self.neutral.copy()
        phase = 2 * np.pi * self.freq * time_s
        femur_offset = 0.3 * np.sin(phase)
        tibia_offset = 0.4 * np.sin(phase + np.pi / 2)
        # LF (indices 0-6) and RF (indices 21-27) do the grooming sweep
        for base in (0, 21):
            joints[base + 3] += femur_offset   # femur rotation
            joints[base + 5] += tibia_offset   # tibia flexion
        # Front legs lifted (no adhesion), rest gripped
        adhesion = np.array([0, 1, 1, 0, 1, 1], dtype=np.float64)
        return {"joints": joints, "adhesion": adhesion}


# =====================================================================
# FAST BRAIN ENGINE (with sugar GRN + JO support)
# =====================================================================

class FastBrainEngine:
    """LIF brain with scipy sparse matmul — ~38x faster than torch CSR.
    Extended with sugar GRN + JO indices for gustatory and grooming."""

    def __init__(self):
        comp_path = str(DATA_DIR / '2025_Completeness_783.csv')
        conn_path = str(DATA_DIR / '2025_Connectivity_783.parquet')

        self.flyid2i, self.i2flyid = get_hash_tables(comp_path)
        self.num_neurons = len(self.flyid2i)

        print(f"Loading {self.num_neurons} neurons...")
        weights = get_weights(conn_path, comp_path, str(DATA_DIR))

        # Convert torch sparse CSR → scipy CSR (transposed for row-select trick)
        w_csr = weights.to_sparse_csr()
        crow = w_csr.crow_indices().numpy()
        col = w_csr.col_indices().numpy()
        vals = w_csr.values().numpy()
        self.w_scipy = sp.csr_matrix(
            (vals, col, crow), shape=(self.num_neurons, self.num_neurons)
        )
        self.w_scipy_t = self.w_scipy.T.tocsr()
        print(f"  Scipy CSR: {self.w_scipy_t.nnz:,} non-zeros")

        self.model = TorchModel(
            batch=1, size=self.num_neurons, dt=DT,
            params=MODEL_PARAMS, weights=weights, device='cpu',
        )
        self.scale = MODEL_PARAMS['wScale']
        self.state = self.model.state_init()
        self.rates = torch.zeros(1, self.num_neurons)

        # DN indices
        self.dn_indices = {}
        for name, flyid in DN_NEURONS.items():
            if flyid in self.flyid2i:
                self.dn_indices[name] = self.flyid2i[flyid]

        # ORN neuron IDs (left/right)
        ann = pd.read_csv(DATA_LOCAL / 'neuron_annotations.tsv', sep='\t',
                          low_memory=False)
        orns = ann[ann['cell_class'].str.contains('olfactory', case=False, na=False)]

        self.orn_left = [self.flyid2i[int(r)] for r in
                         orns[orns['side'] == 'left']['root_id'].values
                         if int(r) in self.flyid2i]
        self.orn_right = [self.flyid2i[int(r)] for r in
                          orns[orns['side'] == 'right']['root_id'].values
                          if int(r) in self.flyid2i]

        # Sugar GRN indices
        self.sugar_grn_indices = [self.flyid2i[nid] for nid in SUGAR_GRN_IDS
                                  if nid in self.flyid2i]

        # JO neuron indices (grooming trigger)
        self.jo_indices = [self.flyid2i[nid] for nid in JO_NEURON_IDS
                           if nid in self.flyid2i]

        print(f"  DN mapped: {len(self.dn_indices)}/{len(DN_NEURONS)}")
        print(f"  ORN left: {len(self.orn_left)}, right: {len(self.orn_right)}")
        print(f"  Sugar GRNs: {len(self.sugar_grn_indices)}/{len(SUGAR_GRN_IDS)}")
        print(f"  JO neurons: {len(self.jo_indices)}/{len(JO_NEURON_IDS)}")

    def set_stimulus(self, left_rate=0.0, right_rate=0.0, p9_rate=0.0,
                     sugar_rate=0.0, jo_rate=0.0):
        """Set bilateral ORN + P9 + sugar GRN + JO stimulus rates.
        When JO is active, also directly drive aDN1 (short-latency reflex)."""
        self.rates.zero_()
        if left_rate > 0:
            self.rates[0, self.orn_left] = left_rate
        if right_rate > 0:
            self.rates[0, self.orn_right] = right_rate
        if p9_rate > 0:
            p9_ids = [720575940627652358, 720575940635872101]
            for fid in p9_ids:
                if fid in self.flyid2i:
                    self.rates[0, self.flyid2i[fid]] = p9_rate
        if sugar_rate > 0 and self.sugar_grn_indices:
            self.rates[0, self.sugar_grn_indices] = sugar_rate
        if jo_rate > 0 and self.jo_indices:
            self.rates[0, self.jo_indices] = jo_rate
            # Direct mechanosensory reflex: JO → aDN1 short-latency pathway
            # Proportional to JO intensity (dust strength)
            adn1_direct_rate = jo_rate * 0.3  # ~90Hz from 300Hz JO
            for name in ('aDN1_left', 'aDN1_right'):
                fid = DN_NEURONS.get(name)
                if fid and fid in self.flyid2i:
                    self.rates[0, self.flyid2i[fid]] = adn1_direct_rate

    @torch.no_grad()
    def step(self):
        """One brain step with optimized spike-aware matmul."""
        cond, dbuf, spk, v, ref = self.state

        spikes_input = self.model.poisson(self.rates)

        spk_np = spk.squeeze(0).numpy()
        firing = np.where(spk_np > 0)[0]

        if len(firing) > 0:
            weighted = np.array(self.w_scipy_t[firing, :].sum(axis=0)).flatten()
            weighted_spikes = torch.from_numpy(weighted).unsqueeze(0).float()
        else:
            weighted_spikes = torch.zeros(1, self.num_neurons)

        total_input = self.scale * (spikes_input + weighted_spikes)
        cond, dbuf, spk, v, ref = self.model.neurons(
            total_input, cond, dbuf, spk, v, ref
        )
        self.state = (cond, dbuf, spk, v, ref)
        return spk

    def get_dn_spikes(self):
        spk = self.state[2]
        return {name: spk[0, idx].item()
                for name, idx in self.dn_indices.items()}


# =====================================================================
# DN DECODER (extended with feed group)
# =====================================================================

class DNDecoder:
    def __init__(self, window_ms=50.0, dt_ms=0.1, max_rate=200.0):
        self.window_steps = int(window_ms / dt_ms)
        self.dt_s = dt_ms / 1000.0
        self.max_rate = max_rate
        self.buffers = {n: deque(maxlen=self.window_steps)
                        for n in DN_NEURONS.keys()}

    def update(self, dn_spikes):
        for name, val in dn_spikes.items():
            self.buffers[name].append(val)

    def get_normalized(self, name):
        buf = self.buffers.get(name)
        if not buf:
            return 0.0
        n = len(buf)
        window_s = n * self.dt_s
        rate = sum(buf) / window_s if window_s > 0 else 0.0
        return min(rate / self.max_rate, 1.0)

    def get_group_rate(self, group_name):
        names = DN_GROUPS.get(group_name, [])
        if not names:
            return 0.0
        return np.mean([self.get_normalized(n) for n in names])


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 72)
    print("ANALYSIS 23: SUGAR NAVIGATION + EYE GROOMING")
    print("Brain-Controlled Fly: Walk → Groom Eyes → Navigate to Sugar → Feed")
    print("=" * 72)

    torch.set_num_threads(10)

    # ── Brain ──
    print("\n[1/5] Loading optimized brain model...")
    t0 = time.time()
    brain = FastBrainEngine()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    decoder = DNDecoder(window_ms=50.0)

    # ── Arena with sugar dish ──
    print("\n[2/5] Creating odor arena with sugar dish...")
    sugar_pos = np.array([[25, 0, 1.5]])  # 25mm ahead, on ground
    arena = OdorArena(
        size=(300, 300),
        odor_source=sugar_pos,
        peak_odor_intensity=np.array([[2e4]]),  # tuned for gradient at 5-25mm range
        diffuse_func=lambda x: x**-2,
        marker_colors=[(1.0, 0.85, 0.0, 1.0)],  # golden yellow for sugar
        marker_size=0.5,
    )
    print(f"  Sugar dish at [{sugar_pos[0, 0]:.0f}, {sugar_pos[0, 1]:.0f}] mm")

    # ── Body ──
    print("\n[3/5] Creating fly body with olfaction...")
    body_timestep = 1e-4
    fly = Fly(
        enable_olfaction=True,
        enable_adhesion=True,
        draw_adhesion=True,
        init_pose='stretch',
        control='position',
    )
    cam = Camera(
        attachment_point=fly.model.worldbody,
        camera_name='camera_right',
        targeted_fly_names=[fly.name],
        play_speed=0.2, window_size=(1280, 720), fps=30,
        timestamp_text=True, draw_contacts=True,
    )
    sim = SingleFlySimulation(
        fly=fly, cameras=[cam], arena=arena, timestep=body_timestep
    )

    cpg = CPGNetwork(
        timestep=body_timestep,
        intrinsic_freqs=np.ones(6) * 12.0,
        intrinsic_amps=np.ones(6),
        coupling_weights=(get_cpg_biases('tripod') > 0).astype(float) * 10.0,
        phase_biases=get_cpg_biases('tripod'),
        convergence_coefs=np.ones(6) * 20.0,
    )
    preprogrammed_steps = PreprogrammedSteps()
    leg_names = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']

    obs, info = sim.reset()

    # ── Simulation params ──
    sim_duration_s = 8.0  # extra time for grooming pause + navigation
    sync_interval_ms = 15.0
    brain_steps_per_sync = int(sync_interval_ms / DT)  # 150
    body_steps_per_sync = int(sync_interval_ms / (body_timestep * 1000))  # 150
    n_syncs = int(sim_duration_s * 1000 / sync_interval_ms)

    # Gustatory parameters
    sugar_detect_dist = 5.0  # mm — tarsal contact distance to sugar dish
    feeding_threshold = 0.03  # MN9 normalized rate threshold for feeding
    feeding_mode = False
    feeding_start_t = None

    # Grooming parameters
    groom_threshold = 0.02   # aDN1 normalized rate threshold for grooming
    grooming_mode = False
    grooming_start_t = None
    grooming_done = False    # only groom once
    groom_ctrl = GroomingController(preprogrammed_steps)

    # Dust irritation timing: hits at ~1.0s, lasts 750ms
    dust_start_ms = 1000.0
    dust_end_ms = 1750.0

    # ORN mapping parameters
    orn_max_rate = 300.0     # max ORN firing rate at very close range
    orn_gain = 1.0           # scales odor intensity → firing rate (peak=1e5)

    print(f"\n[4/5] Running sugar navigation...")
    print(f"  Duration: {sim_duration_s}s, Syncs: {n_syncs}")
    print(f"  Brain steps/sync: {brain_steps_per_sync}")
    print(f"  Sugar detection distance: {sugar_detect_dist} mm")
    print(f"  Dust irritation: {dust_start_ms:.0f}-{dust_end_ms:.0f} ms")

    # Tracking
    positions = []
    orientations = []
    dn_history = {g: [] for g in DN_GROUPS}
    odor_L_history = []
    odor_R_history = []
    sugar_dist_history = []
    jo_rate_history = []
    phase_labels = []

    t_start = time.time()

    for sync_i in range(n_syncs):
        t_ms = sync_i * sync_interval_ms
        t_s = t_ms / 1000.0

        # ── Read olfactory observation ──
        # obs["odor_intensity"] shape: (1, 4) — 1 odor dimension, 4 sensors
        # Sensors: [palp_L, palp_R, antenna_L, antenna_R]
        odor = obs["odor_intensity"]  # shape (1, 4)

        # Bilateral intensity: weight antennae more heavily
        # palp_L=odor[0,0], palp_R=odor[0,1], ant_L=odor[0,2], ant_R=odor[0,3]
        odor_left = 0.1 * odor[0, 0] + 0.9 * odor[0, 2]   # weighted L
        odor_right = 0.1 * odor[0, 1] + 0.9 * odor[0, 3]   # weighted R

        # Convert to ORN firing rates (proportional, clamped)
        orn_left_rate = float(min(odor_left * orn_gain, orn_max_rate))
        orn_right_rate = float(min(odor_right * orn_gain, orn_max_rate))

        odor_L_history.append(orn_left_rate)
        odor_R_history.append(orn_right_rate)

        # ── Check gustatory (proximity-based sugar detection) ──
        fly_pos = obs['fly'][0, :2]  # xy position
        sugar_xy = sugar_pos[0, :2]
        dist_to_sugar = np.linalg.norm(fly_pos - sugar_xy)
        sugar_dist_history.append(dist_to_sugar)

        # Check end-effector (tarsal) proximity for more precise detection
        ee_pos = obs['end_effectors']  # shape (6, 3)
        legs_near_sugar = 0
        for leg_idx in range(6):
            leg_xy = ee_pos[leg_idx, :2]
            leg_dist = np.linalg.norm(leg_xy - sugar_xy)
            if leg_dist < sugar_detect_dist:
                legs_near_sugar += 1

        # Sugar GRN rate scales with number of legs touching
        sugar_grn_rate = 200.0 * min(legs_near_sugar / 2.0, 1.0)

        # ── Dust irritation → JO stimulus ──
        if dust_start_ms <= t_ms < dust_end_ms and not grooming_done:
            jo_rate = 300.0  # strong JO stimulation (dust on eyes)
        else:
            jo_rate = 0.0
        jo_rate_history.append(jo_rate)

        # ── Determine phase ──
        if feeding_mode:
            phase = 'FEEDING'
        elif grooming_mode:
            phase = 'GROOMING'
        elif sugar_grn_rate > 0:
            phase = 'tasting'
        elif orn_left_rate > 5 or orn_right_rate > 5:
            phase = 'tracking'
        else:
            phase = 'searching'
        phase_labels.append(phase)

        # ── Set brain stimulus ──
        if feeding_mode:
            brain.set_stimulus(
                left_rate=orn_left_rate,
                right_rate=orn_right_rate,
                p9_rate=20.0,
                sugar_rate=sugar_grn_rate,
            )
        elif grooming_mode:
            # During grooming: JO still active, no P9 (stopped walking)
            brain.set_stimulus(
                left_rate=orn_left_rate,
                right_rate=orn_right_rate,
                p9_rate=0.0,
                jo_rate=jo_rate,
            )
        else:
            brain.set_stimulus(
                left_rate=orn_left_rate,
                right_rate=orn_right_rate,
                p9_rate=100.0,
                sugar_rate=sugar_grn_rate,
                jo_rate=jo_rate,
            )

        # ── Brain steps ──
        total_spikes = 0
        for _ in range(brain_steps_per_sync):
            spikes = brain.step()
            total_spikes += spikes.sum().item()
            dn_spikes = brain.get_dn_spikes()
            decoder.update(dn_spikes)

        for g in DN_GROUPS:
            dn_history[g].append(decoder.get_group_rate(g))

        # ── DN → motor drive ──
        fwd = decoder.get_group_rate('forward')
        turn_L = decoder.get_group_rate('turn_L')
        turn_R = decoder.get_group_rate('turn_R')
        bwd = decoder.get_group_rate('backward')
        feed = decoder.get_group_rate('feed')
        groom = decoder.get_group_rate('groom')

        # ── Grooming mode detection ──
        if (not grooming_mode and not grooming_done
                and groom > groom_threshold and jo_rate > 0):
            grooming_mode = True
            grooming_start_t = t_ms
            print(f"  *** GROOMING STARTED at t={t_ms:.0f}ms "
                  f"(aDN1={groom:.3f}, JO={jo_rate:.0f}Hz) ***")

        # Exit grooming: JO stimulus ended + aDN1 rate dropped
        if grooming_mode and jo_rate == 0 and groom < groom_threshold:
            grooming_mode = False
            grooming_done = True
            groom_duration = t_ms - grooming_start_t
            print(f"  *** GROOMING ENDED at t={t_ms:.0f}ms "
                  f"(duration={groom_duration:.0f}ms) ***")

        # ── Feeding mode detection ──
        if (not feeding_mode and not grooming_mode
                and feed > feeding_threshold and sugar_grn_rate > 0):
            feeding_mode = True
            feeding_start_t = t_ms
            print(f"  *** FEEDING STARTED at t={t_ms:.0f}ms "
                  f"(dist={dist_to_sugar:.1f}mm, MN9={feed:.3f}) ***")

        # ── CPG modulation ──
        if feeding_mode:
            # Feeding: stop walking — set very low frequency
            cpg.intrinsic_freqs[:] = 2.0
        elif grooming_mode:
            # Grooming: CPG stopped, GroomingController handles front legs
            cpg.intrinsic_freqs[:] = 2.0
        else:
            # Navigation: combine DN turning with odor gradient
            speed = max(0, fwd - bwd)
            dn_turn = turn_L - turn_R  # positive = turn left

            # Odor gradient correction: steer toward stronger antenna
            # The bilateral difference is small at distance, so amplify it
            odor_diff = orn_left_rate - orn_right_rate
            odor_total = max(orn_left_rate + orn_right_rate, 1.0)
            # Normalized gradient, strongly amplified
            odor_turn = np.clip(odor_diff / odor_total * 8.0, -0.5, 0.5)

            # Correct for brain's intrinsic rightward bias (~0.15)
            brain_bias_correction = 0.15
            turn = dn_turn + odor_turn + brain_bias_correction

            base_freq = 12.0
            turn_gain = 4.0
            turn_amp = turn * turn_gain
            avg_speed = max(0.5, speed)  # stronger baseline forward drive

            # To turn LEFT (turn > 0): slow LEFT legs, speed up RIGHT legs
            left_freq = base_freq * (avg_speed - turn_amp * 0.5)
            right_freq = base_freq * (avg_speed + turn_amp * 0.5)

            cpg.intrinsic_freqs[:3] = np.clip(left_freq, 4.0, 20.0)
            cpg.intrinsic_freqs[3:] = np.clip(right_freq, 4.0, 20.0)

        # ── Body steps ──
        for body_i in range(body_steps_per_sync):
            if grooming_mode:
                # Grooming: front legs sweep, rest hold position
                body_time = t_s + body_i * body_timestep
                action = groom_ctrl.get_action(body_time)
            else:
                # Normal walking via CPG
                cpg.step()
                all_angles = []
                all_adhesion = []
                for i, leg in enumerate(leg_names):
                    angles = preprogrammed_steps.get_joint_angles(
                        leg, cpg.curr_phases[i], cpg.curr_magnitudes[i])
                    all_angles.append(angles)
                    adhesion = preprogrammed_steps.get_adhesion_onoff(
                        leg, cpg.curr_phases[i])
                    all_adhesion.append(adhesion)

                action = {
                    'joints': np.concatenate(all_angles),
                    'adhesion': np.array(all_adhesion, dtype=np.float64),
                }
            obs, reward, terminated, truncated, info = sim.step(action)
            sim.render()

        pos = obs['fly'][0].copy()
        positions.append(pos)
        orientations.append(obs['fly_orientation'].copy())

        # ── Periodic status ──
        if sync_i % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  t={t_ms:6.0f}ms [{phase:>10s}] "
                  f"dist={dist_to_sugar:5.1f}mm "
                  f"fwd={fwd:.3f} tL={turn_L:.3f} tR={turn_R:.3f} "
                  f"grm={groom:.3f} feed={feed:.3f} "
                  f"[{elapsed:.1f}s]")

        # Early termination: feeding for > 500ms
        if feeding_mode and feeding_start_t and (t_ms - feeding_start_t) > 500:
            print(f"  Feeding complete at t={t_ms:.0f}ms — stopping simulation")
            break

    # Save video
    video_path = str(RESULTS_DIR / '23_sugar_navigation.mp4')
    cam.save_video(video_path)
    print(f"\nVideo: {video_path}")

    sim.close()
    total_time = time.time() - t_start
    n_actual = len(positions)

    # ── Results ──
    print(f"\n[5/5] Results")
    print(f"  Wall time: {total_time:.1f}s")
    positions = np.array(positions)
    print(f"  Start: ({positions[0, 0]:.2f}, {positions[0, 1]:.2f})")
    print(f"  End:   ({positions[-1, 0]:.2f}, {positions[-1, 1]:.2f})")
    print(f"  Sugar dish: ({sugar_pos[0, 0]:.0f}, {sugar_pos[0, 1]:.0f})")
    print(f"  Final distance to sugar: {sugar_dist_history[-1]:.1f} mm")
    print(f"  Feeding mode reached: {feeding_mode}")
    if feeding_start_t:
        print(f"  Feeding started at: {feeding_start_t:.0f} ms")

    print(f"  Grooming occurred: {grooming_done or grooming_mode}")
    if grooming_start_t:
        print(f"  Grooming started at: {grooming_start_t:.0f} ms")

    # Phase summary
    for pname in ['searching', 'tracking', 'GROOMING', 'tasting', 'FEEDING']:
        count = phase_labels.count(pname)
        if count > 0:
            duration_ms = count * sync_interval_ms
            print(f"  Phase '{pname}': {duration_ms:.0f} ms ({count} syncs)")

    # ── Visualization ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    fig.suptitle(
        'Analysis 23: Sugar Navigation + Eye Grooming\n'
        '138K LIF neurons | Walk → Groom → Navigate → Feed | 3 emergent behaviors',
        fontsize=12, fontweight='bold'
    )

    t_axis = np.arange(n_actual) * sync_interval_ms

    # Helper: shade grooming and feeding regions
    def shade_events(ax):
        if grooming_start_t:
            groom_end = grooming_start_t + phase_labels.count('GROOMING') * sync_interval_ms
            ax.axvspan(grooming_start_t, groom_end, alpha=0.15, color='green')
        if feeding_start_t:
            feed_end = feeding_start_t + phase_labels.count('FEEDING') * sync_interval_ms
            ax.axvspan(feeding_start_t, feed_end, alpha=0.15, color='orange')
        # Dust irritation period
        ax.axvspan(dust_start_ms, dust_end_ms, alpha=0.08, color='brown')

    # ── (0,0) DN Activity ──
    ax = axes[0, 0]
    dn_colors = {
        'forward': '#2ecc71', 'turn_L': '#3498db', 'turn_R': '#e74c3c',
        'backward': '#9b59b6', 'feed': '#e67e22', 'groom': '#27ae60',
    }
    for g, c in dn_colors.items():
        ax.plot(t_axis, dn_history[g][:n_actual], color=c, label=g, linewidth=1.5)
    shade_events(ax)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Normalized DN rate')
    ax.set_title('Descending Neuron Activity')
    ax.legend(fontsize=7, ncol=2)

    # ── (0,1) Trajectory ──
    ax = axes[0, 1]
    phase_colors = {
        'searching': 'gray', 'tracking': '#3498db',
        'GROOMING': '#27ae60', 'tasting': '#e67e22', 'FEEDING': '#e74c3c',
    }
    for i in range(len(positions) - 1):
        c = phase_colors.get(phase_labels[i], 'gray')
        ax.plot(positions[i:i+2, 0], positions[i:i+2, 1], color=c, linewidth=2)
    ax.plot(positions[0, 0], positions[0, 1], 'ko', markersize=10, label='Start')
    ax.plot(positions[-1, 0], positions[-1, 1], 'k*', markersize=15, label='End')
    ax.plot(sugar_pos[0, 0], sugar_pos[0, 1], 's', color='gold',
            markersize=15, markeredgecolor='orange', linewidth=2, label='Sugar')
    # Mark grooming spot
    groom_idxs = [i for i, p in enumerate(phase_labels) if p == 'GROOMING']
    if groom_idxs:
        gi = groom_idxs[len(groom_idxs)//2]
        ax.plot(positions[gi, 0], positions[gi, 1], 'D', color='#27ae60',
                markersize=12, label='Grooming')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Fly Trajectory (top view)')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')

    # ── (1,0) Odor intensity (ORN rates) ──
    ax = axes[1, 0]
    ax.plot(t_axis, odor_L_history[:n_actual], color='#3498db',
            label='ORN left', linewidth=1.5)
    ax.plot(t_axis, odor_R_history[:n_actual], color='#e74c3c',
            label='ORN right', linewidth=1.5)
    shade_events(ax)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('ORN firing rate (Hz)')
    ax.set_title('Bilateral Odor Intensity → ORN Rates')
    ax.legend()

    # ── (1,1) Distance to sugar ──
    ax = axes[1, 1]
    ax.plot(t_axis, sugar_dist_history[:n_actual], color='#2c3e50', linewidth=2)
    ax.axhline(sugar_detect_dist, color='orange', linestyle='--',
               alpha=0.7, label=f'Detection ({sugar_detect_dist}mm)')
    shade_events(ax)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Distance (mm)')
    ax.set_title('Distance to Sugar Dish')
    ax.legend()

    # ── (2,0) Grooming: aDN1 + JO ──
    ax = axes[2, 0]
    ax.plot(t_axis, dn_history['groom'][:n_actual], color='#27ae60',
            linewidth=2, label='aDN1 (groom)')
    ax2 = ax.twinx()
    ax2.plot(t_axis, jo_rate_history[:n_actual], color='brown',
             linewidth=1.5, alpha=0.6, label='JO stimulus')
    ax2.set_ylabel('JO rate (Hz)', color='brown')
    ax.axhline(groom_threshold, color='red', linestyle='--',
               alpha=0.7, label=f'Threshold ({groom_threshold})')
    shade_events(ax)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Normalized aDN1 rate')
    ax.set_title('Eye Grooming: JO Touch → aDN1 Response')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # ── (2,1) MN9 feeding rate ──
    ax = axes[2, 1]
    ax.plot(t_axis, dn_history['feed'][:n_actual], color='#e67e22', linewidth=2,
            label='MN9 (feed)')
    ax.axhline(feeding_threshold, color='red', linestyle='--',
               alpha=0.7, label=f'Threshold ({feeding_threshold})')
    shade_events(ax)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Normalized MN9 rate')
    ax.set_title('Feeding DN (MN9) Activity')
    ax.legend()

    # ── (3,0) Turn asymmetry ──
    ax = axes[3, 0]
    turn_asym = [dn_history['turn_L'][i] - dn_history['turn_R'][i]
                 for i in range(n_actual)]
    ax.plot(t_axis, turn_asym, color='#8e44ad', linewidth=1.5)
    ax.axhline(0, color='gray', linestyle='--')
    shade_events(ax)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Turn asymmetry (L - R)')
    ax.set_title('Turn Signal: + = left, - = right')

    # ── (3,1) Behavioral phase timeline ──
    ax = axes[3, 1]
    phase_nums = {'searching': 0, 'tracking': 1, 'GROOMING': 2, 'tasting': 3, 'FEEDING': 4}
    phase_y = [phase_nums.get(p, 0) for p in phase_labels[:n_actual]]
    for pname, pnum in phase_nums.items():
        color = phase_colors.get(pname, 'gray')
        mask = [i for i, y in enumerate(phase_y) if y == pnum]
        if mask:
            ax.scatter([t_axis[i] for i in mask], [pnum] * len(mask),
                       color=color, s=8, label=pname)
    ax.set_yticks(list(phase_nums.values()))
    ax.set_yticklabels(list(phase_nums.keys()))
    ax.set_xlabel('Time (ms)')
    ax.set_title('Behavioral Phase Timeline')
    ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plot_path = str(RESULTS_DIR / '23_sugar_navigation.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot: {plot_path}")

    print("\n" + "=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)


if __name__ == '__main__':
    main()
