"""
Analysis 24: Autonomous Fly — Hunger, Grooming, Navigation, Feeding

Fully autonomous brain-controlled fly with internal motivation:
  1. Hunger drive → P9 forward locomotion (no external stimulation)
  2. Olfactory navigation → bilateral ORN → turn toward sugar
  3. Eye grooming → dust/JO → aDN1 → front-leg cleaning
  4. Gustatory feeding → sugar GRN → MN9 → stop + feed
  5. Dopamine reward → hunger decreases → satiation → stop

Autonomy loop:
  Hunger ↑ → P9 drive → walk → smell sugar → navigate →
  DUST! → groom → resume → reach sugar → taste → feed →
  DA reward → hunger ↓ → satiated → stop
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
# DN + STIMULI DEFINITIONS
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

SUGAR_GRN_IDS = [
    720575940624963786, 720575940630233916, 720575940637568838,
    720575940638202345, 720575940617000768, 720575940630797113,
    720575940632889389, 720575940621754367, 720575940621502051,
    720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940639259967, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570,
    720575940628853239, 720575940629176663, 720575940611875570,
]

JO_NEURON_IDS = [
    720575940645106376, 720575940615272415, 720575940619869120,
    720575940620257345, 720575940620382889, 720575940630834683,
    720575940632449619, 720575940634020508, 720575940605530302,
    720575940607140035, 720575940608742409, 720575940615590843,
    720575940620410177, 720575940621870618, 720575940622344170,
    720575940623298559, 720575940626042149, 720575940627379333,
    720575940630080071, 720575940632128031, 720575940632307527,
    720575940634820703,
    720575940606154370, 720575940605919334, 720575940608884931,
    720575940616655989, 720575940620543110, 720575940622937528,
    720575940624799290, 720575940626565455, 720575940627941431,
    720575940627977457, 720575940628160617, 720575940629188251,
    720575940641921421,
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
    720575940609522461, 720575940610261346, 720575940613641915,
    720575940615469785, 720575940616589878, 720575940616951124,
    720575940619479979, 720575940621218985, 720575940628444667,
    720575940634634606, 720575940640753267, 720575940650244342,
    720575940615573597, 720575940615848788, 720575940619083349,
    720575940621397417, 720575940621625597, 720575940622283912,
    720575940627049731, 720575940629022149, 720575940630122015,
    720575940630564179, 720575940633153375, 720575940637410869,
    720575940638681845, 720575940621033477, 720575940621776410,
    720575940621815690, 720575940622234211, 720575940622635817,
    720575940623897096, 720575940626148354, 720575940626540821,
    720575940628258715, 720575940629743063, 720575940630202624,
    720575940630544967, 720575940633553820, 720575940644036644,
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
    720575940620444654, 720575940631866508, 720575940607853833,
    720575940611088563, 720575940612773374, 720575940613221928,
    720575940615024543, 720575940615986459, 720575940617811013,
    720575940618467195, 720575940621442224, 720575940622199977,
    720575940624915230, 720575940625559358, 720575940627104649,
    720575940627314088, 720575940633058989, 720575940636335735,
    720575940605800369, 720575940608784579, 720575940618135109,
    720575940626719101, 720575940629296185, 720575940636137591,
    720575940602720940, 720575940610079857, 720575940614427195,
    720575940616501787, 720575940617156445, 720575940625909962,
    720575940626241369, 720575940629105658, 720575940629138959,
    720575940636559534, 720575940641372661,
    720575940626135548, 720575940627751567, 720575940604753437,
    720575940613971485, 720575940614835362, 720575940623399059,
    720575940630319671, 720575940639082062,
    720575940607386307, 720575940634512992, 720575940614035485,
    720575940618901424, 720575940630070343, 720575940633443353,
    720575940635058612, 720575940637632419, 720575940625626000,
]

# =====================================================================
# HUNGER STATE — Internal motivation drive
# =====================================================================

class HungerState:
    """Internal hunger motivation: drives P9, reduced by feeding reward.

    hunger=0.8 → aç sinek, P9 ~96Hz → hızlı yürüyüş
    hunger=0.1 → doymuş sinek → durur
    """

    def __init__(self, initial=0.8, increase_rate=0.02,
                 reward_decrease=0.15, satiation_threshold=0.1):
        self.level = initial
        self.increase_rate = increase_rate       # per second
        self.reward_decrease = reward_decrease    # per second while feeding
        self.satiation_threshold = satiation_threshold
        self.total_reward = 0.0

    @property
    def p9_rate(self):
        """Hunger → P9 firing rate. Hungrier fly walks faster."""
        return self.level * 120.0  # 0-120 Hz

    @property
    def is_satiated(self):
        return self.level < self.satiation_threshold

    def tick(self, dt_s, is_feeding, pam_activity=0.0):
        """Update hunger each sync cycle."""
        # Natural hunger increase over time
        self.level = min(1.0, self.level + self.increase_rate * dt_s)
        # Feeding reduces hunger, amplified by PAM dopamine reward
        if is_feeding:
            reward = self.reward_decrease * (1.0 + pam_activity * 5.0)
            self.level = max(0.0, self.level - reward * dt_s)
            self.total_reward += reward * dt_s


# =====================================================================
# GROOMING CONTROLLER
# =====================================================================

class GroomingController:
    """Front-leg oscillation for eye/antennal grooming."""

    def __init__(self, preprogrammed_steps=None, freq_hz=4.0):
        self.steps = preprogrammed_steps or PreprogrammedSteps()
        self.freq = freq_hz
        self.neutral = np.zeros(42)
        for i, leg in enumerate(self.steps.legs):
            self.neutral[i * 7:(i + 1) * 7] = self.steps.get_joint_angles(
                leg, np.pi, 0.0
            )

    def get_action(self, time_s):
        joints = self.neutral.copy()
        phase = 2 * np.pi * self.freq * time_s
        femur_offset = 0.3 * np.sin(phase)
        tibia_offset = 0.4 * np.sin(phase + np.pi / 2)
        for base in (0, 21):
            joints[base + 3] += femur_offset
            joints[base + 5] += tibia_offset
        adhesion = np.array([0, 1, 1, 0, 1, 1], dtype=np.float64)
        return {"joints": joints, "adhesion": adhesion}


# =====================================================================
# FAST BRAIN ENGINE (with PAM monitoring)
# =====================================================================

class FastBrainEngine:
    """LIF brain with scipy sparse matmul + PAM dopamine monitoring."""

    def __init__(self):
        comp_path = str(DATA_DIR / '2025_Completeness_783.csv')
        conn_path = str(DATA_DIR / '2025_Connectivity_783.parquet')

        self.flyid2i, self.i2flyid = get_hash_tables(comp_path)
        self.num_neurons = len(self.flyid2i)

        print(f"Loading {self.num_neurons} neurons...")
        weights = get_weights(conn_path, comp_path, str(DATA_DIR))

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

        # JO neuron indices
        self.jo_indices = [self.flyid2i[nid] for nid in JO_NEURON_IDS
                           if nid in self.flyid2i]

        # PAM dopamine neuron indices (passive monitoring — reward signal)
        ann_pam = ann[ann['cell_type'].str.startswith('PAM', na=False)]
        self.pam_indices = [self.flyid2i[int(r)] for r in
                            ann_pam['root_id'].values
                            if int(r) in self.flyid2i]

        # Neuromodulation: find sugar→X→PAM intermediate neurons
        # and store original weights for dynamic boosting
        self._find_sugar_pam_pathway()

        print(f"  DN mapped: {len(self.dn_indices)}/{len(DN_NEURONS)}")
        print(f"  ORN left: {len(self.orn_left)}, right: {len(self.orn_right)}")
        print(f"  Sugar GRNs: {len(self.sugar_grn_indices)}/{len(SUGAR_GRN_IDS)}")
        print(f"  JO neurons: {len(self.jo_indices)}/{len(JO_NEURON_IDS)}")
        print(f"  PAM dopamine: {len(self.pam_indices)} neurons (passive monitoring)")
        print(f"  Neuromodulation: {len(self.neuromod_intermediates)} intermediate neurons on sugar→PAM path")

    def _find_sugar_pam_pathway(self):
        """Find intermediate neurons on sugar GRN → X → PAM path.

        Uses raw connectivity (not the signed weight matrix) to find the path,
        because some connections are zeroed out in w_scipy due to
        excitatory/inhibitory sign transformation. Then injects new excitatory
        weights along this path when neuromodulation is enabled.
        """
        import pandas as pd
        conn = pd.read_parquet(str(DATA_DIR / '2025_Connectivity_783.parquet'))
        N = self.num_neurons
        adj = sp.csr_matrix(
            (conn['Connectivity'].values.astype(np.float32),
             (conn['Presynaptic_Index'].values, conn['Postsynaptic_Index'].values)),
            shape=(N, N))

        pam_set = set(self.pam_indices)
        sugar_set = set(self.sugar_grn_indices)

        # Find intermediates using raw connectivity
        self.neuromod_intermediates = []
        for s_idx in sugar_set:
            row = adj.getrow(s_idx)
            for mid_idx in row.indices:
                if mid_idx in self.neuromod_intermediates:
                    continue
                mid_row = adj.getrow(mid_idx)
                if set(mid_row.indices) & pam_set:
                    self.neuromod_intermediates.append(mid_idx)

        # Collect entries to boost: sugar→inter and inter→PAM
        # Use raw connectivity weights as base (always positive)
        self._neuromod_inter_pam = []  # (mid_idx, pam_idx, raw_weight)
        for mid_idx in self.neuromod_intermediates:
            row = adj.getrow(mid_idx)
            for col_idx, val in zip(row.indices, row.data):
                if col_idx in pam_set:
                    self._neuromod_inter_pam.append((mid_idx, col_idx, float(val)))

        self._neuromod_sugar_inter = []  # (sugar_idx, mid_idx, raw_weight)
        mid_set = set(self.neuromod_intermediates)
        for s_idx in sugar_set:
            row = adj.getrow(s_idx)
            for col_idx, val in zip(row.indices, row.data):
                if col_idx in mid_set:
                    self._neuromod_sugar_inter.append((s_idx, col_idx, float(val)))

        self._neuromod_active = False
        del adj, conn

    def enable_neuromodulation(self, boost_factor=8.0):
        """Boost sugar→X→PAM weights (simulates serotonin/octopamine).

        Injects excitatory weights along the sugar→intermediate→PAM path.
        Uses raw connectivity values × boost_factor as the weight.
        """
        if self._neuromod_active:
            return
        # Save originals and set boosted values
        self._saved_inter_pam = []
        for mid_idx, pam_idx, raw_w in self._neuromod_inter_pam:
            orig = self.w_scipy[mid_idx, pam_idx]
            self._saved_inter_pam.append((mid_idx, pam_idx, orig))
            self.w_scipy[mid_idx, pam_idx] = raw_w * boost_factor

        self._saved_sugar_inter = []
        for s_idx, mid_idx, raw_w in self._neuromod_sugar_inter:
            orig = self.w_scipy[s_idx, mid_idx]
            self._saved_sugar_inter.append((s_idx, mid_idx, orig))
            self.w_scipy[s_idx, mid_idx] = raw_w * boost_factor

        self.w_scipy_t = self.w_scipy.T.tocsr()
        self._neuromod_active = True

    def disable_neuromodulation(self):
        """Restore original weights."""
        if not self._neuromod_active:
            return
        for mid_idx, pam_idx, orig in self._saved_inter_pam:
            self.w_scipy[mid_idx, pam_idx] = orig
        for s_idx, mid_idx, orig in self._saved_sugar_inter:
            self.w_scipy[s_idx, mid_idx] = orig
        self.w_scipy_t = self.w_scipy.T.tocsr()
        self._neuromod_active = False

    def set_stimulus(self, left_rate=0.0, right_rate=0.0, p9_rate=0.0,
                     sugar_rate=0.0, jo_rate=0.0):
        """Set stimulus rates. p9_rate now comes from HungerState."""
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
            adn1_direct_rate = jo_rate * 0.3
            for name in ('aDN1_left', 'aDN1_right'):
                fid = DN_NEURONS.get(name)
                if fid and fid in self.flyid2i:
                    self.rates[0, self.flyid2i[fid]] = adn1_direct_rate

    @torch.no_grad()
    def step(self):
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

    def get_pam_activity(self):
        """PAM population mean spike fraction — passive reward monitoring."""
        spk = self.state[2]
        if not self.pam_indices:
            return 0.0
        return float(spk[0, self.pam_indices].mean())


# =====================================================================
# DN DECODER
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
    print("ANALYSIS 24: AUTONOMOUS FLY")
    print("Internal Hunger → Walk → Groom → Navigate → Feed → Satiated → Stop")
    print("=" * 72)

    torch.set_num_threads(10)

    # ── Brain ──
    print("\n[1/5] Loading optimized brain model...")
    t0 = time.time()
    brain = FastBrainEngine()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    decoder = DNDecoder(window_ms=50.0)

    # ── Internal state ──
    hunger = HungerState(initial=0.8, increase_rate=0.02,
                         reward_decrease=0.15, satiation_threshold=0.1)
    print(f"\n  Hunger: initial={hunger.level:.1f}, "
          f"satiation threshold={hunger.satiation_threshold}")

    # ── Arena ──
    print("\n[2/5] Creating odor arena with sugar dish...")
    sugar_pos = np.array([[25, 0, 1.5]])
    arena = OdorArena(
        size=(300, 300),
        odor_source=sugar_pos,
        peak_odor_intensity=np.array([[2e4]]),
        diffuse_func=lambda x: x**-2,
        marker_colors=[(1.0, 0.85, 0.0, 1.0)],
        marker_size=0.5,
    )
    print(f"  Sugar dish at [{sugar_pos[0, 0]:.0f}, {sugar_pos[0, 1]:.0f}] mm")

    # ── Body ──
    print("\n[3/5] Creating fly body with olfaction...")
    body_timestep = 1e-4
    fly = Fly(
        enable_olfaction=True, enable_adhesion=True, draw_adhesion=True,
        init_pose='stretch', control='position',
    )
    cam = Camera(
        attachment_point=fly.model.worldbody,
        camera_name='camera_right',
        targeted_fly_names=[fly.name],
        play_speed=1.0, window_size=(1280, 720), fps=30,
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

    # ── Params ──
    sim_duration_s = 10.0  # max, will stop early on satiation
    sync_interval_ms = 15.0
    sync_dt_s = sync_interval_ms / 1000.0
    brain_steps_per_sync = int(sync_interval_ms / DT)  # 150 steps = 22.5ms
    body_steps_per_sync = int(sync_interval_ms / (body_timestep * 1000))
    n_syncs = int(sim_duration_s * 1000 / sync_interval_ms)

    sugar_detect_dist = 5.0
    feeding_threshold = 0.03
    feeding_mode = False
    feeding_start_t = None

    groom_threshold = 0.02
    grooming_mode = False
    grooming_start_t = None
    grooming_done = False
    groom_ctrl = GroomingController(preprogrammed_steps)

    dust_start_ms = 1000.0
    dust_end_ms = 1750.0

    orn_max_rate = 300.0
    orn_gain = 1.0

    print(f"\n[4/5] Running autonomous simulation...")
    print(f"  Max duration: {sim_duration_s}s, Syncs: {n_syncs}")
    print(f"  Hunger drives P9: {hunger.level:.1f} → {hunger.p9_rate:.0f}Hz")
    print(f"  Satiation stops at hunger < {hunger.satiation_threshold}")
    print(f"  Dust: {dust_start_ms:.0f}-{dust_end_ms:.0f}ms")

    # Tracking
    positions = []
    dn_history = {g: [] for g in DN_GROUPS}
    odor_L_history = []
    odor_R_history = []
    sugar_dist_history = []
    jo_rate_history = []
    hunger_history = []
    pam_history = []
    p9_rate_history = []
    phase_labels = []

    pam_accumulator = 0.0
    pam_count = 0

    t_start = time.time()

    for sync_i in range(n_syncs):
        t_ms = sync_i * sync_interval_ms
        t_s = t_ms / 1000.0

        # ── Hunger state update ──
        pam_avg = pam_accumulator / max(pam_count, 1)
        hunger.tick(sync_dt_s, feeding_mode, pam_avg)
        hunger_history.append(hunger.level)
        pam_history.append(pam_avg)
        p9_rate_history.append(hunger.p9_rate)
        pam_accumulator = 0.0
        pam_count = 0

        # ── Satiation check ──
        if hunger.is_satiated and feeding_mode:
            print(f"  *** SATIATED at t={t_ms:.0f}ms "
                  f"(hunger={hunger.level:.3f}) — fly stops autonomously ***")
            phase_labels.append('SATIATED')
            positions.append(obs['fly'][0].copy())
            for g in DN_GROUPS:
                dn_history[g].append(decoder.get_group_rate(g))
            odor_L_history.append(0)
            odor_R_history.append(0)
            sugar_dist_history.append(sugar_dist_history[-1] if sugar_dist_history else 0)
            jo_rate_history.append(0)
            break

        # ── Olfaction ──
        odor = obs["odor_intensity"]
        odor_left = 0.1 * odor[0, 0] + 0.9 * odor[0, 2]
        odor_right = 0.1 * odor[0, 1] + 0.9 * odor[0, 3]
        orn_left_rate = float(min(odor_left * orn_gain, orn_max_rate))
        orn_right_rate = float(min(odor_right * orn_gain, orn_max_rate))
        odor_L_history.append(orn_left_rate)
        odor_R_history.append(orn_right_rate)

        # ── Gustation ──
        fly_pos = obs['fly'][0, :2]
        sugar_xy = sugar_pos[0, :2]
        dist_to_sugar = np.linalg.norm(fly_pos - sugar_xy)
        sugar_dist_history.append(dist_to_sugar)

        ee_pos = obs['end_effectors']
        legs_near_sugar = 0
        for leg_idx in range(6):
            leg_dist = np.linalg.norm(ee_pos[leg_idx, :2] - sugar_xy)
            if leg_dist < sugar_detect_dist:
                legs_near_sugar += 1
        sugar_grn_rate = 200.0 * min(legs_near_sugar / 2.0, 1.0)

        # ── Dust ──
        jo_rate = 300.0 if (dust_start_ms <= t_ms < dust_end_ms
                            and not grooming_done) else 0.0
        jo_rate_history.append(jo_rate)

        # ── Phase ──
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

        # ── P9 rate from hunger (the key autonomy change!) ──
        current_p9 = hunger.p9_rate

        # ── Neuromodulation: boost sugar→PAM pathway when tasting ──
        if sugar_grn_rate > 0:
            brain.enable_neuromodulation(boost_factor=8.0)
        else:
            brain.disable_neuromodulation()

        # ── Brain stimulus ──
        if feeding_mode:
            brain.set_stimulus(
                left_rate=orn_left_rate, right_rate=orn_right_rate,
                p9_rate=current_p9 * 0.2,  # reduced but hunger-driven
                sugar_rate=sugar_grn_rate,
            )
        elif grooming_mode:
            brain.set_stimulus(
                left_rate=orn_left_rate, right_rate=orn_right_rate,
                p9_rate=0.0, jo_rate=jo_rate,
            )
        else:
            brain.set_stimulus(
                left_rate=orn_left_rate, right_rate=orn_right_rate,
                p9_rate=current_p9,  # HUNGER-DRIVEN, not fixed!
                sugar_rate=sugar_grn_rate, jo_rate=jo_rate,
            )

        # ── Brain steps ──
        for _ in range(brain_steps_per_sync):
            spikes = brain.step()
            dn_spikes = brain.get_dn_spikes()
            decoder.update(dn_spikes)
            # Accumulate PAM activity
            pam_accumulator += brain.get_pam_activity()
            pam_count += 1

        for g in DN_GROUPS:
            dn_history[g].append(decoder.get_group_rate(g))

        # ── DN decode ──
        fwd = decoder.get_group_rate('forward')
        turn_L = decoder.get_group_rate('turn_L')
        turn_R = decoder.get_group_rate('turn_R')
        bwd = decoder.get_group_rate('backward')
        feed = decoder.get_group_rate('feed')
        groom = decoder.get_group_rate('groom')

        # ── Grooming detection ──
        if (not grooming_mode and not grooming_done
                and groom > groom_threshold and jo_rate > 0):
            grooming_mode = True
            grooming_start_t = t_ms
            print(f"  *** GROOMING at t={t_ms:.0f}ms (aDN1={groom:.3f}) ***")

        if grooming_mode and jo_rate == 0 and groom < groom_threshold:
            grooming_mode = False
            grooming_done = True
            print(f"  *** GROOMING ENDED at t={t_ms:.0f}ms ***")

        # ── Feeding detection ──
        if (not feeding_mode and not grooming_mode
                and feed > feeding_threshold and sugar_grn_rate > 0):
            feeding_mode = True
            feeding_start_t = t_ms
            print(f"  *** FEEDING at t={t_ms:.0f}ms "
                  f"(dist={dist_to_sugar:.1f}mm, hunger={hunger.level:.2f}) ***")

        # ── CPG modulation ──
        if feeding_mode or grooming_mode:
            cpg.intrinsic_freqs[:] = 2.0
        else:
            speed = max(0, fwd - bwd)
            dn_turn = turn_L - turn_R
            odor_diff = orn_left_rate - orn_right_rate
            odor_total = max(orn_left_rate + orn_right_rate, 1.0)
            odor_turn = np.clip(odor_diff / odor_total * 8.0, -0.5, 0.5)
            brain_bias_correction = 0.15
            turn = dn_turn + odor_turn + brain_bias_correction

            base_freq = 12.0
            turn_gain = 4.0
            turn_amp = turn * turn_gain
            avg_speed = max(0.5, speed)
            left_freq = base_freq * (avg_speed - turn_amp * 0.5)
            right_freq = base_freq * (avg_speed + turn_amp * 0.5)
            cpg.intrinsic_freqs[:3] = np.clip(left_freq, 4.0, 20.0)
            cpg.intrinsic_freqs[3:] = np.clip(right_freq, 4.0, 20.0)

        # ── Body steps ──
        for body_i in range(body_steps_per_sync):
            if grooming_mode:
                body_time = t_s + body_i * body_timestep
                action = groom_ctrl.get_action(body_time)
            else:
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

        positions.append(obs['fly'][0].copy())

        # ── Status ──
        if sync_i % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  t={t_ms:6.0f}ms [{phase:>10s}] "
                  f"hunger={hunger.level:.2f} P9={current_p9:.0f}Hz "
                  f"dist={dist_to_sugar:5.1f}mm "
                  f"PAM={pam_avg:.4f} "
                  f"[{elapsed:.1f}s]")

    # ── Save ──
    video_path = str(RESULTS_DIR / '24_autonomous_fly.mp4')
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
    print(f"  Final hunger: {hunger_history[-1]:.3f}")
    print(f"  Total DA reward: {hunger.total_reward:.4f}")
    print(f"  Satiated: {hunger.is_satiated}")
    print(f"  Grooming: {grooming_done}")
    print(f"  Feeding: {feeding_mode}")
    print(f"  Final distance: {sugar_dist_history[-1]:.1f}mm")

    for pname in ['searching', 'tracking', 'GROOMING', 'tasting', 'FEEDING', 'SATIATED']:
        count = phase_labels.count(pname)
        if count > 0:
            print(f"  Phase '{pname}': {count * sync_interval_ms:.0f}ms")

    # ── Visualization (5×2 = 10 panels) ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 2, figsize=(16, 22))
    fig.suptitle(
        'Analysis 24: Autonomous Fly — Internal Hunger Drive\n'
        '138K LIF neurons | Hunger → Walk → Groom → Navigate → Feed → Satiated',
        fontsize=12, fontweight='bold'
    )

    t_axis = np.arange(n_actual) * sync_interval_ms

    phase_colors = {
        'searching': 'gray', 'tracking': '#3498db',
        'GROOMING': '#27ae60', 'tasting': '#e67e22',
        'FEEDING': '#e74c3c', 'SATIATED': '#2c3e50',
    }

    def shade_events(ax):
        if grooming_start_t:
            groom_end = grooming_start_t + phase_labels.count('GROOMING') * sync_interval_ms
            ax.axvspan(grooming_start_t, groom_end, alpha=0.12, color='green')
        if feeding_start_t:
            feed_end_t = t_axis[-1] if 'SATIATED' in phase_labels else (
                feeding_start_t + phase_labels.count('FEEDING') * sync_interval_ms)
            ax.axvspan(feeding_start_t, feed_end_t, alpha=0.12, color='orange')
        ax.axvspan(dust_start_ms, dust_end_ms, alpha=0.06, color='brown')

    # (0,0) DN Activity
    ax = axes[0, 0]
    dn_c = {'forward': '#2ecc71', 'turn_L': '#3498db', 'turn_R': '#e74c3c',
            'backward': '#9b59b6', 'feed': '#e67e22', 'groom': '#27ae60'}
    for g, c in dn_c.items():
        ax.plot(t_axis, dn_history[g][:n_actual], color=c, label=g, linewidth=1.5)
    shade_events(ax)
    ax.set_ylabel('Normalized DN rate'); ax.set_title('Descending Neuron Activity')
    ax.legend(fontsize=6, ncol=3)

    # (0,1) Trajectory
    ax = axes[0, 1]
    for i in range(len(positions) - 1):
        c = phase_colors.get(phase_labels[i], 'gray')
        ax.plot(positions[i:i+2, 0], positions[i:i+2, 1], color=c, linewidth=2)
    ax.plot(positions[0, 0], positions[0, 1], 'ko', ms=10, label='Start')
    ax.plot(positions[-1, 0], positions[-1, 1], 'k*', ms=15, label='End')
    ax.plot(sugar_pos[0, 0], sugar_pos[0, 1], 's', color='gold', ms=15,
            markeredgecolor='orange', lw=2, label='Sugar')
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
    ax.set_title('Fly Trajectory'); ax.legend(fontsize=8); ax.set_aspect('equal')

    # (1,0) HUNGER + P9 rate (KEY NEW PANEL)
    ax = axes[1, 0]
    ax.plot(t_axis, hunger_history[:n_actual], color='#c0392b', linewidth=2.5,
            label='Hunger level')
    ax.axhline(hunger.satiation_threshold, color='green', ls='--', alpha=0.7,
               label=f'Satiation ({hunger.satiation_threshold})')
    ax2 = ax.twinx()
    ax2.plot(t_axis, p9_rate_history[:n_actual], color='#2ecc71', linewidth=1.5,
             alpha=0.7, label='P9 rate (Hz)')
    ax2.set_ylabel('P9 rate (Hz)', color='#2ecc71')
    shade_events(ax)
    ax.set_ylabel('Hunger (0-1)'); ax.set_title('Internal Hunger State → P9 Drive')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # (1,1) PAM dopamine activity (KEY NEW PANEL)
    ax = axes[1, 1]
    ax.plot(t_axis, pam_history[:n_actual], color='#8e44ad', linewidth=2,
            label='PAM dopamine')
    shade_events(ax)
    ax.set_ylabel('PAM mean spike fraction'); ax.set_title('PAM Dopamine Reward Signal')
    ax.legend()

    # (2,0) ORN rates
    ax = axes[2, 0]
    ax.plot(t_axis, odor_L_history[:n_actual], color='#3498db', label='ORN left', lw=1.5)
    ax.plot(t_axis, odor_R_history[:n_actual], color='#e74c3c', label='ORN right', lw=1.5)
    shade_events(ax)
    ax.set_ylabel('ORN rate (Hz)'); ax.set_title('Bilateral Odor → ORN'); ax.legend()

    # (2,1) Distance
    ax = axes[2, 1]
    ax.plot(t_axis, sugar_dist_history[:n_actual], color='#2c3e50', linewidth=2)
    ax.axhline(sugar_detect_dist, color='orange', ls='--', alpha=0.7,
               label=f'Detection ({sugar_detect_dist}mm)')
    shade_events(ax)
    ax.set_ylabel('Distance (mm)'); ax.set_title('Distance to Sugar'); ax.legend()

    # (3,0) Grooming
    ax = axes[3, 0]
    ax.plot(t_axis, dn_history['groom'][:n_actual], color='#27ae60', lw=2, label='aDN1')
    ax2 = ax.twinx()
    ax2.plot(t_axis, jo_rate_history[:n_actual], color='brown', lw=1.5, alpha=0.6, label='JO')
    ax2.set_ylabel('JO (Hz)', color='brown')
    shade_events(ax)
    ax.set_ylabel('aDN1 rate'); ax.set_title('Eye Grooming: JO → aDN1')
    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, fontsize=8)

    # (3,1) MN9 feeding
    ax = axes[3, 1]
    ax.plot(t_axis, dn_history['feed'][:n_actual], color='#e67e22', lw=2, label='MN9')
    ax.axhline(feeding_threshold, color='red', ls='--', alpha=0.7)
    shade_events(ax)
    ax.set_ylabel('MN9 rate'); ax.set_title('Feeding DN (MN9)'); ax.legend()

    # (4,0) Turn asymmetry
    ax = axes[4, 0]
    turn_asym = [dn_history['turn_L'][i] - dn_history['turn_R'][i] for i in range(n_actual)]
    ax.plot(t_axis, turn_asym, color='#8e44ad', lw=1.5)
    ax.axhline(0, color='gray', ls='--')
    shade_events(ax)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Turn (L-R)')
    ax.set_title('Turn Asymmetry')

    # (4,1) Phase timeline
    ax = axes[4, 1]
    phase_nums = {'searching': 0, 'tracking': 1, 'GROOMING': 2,
                  'tasting': 3, 'FEEDING': 4, 'SATIATED': 5}
    phase_y = [phase_nums.get(p, 0) for p in phase_labels[:n_actual]]
    for pname, pnum in phase_nums.items():
        color = phase_colors.get(pname, 'gray')
        mask = [i for i, y in enumerate(phase_y) if y == pnum]
        if mask:
            ax.scatter([t_axis[i] for i in mask], [pnum] * len(mask),
                       color=color, s=8, label=pname)
    ax.set_yticks(list(phase_nums.values()))
    ax.set_yticklabels(list(phase_nums.keys()))
    ax.set_xlabel('Time (ms)'); ax.set_title('Behavioral Phase Timeline')
    ax.legend(fontsize=6, ncol=3)

    plt.tight_layout()
    plot_path = str(RESULTS_DIR / '24_autonomous_fly.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot: {plot_path}")

    print("\n" + "=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)


if __name__ == '__main__':
    main()
