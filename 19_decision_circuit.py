"""
FlyWire Analysis 19 - Decision-Making Circuit Simulation

Models the complete sensory → valence → action pathway:
  ORN → PN → KC → MBON (approach vs avoidance) → downstream targets → motor

Key questions:
1. How does the MB compute valence (approach vs avoidance)?
2. How do approach and avoidance signals compete?
3. What role does the PPL1/PAM feedback play in decision bias?
4. How does the Lateral Horn (innate) vs MB (learned) pathway compare?
5. Does GABA inhibition shape the decision threshold?

Data: FlyWire v783 connectome (Dorkenwald et al., 2024).
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 80)
print("ANALYSIS 19: DECISION-MAKING CIRCUIT SIMULATION")
print("=" * 80)

# =====================================================================
# 1. LOAD DATA
# =====================================================================
print("\nLoading data...")
conn = pd.read_feather(os.path.join(PROJECT_DIR, "data/proofread_connections_783.feather"))
ann = pd.read_csv(os.path.join(PROJECT_DIR, "data/neuron_annotations.tsv"), sep="\t", low_memory=False)

# =====================================================================
# 2. DEFINE NEURON GROUPS
# =====================================================================
print("Defining neuron groups...")

orn_ids = set(ann[ann['cell_class'] == 'olfactory']['root_id'])
pn_ids = set(ann[ann['cell_class'].isin(['ALPN', 'ALLN'])]['root_id'])
kc_ids = set(ann[ann['cell_class'] == 'Kenyon_Cell']['root_id'])
mbon_ids = set(ann[ann['cell_class'] == 'MBON']['root_id'])
pam_ids = set(ann[ann['cell_type'].str.startswith('PAM', na=False)]['root_id'])
ppl1_ids = set(ann[ann['cell_type'].str.startswith('PPL1', na=False)]['root_id'])
dan_ids = pam_ids | ppl1_ids
desc_ids = set(ann[ann['super_class'] == 'descending']['root_id'])
motor_ids = set(ann[ann['super_class'] == 'motor']['root_id'])
cx_ids = set(ann[ann['cell_class'] == 'CX']['root_id'])
fb_ids = set(ann[ann['cell_type'].str.contains('FB', na=False)]['root_id'])
lh_ids = set(ann[ann['cell_class'].str.contains('LH', na=False)]['root_id'])
gaba_ids = set(ann[ann['top_nt'] == 'gaba']['root_id'])

# Classify MBONs by NT type (proxy for valence)
# Glutamate MBONs → mostly avoidance (Aso et al. 2014)
# ACh MBONs → mostly approach
# GABA MBONs → mostly suppression/modulation
mbon_ann = ann[ann['cell_class'] == 'MBON']
approach_mbon_ids = set(mbon_ann[mbon_ann['top_nt'] == 'acetylcholine']['root_id'])  # 52
avoidance_mbon_ids = set(mbon_ann[mbon_ann['top_nt'] == 'glutamate']['root_id'])    # 25
suppress_mbon_ids = set(mbon_ann[mbon_ann['top_nt'] == 'gaba']['root_id'])           # 19

print(f"  ORN: {len(orn_ids)}")
print(f"  PN (ALPN+ALLN): {len(pn_ids)}")
print(f"  Kenyon Cells: {len(kc_ids)}")
print(f"  MBON total: {len(mbon_ids)}")
print(f"    Approach (ACh): {len(approach_mbon_ids)}")
print(f"    Avoidance (Glut): {len(avoidance_mbon_ids)}")
print(f"    Suppression (GABA): {len(suppress_mbon_ids)}")
print(f"  PAM (reward): {len(pam_ids)}")
print(f"  PPL1 (punishment): {len(ppl1_ids)}")
print(f"  Central Complex: {len(cx_ids)}")
print(f"  Fan-shaped Body: {len(fb_ids)}")
print(f"  Lateral Horn: {len(lh_ids)}")
print(f"  Descending: {len(desc_ids)}")
print(f"  Motor: {len(motor_ids)}")

# =====================================================================
# 3. BUILD NETWORK
# =====================================================================
print("\nBuilding network...")
strong = conn[conn['syn_count'] >= 3].copy()

# NT sign
def get_nt_sign(row):
    scores = {
        'ach': row['ach_avg'], 'gaba': row['gaba_avg'], 'glut': row['glut_avg'],
        'oct': row['oct_avg'], 'ser': row['ser_avg'], 'da': row['da_avg'],
    }
    dominant = max(scores, key=scores.get)
    return -1.0 if dominant == 'gaba' else 1.0

strong['nt_sign'] = strong.apply(get_nt_sign, axis=1)

# Pre-compute total output per neuron
pre_total = strong.groupby('pre_pt_root_id')['syn_count'].sum()

# Build adjacency
adjacency = defaultdict(list)
for _, row in strong.iterrows():
    pre = row['pre_pt_root_id']
    post = row['post_pt_root_id']
    weight = row['syn_count']
    sign = row['nt_sign']
    total = pre_total.get(pre, weight)
    norm_weight = (weight / total) * sign
    adjacency[pre].append((post, norm_weight))

print(f"  Network: {len(adjacency):,} source neurons")

# =====================================================================
# 4. SPREADING ACTIVATION MODEL
# =====================================================================

def simulate(seed_ids, steps=12, decay=0.3, gain=2.0, threshold=0.1,
             remove_gaba=False, boost_ids=None, boost_factor=2.0,
             silence_ids=None):
    """Run spreading activation and return activation history."""
    activation = defaultdict(float)
    for nid in seed_ids:
        activation[nid] = 1.0

    history = []

    for t in range(steps):
        new_activation = defaultdict(float)

        for nid, act in activation.items():
            if act < threshold:
                continue
            if nid in adjacency:
                for post, w in adjacency[nid]:
                    if remove_gaba and w < 0:
                        continue
                    if silence_ids and nid in silence_ids:
                        continue
                    effective_w = w
                    if boost_ids and post in boost_ids:
                        effective_w = abs(w) * boost_factor
                    new_activation[post] += gain * effective_w * act

        for nid in activation:
            new_activation[nid] += decay * activation[nid]

        # Clamp to [0, 1]
        activation = defaultdict(float)
        for nid, val in new_activation.items():
            activation[nid] = max(0.0, min(1.0, val))

        # Record group activations
        def count_active(ids):
            return sum(1 for nid in ids if activation.get(nid, 0) >= threshold)

        def mean_act(ids):
            vals = [activation.get(nid, 0) for nid in ids]
            return np.mean(vals) if vals else 0

        record = {
            'step': t + 1,
            'ORN': count_active(orn_ids),
            'PN': count_active(pn_ids),
            'KC': count_active(kc_ids),
            'MBON_approach': count_active(approach_mbon_ids),
            'MBON_avoidance': count_active(avoidance_mbon_ids),
            'MBON_suppress': count_active(suppress_mbon_ids),
            'PAM': count_active(pam_ids),
            'PPL1': count_active(ppl1_ids),
            'LH': count_active(lh_ids),
            'CX': count_active(cx_ids),
            'FB': count_active(fb_ids),
            'Descending': count_active(desc_ids),
            'Motor': count_active(motor_ids),
            'approach_mean': mean_act(approach_mbon_ids),
            'avoidance_mean': mean_act(avoidance_mbon_ids),
            'suppress_mean': mean_act(suppress_mbon_ids),
        }
        history.append(record)

    return history, activation


def print_propagation(history, label=""):
    print(f"\n{'─'*90}")
    print(f"  {label}")
    print(f"{'─'*90}")
    print(f"  {'Step':<6} {'ORN':>5} {'PN':>5} {'KC':>6} {'App':>5} {'Avd':>5} "
          f"{'Sup':>5} {'PAM':>5} {'PPL1':>5} {'LH':>5} {'CX':>5} {'Desc':>5} {'Mot':>5}")
    print(f"  {'─'*84}")
    for r in history:
        print(f"  t+{r['step']:<3} {r['ORN']:>5} {r['PN']:>5} {r['KC']:>6} "
              f"{r['MBON_approach']:>5} {r['MBON_avoidance']:>5} {r['MBON_suppress']:>5} "
              f"{r['PAM']:>5} {r['PPL1']:>5} {r['LH']:>5} {r['CX']:>5} "
              f"{r['Descending']:>5} {r['Motor']:>5}")

# =====================================================================
# 5. EXPERIMENT 1: NEUTRAL ODOR — BASELINE
# =====================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 1: NEUTRAL ODOR (all ORNs)")
print("=" * 80)

hist_neutral, act_neutral = simulate(orn_ids, steps=12)
print_propagation(hist_neutral, "Neutral odor — full ORN activation")

# Valence score at each step
print("\n  Valence dynamics (approach - avoidance mean activation):")
for r in hist_neutral:
    valence = r['approach_mean'] - r['avoidance_mean']
    bar = '█' * int(abs(valence) * 50)
    sign = '+' if valence >= 0 else '-'
    print(f"  t+{r['step']}: {sign}{abs(valence):.3f}  {'→ APPROACH' if valence > 0 else '→ AVOID' if valence < 0 else '→ NEUTRAL':>12}  {bar}")

# =====================================================================
# 6. EXPERIMENT 2: APPROACH vs AVOIDANCE BIAS
# =====================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 2: PPL1 BOOST (danger context) vs PAM BOOST (reward context)")
print("=" * 80)

# Danger context: PPL1 feedback boosted (= punishment signal active)
hist_danger, _ = simulate(orn_ids, steps=12, boost_ids=ppl1_ids, boost_factor=3.0)
print_propagation(hist_danger, "DANGER context — PPL1 boosted 3x")

# Reward context: PAM feedback boosted
hist_reward, _ = simulate(orn_ids, steps=12, boost_ids=pam_ids, boost_factor=3.0)
print_propagation(hist_reward, "REWARD context — PAM boosted 3x")

print("\n  Valence comparison at each step:")
print(f"  {'Step':<6} {'Neutral':>10} {'Danger':>10} {'Reward':>10}")
print(f"  {'─'*40}")
for n, d, r in zip(hist_neutral, hist_danger, hist_reward):
    vn = n['approach_mean'] - n['avoidance_mean']
    vd = d['approach_mean'] - d['avoidance_mean']
    vr = r['approach_mean'] - r['avoidance_mean']
    print(f"  t+{n['step']:<3} {vn:>+10.4f} {vd:>+10.4f} {vr:>+10.4f}")

# =====================================================================
# 7. EXPERIMENT 3: GABA REMOVAL — DECISION WITHOUT BRAKES
# =====================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 3: GABA REMOVAL — WHAT HAPPENS WITHOUT INHIBITION?")
print("=" * 80)

hist_nogaba, _ = simulate(orn_ids, steps=12, remove_gaba=True)
print_propagation(hist_nogaba, "No GABA — excitation only")

print("\n  GABA effect on MBON activation:")
print(f"  {'':>20} {'With GABA':>12} {'No GABA':>12} {'Change':>10}")
for i in [3, 5, 7, 9, 11]:
    if i < len(hist_neutral) and i < len(hist_nogaba):
        n_app = hist_neutral[i]['MBON_approach']
        n_avd = hist_neutral[i]['MBON_avoidance']
        g_app = hist_nogaba[i]['MBON_approach']
        g_avd = hist_nogaba[i]['MBON_avoidance']
        print(f"  t+{i+1} Approach    {n_app:>10}/52  {g_app:>10}/52  {g_app-n_app:>+10}")
        print(f"  t+{i+1} Avoidance   {n_avd:>10}/25  {g_avd:>10}/25  {g_avd-n_avd:>+10}")

# =====================================================================
# 8. EXPERIMENT 4: LEARNED vs INNATE PATHWAY RACE
# =====================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 4: LEARNED (MB) vs INNATE (LH) PATHWAY TO DESCENDING NEURONS")
print("=" * 80)

# Track first activation of descending neurons via different routes
# MB pathway: ORN → PN → KC → MBON → desc
# LH pathway: ORN → PN → LH → desc

# Find which descending neurons receive from MBONs vs LH
mbon_to_desc = conn[(conn['pre_pt_root_id'].isin(mbon_ids)) &
                     (conn['post_pt_root_id'].isin(desc_ids))]
lh_to_desc = conn[(conn['pre_pt_root_id'].isin(lh_ids)) &
                   (conn['post_pt_root_id'].isin(desc_ids))]

desc_from_mbon = set(mbon_to_desc['post_pt_root_id'])
desc_from_lh = set(lh_to_desc['post_pt_root_id'])
desc_from_both = desc_from_mbon & desc_from_lh

print(f"\n  Descending neurons receiving from:")
print(f"    MB pathway (MBON): {len(desc_from_mbon)} neurons ({mbon_to_desc['syn_count'].sum():,} synapses)")
print(f"    LH pathway (innate): {len(desc_from_lh)} neurons ({lh_to_desc['syn_count'].sum():,} synapses)")
print(f"    Both pathways: {len(desc_from_both)} neurons (convergence)")
print(f"    MB only: {len(desc_from_mbon - desc_from_lh)} neurons")
print(f"    LH only: {len(desc_from_lh - desc_from_mbon)} neurons")

# When do LH vs MBON neurons activate?
print(f"\n  Activation timing:")
print(f"  {'Step':<6} {'LH':>8} {'MBON_app':>10} {'MBON_avd':>10} {'Desc':>8}")
print(f"  {'─'*45}")
for r in hist_neutral:
    print(f"  t+{r['step']:<3} {r['LH']:>7}/{len(lh_ids)} "
          f"{r['MBON_approach']:>9}/52 {r['MBON_avoidance']:>9}/25 "
          f"{r['Descending']:>7}/{len(desc_ids)}")

# =====================================================================
# 9. EXPERIMENT 5: MBON→DAN FEEDBACK LOOP ANALYSIS
# =====================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 5: MBON → DAN FEEDBACK LOOP")
print("=" * 80)

# Which MBONs feed back to PAM vs PPL1?
mbon_to_pam = conn[(conn['pre_pt_root_id'].isin(mbon_ids)) &
                    (conn['post_pt_root_id'].isin(pam_ids))]
mbon_to_ppl1 = conn[(conn['pre_pt_root_id'].isin(mbon_ids)) &
                     (conn['post_pt_root_id'].isin(ppl1_ids))]

print(f"\n  MBON → PAM feedback: {mbon_to_pam['syn_count'].sum():,} synapses")
print(f"  MBON → PPL1 feedback: {mbon_to_ppl1['syn_count'].sum():,} synapses")
print(f"  Ratio (PAM/PPL1): {mbon_to_pam['syn_count'].sum() / mbon_to_ppl1['syn_count'].sum():.1f}x")

# By MBON type
print(f"\n  Feedback by MBON type:")
print(f"  {'MBON type':<20} {'→ PAM':>8} {'→ PPL1':>8} {'Bias':>12}")
print(f"  {'─'*50}")

for nt_type, mbon_set, label in [
    ('acetylcholine', approach_mbon_ids, 'Approach (ACh)'),
    ('glutamate', avoidance_mbon_ids, 'Avoidance (Glut)'),
    ('gaba', suppress_mbon_ids, 'Suppress (GABA)')
]:
    to_pam = mbon_to_pam[mbon_to_pam['pre_pt_root_id'].isin(mbon_set)]['syn_count'].sum()
    to_ppl1 = mbon_to_ppl1[mbon_to_ppl1['pre_pt_root_id'].isin(mbon_set)]['syn_count'].sum()
    if to_pam + to_ppl1 > 0:
        bias = "→ PAM" if to_pam > to_ppl1 else "→ PPL1"
        print(f"  {label:<20} {to_pam:>8,} {to_ppl1:>8,} {bias:>12}")

# =====================================================================
# 10. EXPERIMENT 6: DECISION COMPETITION — SILENCE ONE PATHWAY
# =====================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 6: DECISION COMPETITION — SILENCE APPROACH vs AVOIDANCE MBONs")
print("=" * 80)

# Silence approach MBONs
hist_no_approach, _ = simulate(orn_ids, steps=12, silence_ids=approach_mbon_ids)
# Silence avoidance MBONs
hist_no_avoidance, _ = simulate(orn_ids, steps=12, silence_ids=avoidance_mbon_ids)

print(f"\n  Downstream effect of silencing:")
print(f"  {'Step':<6} {'Normal Desc':>12} {'No Approach':>12} {'No Avoidance':>13}")
print(f"  {'─'*45}")
for n, a, v in zip(hist_neutral, hist_no_approach, hist_no_avoidance):
    print(f"  t+{n['step']:<3} {n['Descending']:>12} {a['Descending']:>12} {v['Descending']:>13}")

print(f"\n  PAM/PPL1 effect:")
print(f"  {'Step':<6} {'Normal':>18} {'No Approach':>18} {'No Avoidance':>18}")
print(f"  {'':>6} {'PAM':>8} {'PPL1':>8} {'PAM':>8} {'PPL1':>8} {'PAM':>8} {'PPL1':>8}")
print(f"  {'─'*60}")
for n, a, v in zip(hist_neutral, hist_no_approach, hist_no_avoidance):
    print(f"  t+{n['step']:<3} {n['PAM']:>8} {n['PPL1']:>8} "
          f"{a['PAM']:>8} {a['PPL1']:>8} "
          f"{v['PAM']:>8} {v['PPL1']:>8}")

# =====================================================================
# 11. SUMMARY
# =====================================================================
print("\n" + "=" * 80)
print("SUMMARY: DECISION-MAKING ARCHITECTURE")
print("=" * 80)

# Compute key metrics
neutral_final = hist_neutral[-1]
danger_final = hist_danger[-1]
reward_final = hist_reward[-1]

v_neutral = neutral_final['approach_mean'] - neutral_final['avoidance_mean']
v_danger = danger_final['approach_mean'] - danger_final['avoidance_mean']
v_reward = reward_final['approach_mean'] - reward_final['avoidance_mean']

print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │  DECISION-MAKING CIRCUIT METRICS                            │
  ├──────────────────────────────────────────────────────────────┤
  │                                                              │
  │  VALENCE COMPUTATION (approach - avoidance):                │
  │    Neutral context:  {v_neutral:>+.4f}                              │
  │    Danger context:   {v_danger:>+.4f}  (PPL1 boosted)              │
  │    Reward context:   {v_reward:>+.4f}  (PAM boosted)               │
  │                                                              │
  │  PATHWAY ARCHITECTURE:                                      │
  │    MB pathway (learned):  ORN→PN→KC→MBON→Desc               │
  │    LH pathway (innate):   ORN→PN→LH→Desc                   │
  │    Convergence: {len(desc_from_both)} descending neurons receive both      │
  │                                                              │
  │  FEEDBACK LOOPS:                                            │
  │    MBON → PAM:  {mbon_to_pam['syn_count'].sum():>6,} synapses (reward feedback)     │
  │    MBON → PPL1: {mbon_to_ppl1['syn_count'].sum():>6,} synapses (punishment feedback) │
  │    Ratio: {mbon_to_pam['syn_count'].sum() / mbon_to_ppl1['syn_count'].sum():.1f}x more feedback to reward system          │
  │                                                              │
  │  MBON COMPOSITION:                                          │
  │    Approach (ACh):    52/96  (54%)                           │
  │    Avoidance (Glut):  25/96  (26%)                          │
  │    Suppression (GABA):19/96  (20%)                          │
  └──────────────────────────────────────────────────────────────┘
""")

# =====================================================================
# 12. VISUALIZATION
# =====================================================================
print("Generating visualization...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Analysis 19: Decision-Making Circuit in the Drosophila Brain',
             fontsize=14, fontweight='bold')

# Plot 1: Propagation timeline
ax = axes[0, 0]
steps = [r['step'] for r in hist_neutral]
ax.plot(steps, [r['MBON_approach'] for r in hist_neutral], 'g-o', label='Approach MBON', linewidth=2)
ax.plot(steps, [r['MBON_avoidance'] for r in hist_neutral], 'r-s', label='Avoidance MBON', linewidth=2)
ax.plot(steps, [r['MBON_suppress'] for r in hist_neutral], 'b-^', label='Suppress MBON', linewidth=2)
ax.set_xlabel('Time step')
ax.set_ylabel('Active neurons')
ax.set_title('MBON Activation Timeline')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: Valence dynamics across contexts
ax = axes[0, 1]
v_n = [r['approach_mean'] - r['avoidance_mean'] for r in hist_neutral]
v_d = [r['approach_mean'] - r['avoidance_mean'] for r in hist_danger]
v_r = [r['approach_mean'] - r['avoidance_mean'] for r in hist_reward]
ax.plot(steps, v_n, 'k-o', label='Neutral', linewidth=2)
ax.plot(steps, v_d, 'r-s', label='Danger (PPL1 boost)', linewidth=2)
ax.plot(steps, v_r, 'g-^', label='Reward (PAM boost)', linewidth=2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Time step')
ax.set_ylabel('Valence (approach - avoidance)')
ax.set_title('Valence Dynamics by Context')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 3: GABA effect
ax = axes[0, 2]
app_normal = [r['MBON_approach'] for r in hist_neutral]
app_nogaba = [r['MBON_approach'] for r in hist_nogaba]
avd_normal = [r['MBON_avoidance'] for r in hist_neutral]
avd_nogaba = [r['MBON_avoidance'] for r in hist_nogaba]
ax.plot(steps, app_normal, 'g-o', label='Approach (normal)', linewidth=2)
ax.plot(steps, app_nogaba, 'g--o', label='Approach (no GABA)', linewidth=1, alpha=0.7)
ax.plot(steps, avd_normal, 'r-s', label='Avoidance (normal)', linewidth=2)
ax.plot(steps, avd_nogaba, 'r--s', label='Avoidance (no GABA)', linewidth=1, alpha=0.7)
ax.set_xlabel('Time step')
ax.set_ylabel('Active neurons')
ax.set_title('GABA Effect on Decision')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 4: Learned vs innate pathway
ax = axes[1, 0]
ax.plot(steps, [r['LH'] for r in hist_neutral], 'm-o', label='Lateral Horn (innate)', linewidth=2)
ax.plot(steps, [r['MBON_approach'] + r['MBON_avoidance'] for r in hist_neutral],
        'c-s', label='MBON total (learned)', linewidth=2)
ax.plot(steps, [r['Descending'] for r in hist_neutral], 'k-^', label='Descending (output)', linewidth=2)
ax.set_xlabel('Time step')
ax.set_ylabel('Active neurons')
ax.set_title('Learned (MB) vs Innate (LH) Pathway')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 5: Feedback loop
ax = axes[1, 1]
feedback_data = {'Approach→PAM': 0, 'Approach→PPL1': 0,
                 'Avoidance→PAM': 0, 'Avoidance→PPL1': 0,
                 'Suppress→PAM': 0, 'Suppress→PPL1': 0}
for mbon_set, label in [(approach_mbon_ids, 'Approach'), (avoidance_mbon_ids, 'Avoidance'),
                         (suppress_mbon_ids, 'Suppress')]:
    to_pam = mbon_to_pam[mbon_to_pam['pre_pt_root_id'].isin(mbon_set)]['syn_count'].sum()
    to_ppl1 = mbon_to_ppl1[mbon_to_ppl1['pre_pt_root_id'].isin(mbon_set)]['syn_count'].sum()
    feedback_data[f'{label}→PAM'] = to_pam
    feedback_data[f'{label}→PPL1'] = to_ppl1

colors = ['#2ecc71', '#2ecc71', '#e74c3c', '#e74c3c', '#3498db', '#3498db']
hatches = ['', '///', '', '///', '', '///']
bars = ax.bar(range(len(feedback_data)), list(feedback_data.values()), color=colors)
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)
ax.set_xticks(range(len(feedback_data)))
ax.set_xticklabels(list(feedback_data.keys()), rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Synapses')
ax.set_title('MBON → DAN Feedback')
ax.grid(True, alpha=0.3, axis='y')

# Plot 6: Silencing experiment
ax = axes[1, 2]
desc_normal = [r['Descending'] for r in hist_neutral]
desc_no_app = [r['Descending'] for r in hist_no_approach]
desc_no_avd = [r['Descending'] for r in hist_no_avoidance]
ax.plot(steps, desc_normal, 'k-o', label='Normal', linewidth=2)
ax.plot(steps, desc_no_app, 'g--s', label='Approach MBONs silenced', linewidth=2)
ax.plot(steps, desc_no_avd, 'r--^', label='Avoidance MBONs silenced', linewidth=2)
ax.set_xlabel('Time step')
ax.set_ylabel('Active descending neurons')
ax.set_title('Effect of Silencing MBON Types')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
results_dir = os.path.join(PROJECT_DIR, "results")
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, "19_decision_circuit.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
