"""
FlyWire Analysis 17 - Larval Drosophila Signal Propagation
Compare PPL1 temporal priority in larva vs adult connectome.
Data: Winding et al. (2023) Science — 3,016 neurons, 548K synapses.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/larva/Supplementary-Data-S1")

# =====================================================================
# 1. LOAD DATA
# =====================================================================
print("=" * 80)
print("LARVAL DROSOPHILA — SIGNAL PROPAGATION & DAN PRIORITY ANALYSIS")
print("=" * 80)

ann = pd.read_csv(f"{DATA_DIR}/annotations.csv")
conn_matrix = pd.read_csv(f"{DATA_DIR}/all-all_connectivity_matrix.csv", index_col=0)

# Convert column names to int for matching
conn_matrix.columns = conn_matrix.columns.astype(int)

print(f"\nConnectivity matrix: {conn_matrix.shape[0]} x {conn_matrix.shape[1]}")
print(f"Non-zero connections: {(conn_matrix > 0).sum().sum():,}")
print(f"Total synaptic weight: {conn_matrix.sum().sum():,.0f}")

# =====================================================================
# 2. BUILD NEURON ID MAPS
# =====================================================================

# Create mapping from neuron ID to cell type and annotations
neuron_info = {}
for _, row in ann.iterrows():
    for side_col in ['left_id', 'right_id']:
        nid = row[side_col]
        if str(nid) != 'no pair':
            nid = int(nid)
            neuron_info[nid] = {
                'celltype': row['celltype'],
                'annotation': row['additional_annotations'],
                'cluster': row['level_7_cluster']
            }

# Classify neurons by type
all_neuron_ids = set(conn_matrix.index)

def get_ids_by_celltype(ct):
    return set(nid for nid in all_neuron_ids if nid in neuron_info and neuron_info[nid]['celltype'] == ct)

def get_ids_by_annotation(ann_substr):
    return set(nid for nid in all_neuron_ids if nid in neuron_info and ann_substr in str(neuron_info[nid]['annotation']))

# Neuron groups
orn_ids = get_ids_by_annotation('olfactory') & get_ids_by_celltype('sensory')
gust_ids = (get_ids_by_annotation('gustatory-external') | get_ids_by_annotation('gustatory-pharyngeal')) & get_ids_by_celltype('sensory')
kc_ids = get_ids_by_celltype('KC')
mbon_ids = get_ids_by_celltype('MBON')
mbin_ids = get_ids_by_celltype('MBIN')
motor_ids = get_ids_by_celltype('DN-VNC') | get_ids_by_celltype('DN-SEZ')
pn_ids = get_ids_by_celltype('PN')
ln_ids = get_ids_by_celltype('LN')

# DAN subtypes (aversive vs appetitive)
# Based on larval literature: DAN-d1, DAN-f1, DAN-g1, DAN-c1 = aversive (PPL1-like)
# DAN-i1, DAN-j1, DAN-k1 = appetitive (PAM-like)
dan_aversive_ids = set()
dan_appetitive_ids = set()
oan_ids = set()
other_mbin_ids = set()

for nid, info in neuron_info.items():
    if nid not in all_neuron_ids:
        continue
    ann_str = str(info['annotation'])
    if info['celltype'] == 'MBIN':
        if ann_str.startswith('DAN-d') or ann_str.startswith('DAN-f') or ann_str.startswith('DAN-g') or ann_str.startswith('DAN-c'):
            dan_aversive_ids.add(nid)
        elif ann_str.startswith('DAN-i') or ann_str.startswith('DAN-j') or ann_str.startswith('DAN-k'):
            dan_appetitive_ids.add(nid)
        elif ann_str.startswith('OAN'):
            oan_ids.add(nid)
        else:
            other_mbin_ids.add(nid)

print(f"\n--- Neuron Groups ---")
print(f"  Olfactory sensory (ORN):    {len(orn_ids)}")
print(f"  Gustatory sensory:          {len(gust_ids)}")
print(f"  Projection neurons (PN):    {len(pn_ids)}")
print(f"  Local neurons (LN):         {len(ln_ids)}")
print(f"  Kenyon Cells (KC):          {len(kc_ids)}")
print(f"  MBONs:                      {len(mbon_ids)}")
print(f"  DAN aversive (PPL1-like):   {len(dan_aversive_ids)}")
print(f"  DAN appetitive (PAM-like):  {len(dan_appetitive_ids)}")
print(f"  OAN:                        {len(oan_ids)}")
print(f"  Other MBIN:                 {len(other_mbin_ids)}")
print(f"  Motor (DN):                 {len(motor_ids)}")

# =====================================================================
# 3. BUILD NETWORK
# =====================================================================
print(f"\n{'='*80}")
print("BUILDING NETWORK")
print(f"{'='*80}")

# Convert matrix to edge list for efficient propagation
max_weight = conn_matrix.max().max()
print(f"Max synapse count: {max_weight}")

# Build adjacency dict: target -> [(source, weight), ...]
adj_incoming = {}
for post_id in conn_matrix.columns:
    col = conn_matrix[post_id]
    sources = col[col > 0]
    if len(sources) > 0:
        adj_incoming[post_id] = [(pre_id, w / max_weight) for pre_id, w in sources.items()]

print(f"Neurons with incoming connections: {len(adj_incoming)}")

# =====================================================================
# 4. SPREADING ACTIVATION — OLFACTORY
# =====================================================================
print(f"\n{'='*80}")
print("SIMULATION 1: OLFACTORY SIGNAL PROPAGATION")
print(f"{'='*80}")

def run_simulation(start_ids, n_steps=12, decay=0.3, gain=2.0, threshold=0.1):
    """Run spreading activation simulation."""
    activation = {nid: 0.0 for nid in all_neuron_ids}

    # Initialize starting neurons
    for nid in start_ids:
        if nid in activation:
            activation[nid] = 1.0

    history = []

    for t in range(n_steps):
        # Track populations
        step_data = {
            'step': t,
            'ORN': sum(1 for nid in orn_ids if activation.get(nid, 0) > threshold),
            'PN': sum(1 for nid in pn_ids if activation.get(nid, 0) > threshold),
            'LN': sum(1 for nid in ln_ids if activation.get(nid, 0) > threshold),
            'KC': sum(1 for nid in kc_ids if activation.get(nid, 0) > threshold),
            'MBON': sum(1 for nid in mbon_ids if activation.get(nid, 0) > threshold),
            'DAN_aversive': sum(1 for nid in dan_aversive_ids if activation.get(nid, 0) > threshold),
            'DAN_appetitive': sum(1 for nid in dan_appetitive_ids if activation.get(nid, 0) > threshold),
            'Motor': sum(1 for nid in motor_ids if activation.get(nid, 0) > threshold),
            'total': sum(1 for nid in all_neuron_ids if activation.get(nid, 0) > threshold),
        }
        history.append(step_data)

        # Propagate
        new_activation = {}
        for nid in all_neuron_ids:
            incoming = 0.0
            if nid in adj_incoming:
                for src, w in adj_incoming[nid]:
                    if activation.get(src, 0) > 0:
                        incoming += w * activation[src]

            new_act = decay * activation.get(nid, 0) + gain * incoming
            new_activation[nid] = max(0.0, min(1.0, new_act))

        activation = new_activation

    return history

# Run olfactory simulation
olf_history = run_simulation(orn_ids, n_steps=12)

print(f"\n{'Step':<6} {'ORN':<6} {'PN':<6} {'LN':<6} {'KC':<6} {'MBON':<7} {'DAN_av':<8} {'DAN_ap':<8} {'Motor':<7} {'Total':<7}")
print("-" * 75)
for h in olf_history:
    print(f"t+{h['step']:<4} {h['ORN']:<6} {h['PN']:<6} {h['LN']:<6} {h['KC']:<6} {h['MBON']:<7} {h['DAN_aversive']:<8} {h['DAN_appetitive']:<8} {h['Motor']:<7} {h['total']:<7}")

# Find first activation
dan_av_first = next((h['step'] for h in olf_history if h['DAN_aversive'] > 0), None)
dan_ap_first = next((h['step'] for h in olf_history if h['DAN_appetitive'] > 0), None)
motor_first = next((h['step'] for h in olf_history if h['Motor'] > 0), None)

print(f"\n--- First Activation Times (Olfactory) ---")
print(f"  DAN aversive (PPL1-like):  t+{dan_av_first}" if dan_av_first is not None else "  DAN aversive: never")
print(f"  DAN appetitive (PAM-like): t+{dan_ap_first}" if dan_ap_first is not None else "  DAN appetitive: never")
print(f"  Motor neurons:             t+{motor_first}" if motor_first is not None else "  Motor: never")

if dan_av_first is not None and dan_ap_first is not None:
    diff = dan_ap_first - dan_av_first
    if diff > 0:
        print(f"\n  >>> DAN AVERSIVE {diff} STEP(S) BEFORE APPETITIVE — CONSISTENT WITH ADULT! <<<")
    elif diff == 0:
        print(f"\n  >>> SIMULTANEOUS ACTIVATION <<<")
    else:
        print(f"\n  >>> DAN APPETITIVE FIRST — DIFFERENT FROM ADULT <<<")

# =====================================================================
# 5. SPREADING ACTIVATION — GUSTATORY
# =====================================================================
print(f"\n{'='*80}")
print("SIMULATION 2: GUSTATORY SIGNAL PROPAGATION")
print(f"{'='*80}")

gust_history = run_simulation(gust_ids, n_steps=12)

print(f"\n{'Step':<6} {'Gust':<6} {'PN':<6} {'LN':<6} {'KC':<6} {'MBON':<7} {'DAN_av':<8} {'DAN_ap':<8} {'Motor':<7} {'Total':<7}")
print("-" * 75)
for h in gust_history:
    gust_active = sum(1 for nid in gust_ids if nid in all_neuron_ids)  # approximate
    print(f"t+{h['step']:<4} {h['ORN']:<6} {h['PN']:<6} {h['LN']:<6} {h['KC']:<6} {h['MBON']:<7} {h['DAN_aversive']:<8} {h['DAN_appetitive']:<8} {h['Motor']:<7} {h['total']:<7}")

gust_av_first = next((h['step'] for h in gust_history if h['DAN_aversive'] > 0), None)
gust_ap_first = next((h['step'] for h in gust_history if h['DAN_appetitive'] > 0), None)
gust_motor_first = next((h['step'] for h in gust_history if h['Motor'] > 0), None)

print(f"\n--- First Activation Times (Gustatory) ---")
print(f"  DAN aversive (PPL1-like):  t+{gust_av_first}" if gust_av_first is not None else "  DAN aversive: never")
print(f"  DAN appetitive (PAM-like): t+{gust_ap_first}" if gust_ap_first is not None else "  DAN appetitive: never")
print(f"  Motor neurons:             t+{gust_motor_first}" if gust_motor_first is not None else "  Motor: never")

# =====================================================================
# 6. PARAMETER SENSITIVITY — LARVA
# =====================================================================
print(f"\n{'='*80}")
print("PARAMETER SENSITIVITY — 50 COMBINATIONS")
print(f"{'='*80}")

results = {'aversive_first': 0, 'appetitive_first': 0, 'tie': 0, 'neither': 0}
detail_results = []

for decay in [0.1, 0.3, 0.5]:
    for gain_val in [1.0, 2.0, 3.0, 5.0]:
        for thresh in [0.05, 0.1, 0.2, 0.3]:
            hist = run_simulation(orn_ids, n_steps=15, decay=decay, gain=gain_val, threshold=thresh)
            av_first = next((h['step'] for h in hist if h['DAN_aversive'] > 0), None)
            ap_first = next((h['step'] for h in hist if h['DAN_appetitive'] > 0), None)

            if av_first is None and ap_first is None:
                results['neither'] += 1
                cat = 'neither'
            elif av_first is not None and ap_first is None:
                results['aversive_first'] += 1
                cat = 'aversive_only'
            elif av_first is None and ap_first is not None:
                results['appetitive_first'] += 1
                cat = 'appetitive_only'
            elif av_first < ap_first:
                results['aversive_first'] += 1
                cat = 'aversive_first'
            elif av_first > ap_first:
                results['appetitive_first'] += 1
                cat = 'appetitive_first'
            else:
                results['tie'] += 1
                cat = 'tie'

            detail_results.append({
                'decay': decay, 'gain': gain_val, 'threshold': thresh,
                'av_first': av_first, 'ap_first': ap_first, 'category': cat
            })

valid = results['aversive_first'] + results['appetitive_first'] + results['tie']
print(f"\n  Total combinations tested: {len(detail_results)}")
print(f"  Valid (at least one activated): {valid}")
print(f"  Neither activated: {results['neither']}")
print(f"\n  ╔══════════════════════════════════════════════╗")
print(f"  ║  LARVA RESULTS ({valid} valid):{'':>19}║")
print(f"  ║                                              ║")
if valid > 0:
    print(f"  ║  Aversive first:  {results['aversive_first']:>3}/{valid}  ({results['aversive_first']/valid*100:>5.1f}%){'':>12}║")
    print(f"  ║  Appetitive first:{results['appetitive_first']:>3}/{valid}  ({results['appetitive_first']/valid*100:>5.1f}%){'':>12}║")
    print(f"  ║  Tie:             {results['tie']:>3}/{valid}  ({results['tie']/valid*100:>5.1f}%){'':>12}║")
print(f"  ╚══════════════════════════════════════════════╝")

# =====================================================================
# 7. STRUCTURAL ANALYSIS — INPUT COMPARISON
# =====================================================================
print(f"\n{'='*80}")
print("STRUCTURAL ANALYSIS: DAN INPUT COMPARISON")
print(f"{'='*80}")

def get_total_input(neuron_ids):
    total = 0
    for nid in neuron_ids:
        if nid in conn_matrix.columns:
            total += conn_matrix[nid].sum()
    return total

av_input = get_total_input(dan_aversive_ids)
ap_input = get_total_input(dan_appetitive_ids)
av_count = len(dan_aversive_ids)
ap_count = len(dan_appetitive_ids)

print(f"\n  DAN aversive (PPL1-like):")
print(f"    Neurons:        {av_count}")
print(f"    Total input:    {av_input:,.0f} synapses")
print(f"    Per neuron:     {av_input/av_count:,.0f} synapses/neuron" if av_count > 0 else "")

print(f"\n  DAN appetitive (PAM-like):")
print(f"    Neurons:        {ap_count}")
print(f"    Total input:    {ap_input:,.0f} synapses")
print(f"    Per neuron:     {ap_input/ap_count:,.0f} synapses/neuron" if ap_count > 0 else "")

if av_count > 0 and ap_count > 0:
    ratio = (av_input/av_count) / (ap_input/ap_count)
    print(f"\n  Input ratio (aversive/appetitive per neuron): {ratio:.1f}x")

# Input by source type
print(f"\n--- Input Sources ---")
for label, target_ids in [("DAN aversive", dan_aversive_ids), ("DAN appetitive", dan_appetitive_ids)]:
    print(f"\n  {label}:")
    source_types = {}
    for nid in target_ids:
        if nid in conn_matrix.columns:
            col = conn_matrix[nid]
            for src, w in col[col > 0].items():
                if src in neuron_info:
                    ct = neuron_info[src]['celltype']
                else:
                    ct = 'unknown'
                source_types[ct] = source_types.get(ct, 0) + w

    total_in = sum(source_types.values())
    for ct, w in sorted(source_types.items(), key=lambda x: -x[1])[:8]:
        print(f"    {ct:<20} {w:>6.0f} syn  ({w/total_in*100:>5.1f}%)")

# =====================================================================
# 8. COMPARISON TABLE: ADULT vs LARVA
# =====================================================================
print(f"\n{'='*80}")
print("ADULT vs LARVA COMPARISON")
print(f"{'='*80}")

print(f"""
  ┌─────────────────────┬──────────────────┬──────────────────┐
  │                     │  ADULT (FlyWire)  │  LARVA (Winding) │
  ├─────────────────────┼──────────────────┼──────────────────┤
  │ Total neurons       │     139,255      │      3,016       │
  │ Total synapses      │     50M+         │      548K        │
  │ Kenyon Cells        │      5,177       │        121       │
  │ MBONs               │         96       │         24       │
  │ Aversive DAN        │   16 (PPL1)      │  {av_count:>3} (DAN-c/d/f/g)│
  │ Appetitive DAN      │  307 (PAM)       │  {ap_count:>3} (DAN-i/j/k)  │
  │ Aversive/Appetitive │    1:19          │  {av_count}:{ap_count}{'':>14}│
  │ Olf → Aversive      │     t+4          │  t+{dan_av_first if dan_av_first else '?'}{'':>13}│
  │ Olf → Appetitive    │     t+7          │  t+{dan_ap_first if dan_ap_first else '?'}{'':>13}│
  │ Input/neuron (aver) │    4,804         │  {av_input/av_count:>6,.0f}{'':>10}│
  │ Input/neuron (appet)│      414         │  {ap_input/ap_count:>6,.0f}{'':>10}│
  └─────────────────────┴──────────────────┴──────────────────┘
""")

# =====================================================================
# 9. VISUALIZATION
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Olfactory propagation timeline
steps = [h['step'] for h in olf_history]
axes[0].plot(steps, [h['DAN_aversive'] for h in olf_history], 'r-o', linewidth=2, label=f'DAN aversive (n={av_count})', markersize=6)
axes[0].plot(steps, [h['DAN_appetitive'] for h in olf_history], 'g-s', linewidth=2, label=f'DAN appetitive (n={ap_count})', markersize=6)
axes[0].plot(steps, [h['KC'] for h in olf_history], 'b-^', linewidth=1.5, alpha=0.5, label=f'KC (n={len(kc_ids)})', markersize=5)
axes[0].plot(steps, [h['Motor'] for h in olf_history], 'k--', linewidth=1.5, alpha=0.5, label=f'Motor (n={len(motor_ids)})', markersize=4)
axes[0].set_xlabel('Time Step', fontsize=11)
axes[0].set_ylabel('Active Neurons', fontsize=11)
axes[0].set_title('Larva: Olfactory Signal Propagation', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

# Plot 2: Parameter sensitivity
categories = ['Aversive\nFirst', 'Appetitive\nFirst', 'Tie']
counts = [results['aversive_first'], results['appetitive_first'], results['tie']]
colors = ['#e74c3c', '#2ecc71', '#95a5a6']
bars = axes[1].bar(categories, counts, color=colors, edgecolor='white', linewidth=2)
axes[1].set_ylabel('Number of Combinations', fontsize=11)
axes[1].set_title(f'Larva: Parameter Sensitivity ({valid} valid)', fontsize=13, fontweight='bold')
for bar, count in zip(bars, counts):
    if count > 0:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{count}\n({count/valid*100:.0f}%)', ha='center', fontsize=11, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

# Plot 3: Input comparison (adult vs larva)
x = np.arange(2)
width = 0.35
adult_vals = [4804, 414]
larva_vals = [av_input/av_count if av_count > 0 else 0, ap_input/ap_count if ap_count > 0 else 0]

bars1 = axes[2].bar(x - width/2, adult_vals, width, label='Adult', color=['#c0392b', '#27ae60'], alpha=0.7)
bars2 = axes[2].bar(x + width/2, larva_vals, width, label='Larva', color=['#e74c3c', '#2ecc71'], alpha=0.7)
axes[2].set_xticks(x)
axes[2].set_xticklabels(['Aversive DAN\n(PPL1-like)', 'Appetitive DAN\n(PAM-like)'], fontsize=10)
axes[2].set_ylabel('Input Synapses per Neuron', fontsize=11)
axes[2].set_title('Input Density: Adult vs Larva', fontsize=13, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            axes[2].text(bar.get_x() + bar.get_width()/2, h + 50, f'{h:,.0f}', ha='center', fontsize=9)

plt.tight_layout()
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
out_path = os.path.join(RESULTS_DIR, "17_larva_propagation.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
