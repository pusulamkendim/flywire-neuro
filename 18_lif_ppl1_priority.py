"""
FlyWire Analysis 18 — PPL1 Priority Test on Biophysical LIF Model

Tests whether PPL1 (punishment) neurons fire before PAM (reward) neurons
using Eon Systems' leaky integrate-and-fire model of the adult Drosophila brain.

This is a biophysical validation of our spreading-activation finding (Analysis 15).

Model: Shiu et al. — LIF neurons, alpha-function synapses, real connectome weights.
Data: FlyWire v783 (138,639 neurons, ~5M synapses).
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from time import time
from collections import defaultdict

# Add fly-brain code to path
PROJECT_DIR = Path(__file__).resolve().parent
FLY_BRAIN_DIR = PROJECT_DIR / 'fly-brain'
sys.path.insert(0, str(FLY_BRAIN_DIR / 'code'))

# Override paths from benchmark to point to fly-brain/data
import benchmark
benchmark.path_comp = (FLY_BRAIN_DIR / 'data/2025_Completeness_783.csv').resolve()
benchmark.path_con = (FLY_BRAIN_DIR / 'data/2025_Connectivity_783.parquet').resolve()
benchmark.path_res = (FLY_BRAIN_DIR / 'data/results').resolve()
benchmark.path_wt = (FLY_BRAIN_DIR / 'data').resolve()

import torch
from run_pytorch import (
    get_hash_tables, get_weights, TorchModel, MODEL_PARAMS, DT,
)

# ============================================================================
# Load neuron annotations from our project
# ============================================================================
print("=" * 80)
print("ANALYSIS 18: PPL1 PRIORITY — BIOPHYSICAL LIF VALIDATION")
print("=" * 80)

ann = pd.read_csv(PROJECT_DIR / 'data/neuron_annotations.tsv', sep='\t', low_memory=False)
comp = pd.read_csv(benchmark.path_comp, index_col=0)
comp_ids = set(comp.index)

# Neuron groups
orn_ids = [x for x in ann[ann['cell_class'] == 'olfactory']['root_id'] if x in comp_ids]
ppl1_ids = [x for x in ann[ann['cell_type'].str.startswith('PPL1', na=False)]['root_id'] if x in comp_ids]
pam_ids = [x for x in ann[ann['cell_type'].str.startswith('PAM', na=False)]['root_id'] if x in comp_ids]
mbon_ids = [x for x in ann[ann['cell_class'] == 'MBON']['root_id'] if x in comp_ids]
kc_ids = [x for x in ann[ann['cell_class'] == 'Kenyon_Cell']['root_id'] if x in comp_ids]
motor_ids = [x for x in ann[ann['super_class'] == 'motor']['root_id'] if x in comp_ids]

print(f"\nNeuron groups:")
print(f"  ORN (olfactory):  {len(orn_ids):>6}")
print(f"  PPL1 (punishment):{len(ppl1_ids):>6}")
print(f"  PAM (reward):     {len(pam_ids):>6}")
print(f"  MBON:             {len(mbon_ids):>6}")
print(f"  Kenyon Cells:     {len(kc_ids):>6}")
print(f"  Motor:            {len(motor_ids):>6}")

# ============================================================================
# Build ID mappings
# ============================================================================
print("\nBuilding ID mappings...")
flyid2i, i2flyid = get_hash_tables(str(benchmark.path_comp))

orn_indices = [flyid2i[n] for n in orn_ids if n in flyid2i]
ppl1_indices = [flyid2i[n] for n in ppl1_ids if n in flyid2i]
pam_indices = [flyid2i[n] for n in pam_ids if n in flyid2i]
mbon_indices = [flyid2i[n] for n in mbon_ids if n in flyid2i]
kc_indices = [flyid2i[n] for n in kc_ids if n in flyid2i]
motor_indices = [flyid2i[n] for n in motor_ids if n in flyid2i]

print(f"  ORN indices:  {len(orn_indices)}")
print(f"  PPL1 indices: {len(ppl1_indices)}")
print(f"  PAM indices:  {len(pam_indices)}")

# ============================================================================
# Load weights and create model
# ============================================================================
print("\nLoading connectome weights (this may take a minute)...")
device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device_name.upper()}")

t0 = time()
weights = get_weights(
    str(benchmark.path_con), str(benchmark.path_comp),
    str(benchmark.path_wt), csr=True
)
weights = weights.to(device=device_name)
num_neurons = weights.shape[0]
print(f"  Loaded: {num_neurons:,} neurons — {time()-t0:.1f}s")

# ============================================================================
# EXPERIMENT: Stimulate ORNs, record first spike times of PPL1 vs PAM
# ============================================================================

def run_olfactory_experiment(stim_rate, t_run_sec, label=""):
    """Stimulate ORNs and record first spike time per neuron group."""
    t_sim_ms = t_run_sec * 1000.0
    num_steps = int(t_sim_ms / DT)
    n_run = 1

    model = TorchModel(n_run, num_neurons, DT, MODEL_PARAMS, weights, device=device_name)
    conductance, delay_buffer, spikes, v, refrac = model.state_init()

    # Stimulate ORNs
    rates = torch.zeros(n_run, num_neurons, device=device_name)
    rates[:, orn_indices] = stim_rate

    # Track first spike time per group
    groups = {
        'PPL1': (ppl1_indices, None),
        'PAM': (pam_indices, None),
        'MBON': (mbon_indices, None),
        'KC': (kc_indices, None),
        'Motor': (motor_indices, None),
    }

    # Track individual neuron spike counts
    ppl1_spike_counts = torch.zeros(len(ppl1_indices), device='cpu')
    pam_spike_counts = torch.zeros(len(pam_indices), device='cpu')

    # First spike time for each group
    first_spike = {name: None for name in groups}
    # Count of spiking neurons per group at each timestep
    group_activity = {name: [] for name in groups}

    t_start = time()
    with torch.no_grad():
        for t_step in range(num_steps):
            conductance, delay_buffer, spikes, v, refrac = model(
                rates, conductance, delay_buffer, spikes, v, refrac
            )

            for name, (indices, _) in groups.items():
                group_spikes = spikes[0, indices]
                n_spiking = (group_spikes > 0).sum().item()
                group_activity[name].append(n_spiking)

                if first_spike[name] is None and n_spiking > 0:
                    first_spike[name] = t_step * DT  # in ms

            # Track individual PPL1/PAM spikes
            ppl1_spike_counts += (spikes[0, ppl1_indices] > 0).cpu().float()
            pam_spike_counts += (spikes[0, pam_indices] > 0).cpu().float()

    elapsed = time() - t_start

    t_ms = t_run_sec * 1000
    print(f"\n{'─'*60}")
    print(f"  {label} | rate={stim_rate}Hz | t={t_run_sec}s | {elapsed:.1f}s wall")
    print(f"{'─'*60}")
    print(f"  {'Group':<10} {'First spike (ms)':>18} {'Total active':>14}")
    print(f"  {'─'*42}")
    for name in groups:
        total_active = sum(1 for x in group_activity[name] if x > 0)
        fs = f"{first_spike[name]:.1f}" if first_spike[name] is not None else "never"
        steps_with_activity = sum(1 for x in group_activity[name] if x > 0)
        print(f"  {name:<10} {fs:>18} {steps_with_activity:>14} steps")

    ppl1_t = first_spike['PPL1']
    pam_t = first_spike['PAM']

    if ppl1_t is not None and pam_t is not None:
        diff = pam_t - ppl1_t
        if diff > 0:
            print(f"\n  >>> PPL1 fires {diff:.1f}ms BEFORE PAM — CONSISTENT! <<<")
        elif diff < 0:
            print(f"\n  >>> PAM fires {-diff:.1f}ms BEFORE PPL1 — INCONSISTENT! <<<")
        else:
            print(f"\n  >>> TIE — both fire at {ppl1_t:.1f}ms <<<")
    elif ppl1_t is not None:
        print(f"\n  >>> Only PPL1 fired ({ppl1_t:.1f}ms) — PAM never spiked <<<")
    elif pam_t is not None:
        print(f"\n  >>> Only PAM fired ({pam_t:.1f}ms) — PPL1 never spiked <<<")
    else:
        print(f"\n  >>> Neither group spiked <<<")

    return {
        'stim_rate': stim_rate,
        't_run': t_run_sec,
        'first_spike': first_spike,
        'ppl1_active': int(ppl1_spike_counts.sum().item()),
        'pam_active': int(pam_spike_counts.sum().item()),
        'ppl1_neurons_fired': int((ppl1_spike_counts > 0).sum().item()),
        'pam_neurons_fired': int((pam_spike_counts > 0).sum().item()),
        'group_activity': group_activity,
    }


# ============================================================================
# Run experiments at multiple stimulation rates and durations
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 1: ORN stimulation at different rates")
print("=" * 80)

results = []
for rate in [50, 100, 200, 500]:
    r = run_olfactory_experiment(rate, t_run_sec=0.1, label=f"Rate {rate}Hz")
    results.append(r)

print("\n" + "=" * 80)
print("EXPERIMENT 2: Longer simulation at 200 Hz")
print("=" * 80)

r_long = run_olfactory_experiment(200, t_run_sec=1.0, label="Long run 200Hz")
results.append(r_long)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: PPL1 vs PAM FIRST SPIKE TIMING")
print("=" * 80)

print(f"\n  {'Rate':>6}  {'Duration':>10}  {'PPL1 (ms)':>12}  {'PAM (ms)':>12}  {'Delta':>10}  {'Result':>15}")
print(f"  {'─'*70}")

ppl1_wins = 0
pam_wins = 0
ties = 0

for r in results:
    ppl1_t = r['first_spike']['PPL1']
    pam_t = r['first_spike']['PAM']
    ppl1_s = f"{ppl1_t:.1f}" if ppl1_t is not None else "—"
    pam_s = f"{pam_t:.1f}" if pam_t is not None else "—"

    if ppl1_t is not None and pam_t is not None:
        delta = pam_t - ppl1_t
        delta_s = f"{delta:+.1f}"
        if delta > 0:
            result = "PPL1 FIRST"
            ppl1_wins += 1
        elif delta < 0:
            result = "PAM FIRST"
            pam_wins += 1
        else:
            result = "TIE"
            ties += 1
    elif ppl1_t is not None:
        delta_s = "—"
        result = "PPL1 only"
        ppl1_wins += 1
    elif pam_t is not None:
        delta_s = "—"
        result = "PAM only"
        pam_wins += 1
    else:
        delta_s = "—"
        result = "neither"

    print(f"  {r['stim_rate']:>5}Hz  {r['t_run']:>9.1f}s  {ppl1_s:>12}  {pam_s:>12}  {delta_s:>10}  {result:>15}")

total = ppl1_wins + pam_wins + ties
print(f"\n  ╔══════════════════════════════════════════════╗")
print(f"  ║  LIF MODEL RESULTS ({total} conditions):        ║")
print(f"  ║                                              ║")
print(f"  ║  PPL1 first:  {ppl1_wins:>2}/{total}  ({ppl1_wins/total*100 if total else 0:>5.1f}%)              ║")
print(f"  ║  PAM first:   {pam_wins:>2}/{total}  ({pam_wins/total*100 if total else 0:>5.1f}%)              ║")
print(f"  ║  Tie:         {ties:>2}/{total}  ({ties/total*100 if total else 0:>5.1f}%)              ║")
print(f"  ╚══════════════════════════════════════════════╝")

# ============================================================================
# Comparison with spreading activation model
# ============================================================================
print(f"\n{'='*80}")
print("CROSS-MODEL COMPARISON")
print(f"{'='*80}")
print(f"""
  ┌─────────────────────┬────────────────────┬────────────────────┐
  │                     │ Spreading Activ.   │ LIF Biophysical    │
  ├─────────────────────┼────────────────────┼────────────────────┤
  │ Neuron model        │ Binary threshold   │ Leaky Integrate &  │
  │                     │                    │ Fire (V_th=-45mV)  │
  │ Time resolution     │ Discrete steps     │ 0.1ms timesteps    │
  │ Synaptic weights    │ Normalized (0-1)   │ Connectome-derived │
  │                     │                    │ (0.275mV base)     │
  │ Inhibition (GABA)   │ Not modeled        │ Negative weights   │
  │ Refractory period   │ None               │ 2.2ms              │
  │ Synaptic delay      │ None               │ 1.8ms              │
  │ PPL1 before PAM?    │ YES (100%)         │ See results above  │
  └─────────────────────┴────────────────────┴────────────────────┘
""")

print(f"{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
