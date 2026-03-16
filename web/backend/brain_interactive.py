"""
Interactive brain simulation: 138,639 LIF neurons on CPU.

User toggles stimuli via WebSocket → brain responds in real-time.
Streams DN rates + population spike counts back to frontend.

Architecture:
  - Brain runs in background thread
  - Stimulus commands arrive via shared state (thread-safe)
  - Frames emitted to asyncio queue → WebSocket → browser
"""

import os
os.environ.setdefault('MUJOCO_GL', 'disabled')

import sys
import csv
import time
import asyncio
import threading
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
FLY_BRAIN_DIR = PROJECT_DIR / 'fly-brain-embodied'
CODE_DIR = FLY_BRAIN_DIR / 'code'
DATA_DIR = FLY_BRAIN_DIR / 'data'

for p in [str(CODE_DIR), str(FLY_BRAIN_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


class InteractiveBrain:
    """Runs 138K neuron brain, accepts stimulus changes in real-time."""

    def __init__(self):
        self.queue = None
        self._loop = None
        self.running = False
        self._thread = None

        # Thread-safe stimulus state
        self._lock = threading.Lock()
        self._active_stimuli = set()

    def start(self, loop, queue, initial_stimuli=None):
        if self.running:
            return
        self.queue = queue
        self._loop = loop
        self.running = True
        if initial_stimuli:
            self._active_stimuli = set(initial_stimuli)
        self._thread = threading.Thread(target=self._run_safe, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        self._save_log()

    def _save_log(self):
        if not hasattr(self, '_log_rows') or not self._log_rows:
            return
        log_dir = Path(__file__).resolve().parent / 'brain_logs'
        log_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = log_dir / f'brain_{ts}.csv'
        keys = list(self._log_rows[0].keys())
        with open(log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self._log_rows)
        print(f"[Brain] Log saved: {log_path} ({len(self._log_rows)} rows)", flush=True)

    def set_stimuli(self, stimuli_list):
        """Thread-safe update of active stimuli."""
        with self._lock:
            self._active_stimuli = set(stimuli_list)
        print(f"[Brain] Stimuli updated: {stimuli_list}", flush=True)

    def _emit(self, data):
        try:
            asyncio.run_coroutine_threadsafe(self.queue.put(data), self._loop).result(timeout=5)
        except Exception as e:
            print(f"[Brain] emit error: {e}", flush=True)

    def _run_safe(self):
        try:
            self._run()
        except Exception as e:
            import traceback
            print(f"[Brain] ERROR: {e}", flush=True)
            traceback.print_exc()
            self.running = False
            self._emit({"event": "end"})

    def _run(self):
        import torch
        import pandas as pd

        # Fix benchmark paths
        import benchmark
        benchmark.COMP_PATH = str(DATA_DIR / '2025_Completeness_783.csv')
        benchmark.CONN_PATH = str(DATA_DIR / '2025_Connectivity_783.parquet')
        benchmark.DATA_DIR = str(DATA_DIR)

        from run_pytorch import TorchModel, MODEL_PARAMS, DT, get_weights, get_hash_tables
        from brain_body_bridge import DN_NEURONS, DN_GROUPS, STIMULI, DNRateDecoder

        torch.set_num_threads(10)

        # Load brain
        print("[Brain] Loading 138,639 neurons...", flush=True)
        t0 = time.time()
        comp_path = str(DATA_DIR / '2025_Completeness_783.csv')
        conn_path = str(DATA_DIR / '2025_Connectivity_783.parquet')
        flyid2i, _ = get_hash_tables(comp_path)
        num_neurons = len(flyid2i)

        weights = get_weights(conn_path, comp_path, str(DATA_DIR))
        device = 'cpu'
        weights = weights.to(device=device)
        model = TorchModel(
            batch=1, size=num_neurons, dt=DT,
            params=MODEL_PARAMS, weights=weights, device=device,
        )
        state = model.state_init()
        rates = torch.zeros(1, num_neurons, device=device)

        # Stimulus index mapping
        stim_map = {}
        for sn, si in STIMULI.items():
            stim_map[sn] = [flyid2i[nid] for nid in si['neurons'] if nid in flyid2i]

        # DN mapping
        dn_map = {name: flyid2i[fid] for name, fid in DN_NEURONS.items() if fid in flyid2i}
        decoder = DNRateDecoder(window_ms=50.0, dt_ms=DT, max_rate=200.0)

        # Population tensors
        ann = pd.read_csv(PROJECT_DIR / "data" / "neuron_annotations.tsv", sep="\t", low_memory=False)

        def ids_to_tensor(root_ids):
            return torch.tensor([flyid2i[nid] for nid in root_ids if nid in flyid2i],
                                dtype=torch.long, device=device)

        mbon = ann[ann['cell_class'] == 'MBON']
        ser = ann[(ann['top_nt'] == 'serotonin') &
                  (~ann['cell_class'].isin(['olfactory', 'visual', 'mechanosensory', 'unknown_sensory', 'gustatory']))]
        octo = ann[(ann['top_nt'] == 'octopamine') &
                   (~ann['cell_class'].isin(['olfactory', 'visual', 'mechanosensory', 'unknown_sensory', 'gustatory']))]

        pop_tensors = {
            'pam': ids_to_tensor(ann[ann['cell_type'].str.startswith('PAM', na=False)]['root_id']),
            'ppl1': ids_to_tensor(ann[ann['cell_type'].str.startswith('PPL1', na=False)]['root_id']),
            'mbon_approach': ids_to_tensor(mbon[mbon['top_nt'] == 'acetylcholine']['root_id']),
            'mbon_avoidance': ids_to_tensor(mbon[mbon['top_nt'] == 'glutamate']['root_id']),
            'mbon_suppress': ids_to_tensor(mbon[mbon['top_nt'] == 'gaba']['root_id']),
            'serotonin': ids_to_tensor(ser['root_id']),
            'octopamine': ids_to_tensor(octo['root_id']),
            'gaba': ids_to_tensor(ann[ann['top_nt'] == 'gaba']['root_id']),
            'ach': ids_to_tensor(ann[ann['top_nt'] == 'acetylcholine']['root_id']),
            'glut': ids_to_tensor(ann[ann['top_nt'] == 'glutamate']['root_id']),
        }

        load_time = time.time() - t0
        print(f"[Brain] Loaded in {load_time:.1f}s. Running...", flush=True)

        # Simulation loop
        brain_step = 0
        emit_every = 10  # emit every 10 brain steps (~1ms sim time)
        spike_acc = torch.zeros(num_neurons, device=device)
        prev_stimuli = set()
        self._log_rows = []

        while self.running:
            # Check for stimulus changes
            with self._lock:
                current_stimuli = self._active_stimuli.copy()

            if current_stimuli != prev_stimuli:
                rates.zero_()
                for stim_name in current_stimuli:
                    if stim_name in stim_map:
                        rate = STIMULI[stim_name]['rate']
                        for idx in stim_map[stim_name]:
                            rates[0, idx] = rate
                prev_stimuli = current_stimuli.copy()

            # Brain step
            with torch.no_grad():
                cond, dbuf, spk, v, ref = state
                state = model(rates, cond, dbuf, spk, v, ref)
            spike_acc += state[2][0]

            dn_spikes = {name: state[2][0, idx].item() for name, idx in dn_map.items()}
            decoder.update(dn_spikes)

            brain_step += 1

            # Emit frame
            if brain_step % emit_every == 0:
                t_ms = brain_step * DT

                dn_rates = {g: round(decoder.get_group_rate(g), 4) for g in DN_GROUPS}

                # Behavior mode detection
                gf = dn_rates.get('escape', 0)
                grm = dn_rates.get('groom', 0)
                fed = dn_rates.get('feed', 0)
                fwd = dn_rates.get('forward', 0)
                if gf > 0.06:
                    behavior = 'escape'
                elif grm > 0.02:
                    behavior = 'grooming'
                elif fed > 0.05:
                    behavior = 'feeding'
                elif fwd > 0.01:
                    behavior = 'walking'
                else:
                    behavior = 'idle'

                pop = {name: int(spike_acc[idx].sum().item())
                       for name, idx in pop_tensors.items()}
                total = int(spike_acc.sum().item())
                spike_acc.zero_()

                frame_data = {
                    "event": "brain_frame",
                    "t_ms": round(t_ms, 1),
                    "brain_steps": brain_step,
                    "dn": dn_rates,
                    "pop": pop,
                    "total_spikes": total,
                    "behavior_mode": behavior,
                    "active_stimuli": list(current_stimuli),
                }
                self._emit(frame_data)

                # Log row (flat for CSV)
                row = {
                    't_ms': round(t_ms, 1),
                    'brain_steps': brain_step,
                    'stimuli': '+'.join(sorted(current_stimuli)) or 'none',
                    'behavior': behavior,
                    'total_spikes': total,
                }
                row.update({f'dn_{k}': v for k, v in dn_rates.items()})
                row.update({f'pop_{k}': v for k, v in pop.items()})
                self._log_rows.append(row)
