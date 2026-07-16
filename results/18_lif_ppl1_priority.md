# Analysis 18: PPL1 Priority — Biophysical LIF Validation

**Question:** Does PPL1 temporal priority hold in a biophysically realistic leaky integrate-and-fire (LIF) model with real connectome weights, GABA inhibition, and synaptic delays?

**Model:** Shiu et al. (Eon Systems) — 138,639 LIF neurons, ~5M synapses, alpha-function synapses, FlyWire v783 connectome.

---

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Membrane time constant | 20 ms |
| Resting potential | -52 mV |
| Threshold | -45 mV |
| Refractory period | 2.2 ms |
| Synaptic delay | 1.8 ms |
| Base synaptic weight | 0.275 mV |
| Timestep (dt) | 0.1 ms |
| Inhibition | Negative weights (GABA modeled) |

---

## Neuron Groups

| Group | Count |
|-------|------:|
| ORN (olfactory input) | 2,279 |
| PPL1 (punishment) | 16 |
| PAM (reward) | 307 |
| MBON | 96 |
| Kenyon Cells | 5,177 |
| Motor | 110 |

---

## Experiment 1: ORN Stimulation at Different Rates (100ms)

| Stim Rate | PPL1 First Spike | PAM First Spike | Delta | Result |
|----------:|-----------------:|----------------:|------:|--------|
| 50 Hz | 24.6 ms | 29.7 ms | +5.1 ms | **PPL1 FIRST** |
| 100 Hz | 23.7 ms | 27.4 ms | +3.7 ms | **PPL1 FIRST** |
| 200 Hz | 21.7 ms | 25.8 ms | +4.1 ms | **PPL1 FIRST** |
| 500 Hz | 20.0 ms | 24.0 ms | +4.0 ms | **PPL1 FIRST** |

## Experiment 2: Longer Simulation (1 second, 200 Hz)

| Stim Rate | PPL1 First Spike | PAM First Spike | Delta | Result |
|----------:|-----------------:|----------------:|------:|--------|
| 200 Hz | 21.6 ms | 25.2 ms | +3.6 ms | **PPL1 FIRST** |

---

## Signal Propagation Order

Across all conditions, the activation order is consistent:

```
ORN stimulation
  → KC fires first      (~15 ms)
  → PPL1 + MBON fire    (~21 ms)
  → PAM fires           (~25 ms)
  → Motor fires         (~75-80 ms, rare)
```

PPL1 consistently fires **3.6-5.1 ms before PAM** across all stimulation rates.

---

## Summary

| Metric | Value |
|--------|------:|
| Conditions tested | 5 |
| PPL1 first | **5/5 (100%)** |
| PAM first | **0/5 (0%)** |
| Tie | **0/5 (0%)** |
| Mean PPL1-PAM delta | **4.1 ms** |

---

## Cross-Model Comparison

| | Spreading Activation | LIF Biophysical | Larva (Spreading) |
|---|---|---|---|
| **Neuron model** | Binary threshold | LIF (V_th = -45 mV) | Binary threshold |
| **Time resolution** | Discrete steps | 0.1 ms timesteps | Discrete steps |
| **Synaptic weights** | Normalized (0-1) | Connectome-derived (0.275 mV) | Normalized (0-1) |
| **GABA inhibition** | Not modeled | Negative weights | Not modeled |
| **Refractory period** | None | 2.2 ms | None |
| **Synaptic delay** | None | 1.8 ms | None |
| **Data** | Adult FlyWire v783 | Adult FlyWire v783 | Larva (Winding 2023) |
| **Neurons** | 139,255 | 138,639 | 3,016 |
| **PPL1 before PAM?** | YES — 400/400 (100%) | YES — 5/5 (100%) | YES — 27/48 (56%), 0% PAM first |

---

## Key Findings

1. **PPL1 priority is model-independent.** The same result emerges from a simple spreading activation model and a biophysically detailed LIF simulation with inhibition, refractory periods, and synaptic delays. This rules out the possibility that the finding is an artifact of our simplified model.

2. **The temporal gap is ~4 ms in real time.** In the LIF model, PPL1 fires 3.6-5.1 ms before PAM. This is biologically significant — fast enough to gate downstream decision circuits before reward signals arrive.

3. **Stimulus intensity does not change the order.** Whether ORNs fire at 50 Hz or 500 Hz, PPL1 always leads. The priority is structural, not activity-dependent.

4. **Three independent validations converge:**
   - Adult spreading activation: 100% PPL1 first
   - Adult LIF biophysical: 100% PPL1 first
   - Larval spreading activation: 56% PPL1 first, 0% PAM first (rest ties)

   The finding is robust across models, parameters, and life stages.

---

*Model: Shiu et al. (Eon Systems) fly-brain. Data: FlyWire v783 connectome. Code: `18_lif_ppl1_priority.py`*
