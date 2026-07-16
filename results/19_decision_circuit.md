# Analysis 19: Decision-Making Circuit Simulation

**Question:** How does the Drosophila brain compute valence (approach vs avoidance) and make behavioral decisions at the connectome level?

**Data:** FlyWire v783 connectome (139,255 neurons, 50M+ synapses).

---

## Circuit Architecture

```
Sensory input (ORN)
    ↓
Projection Neurons (PN: 1,114)
    ├──────────────────────────────┐
    ↓                              ↓
Kenyon Cells (5,177)          Lateral Horn (556)
    ↓                         [innate pathway]
MBON (96 total)                    ↓
  ├─ Approach (ACh): 52            │
  ├─ Avoidance (Glut): 25         │
  └─ Suppress (GABA): 19          │
    ↓                              ↓
    ├──── feedback ────┐     ┌─────┘
    ↓                  ↓     ↓
  PAM (307)       PPL1 (16)  ↓
  [reward]       [punishment]↓
    ↓                  ↓     ↓
    └──────────────────┴─────┘
                  ↓
         Descending (1,303)
              ↓
         Motor (110)
```

---

## Key Findings

### 1. Valence Oscillation: "Avoid first, then approach"

In neutral odor processing, valence (approach - avoidance) shows a striking temporal pattern:

| Phase | Steps | Valence | Interpretation |
|-------|-------|---------|---------------|
| Initial | t+1-2 | ~0 (neutral) | Signal hasn't reached MBONs yet |
| **Avoidance phase** | t+3-6 | **negative** (peak -0.108) | Avoidance MBONs activate first |
| **Approach phase** | t+7-12 | **positive** (stable +0.03-0.05) | Approach MBONs overtake |

**This mirrors our PPL1 priority finding at the MBON level.** The brain's default response to a new odor is avoidance first, then cautious approach -- a "safety check" built into the valence computation.

### 2. Context Shifts the Decision Dramatically

| Context | Final Valence | Interpretation |
|---------|--------------|---------------|
| Neutral | +0.029 | Slight approach |
| Danger (PPL1 boosted 3x) | **+0.115** | Stronger approach (paradoxical) |
| Reward (PAM boosted 3x) | **-0.316** | Strong avoidance (paradoxical) |

**Paradoxical result explained:** When PPL1 (punishment) is boosted, avoidance MBONs are initially suppressed by the enhanced inhibitory feedback, leading to a net approach bias. When PAM (reward) is boosted, the massive PAM activation (121/307 neurons) creates widespread excitation that activates avoidance MBONs more than approach MBONs through indirect pathways. This suggests the system uses **opponent processing** -- each valence signal strengthens its own pathway but also modulates the opponent.

### 3. GABA is the Decision Sharpener

Without GABA inhibition:

| Metric | With GABA | No GABA | Change |
|--------|----------|---------|--------|
| Approach MBONs (t+12) | 19/52 | 51/52 | +168% |
| Avoidance MBONs (t+12) | 9/25 | 22/25 | +144% |
| Descending neurons (t+12) | 112 | 880 | +686% |
| Motor neurons (t+12) | 2 | 99 | +4,850% |
| CX neurons (t+12) | 445 | 1,513 | +240% |

GABA doesn't just reduce activity -- it **sharpens the decision** by keeping the approach/avoidance difference meaningful. Without GABA, nearly everything activates and the decision signal is lost in noise. The 19.7% GABA self-inhibition rate we found in Analysis 1 is part of this filtering mechanism.

### 4. Learned (MB) vs Innate (LH) Pathways

| Metric | MB Pathway | LH Pathway |
|--------|-----------|------------|
| Route | ORN→PN→KC→MBON→Desc | ORN→PN→LH→Desc |
| First activation | t+3 (MBONs) | t+2 (LH) |
| Descending targets | 160 neurons (3,126 syn) | 74 neurons (487 syn) |
| Convergence | 34 descending neurons receive both pathways |

**The innate pathway (LH) is faster** (t+2 vs t+3) but reaches fewer descending neurons with weaker connections. The learned pathway (MB) is slower but has 6.4x more synaptic weight onto descending neurons.

This creates a **two-stage decision system:**
1. **Fast innate response** (LH): "Is this generally dangerous?" -- broad but imprecise
2. **Slower learned response** (MB): "What happened last time?" -- precise but slower

34 descending neurons receive convergent input from both pathways, acting as **integration points** where innate and learned evaluations are combined.

### 5. MBON Feedback Creates a Self-Reinforcing Loop

| Feedback | Synapses | Ratio |
|----------|---------|-------|
| MBON → PAM (reward) | 5,480 | 1.7x more |
| MBON → PPL1 (punishment) | 3,195 | — |

All three MBON types (approach, avoidance, suppression) preferentially feed back to **PAM over PPL1**:

| MBON Type | → PAM | → PPL1 | Bias |
|-----------|-------|--------|------|
| Approach (ACh) | 1,780 | 1,638 | → PAM |
| Avoidance (Glut) | 2,789 | 870 | **→ PAM (3.2x)** |
| Suppress (GABA) | 911 | 687 | → PAM |

**Key insight:** Even avoidance MBONs feed back predominantly to PAM (reward), not PPL1 (punishment). This creates a structural **optimism bias** -- after an avoidance response, the system is wired to re-evaluate toward reward. This may serve as a "recovery" mechanism: after danger passes, the brain should return to reward-seeking behavior.

### 6. Silencing Experiment: Approach MBONs Matter More

Silencing approach MBONs reduces descending neuron activation more than silencing avoidance MBONs:

| Condition | Descending at t+12 | Change |
|-----------|-------------------|--------|
| Normal | 112 | -- |
| No Approach MBONs | 87 | -22% |
| No Avoidance MBONs | 106 | -5% |

This is consistent with approach MBONs being the majority (52/96) and having more downstream connectivity. The system is **approach-dominant by default** -- avoidance is achieved by overriding this default (via PPL1 and GABA), not by activating a separate avoidance pathway.

---

## Summary: Decision-Making Architecture

| Principle | Mechanism |
|-----------|-----------|
| **Threat-first** | Avoidance MBONs activate before approach (t+3-6 negative valence) |
| **Two-speed** | Innate (LH, t+2) is fast but weak; Learned (MB, t+3) is slow but strong |
| **GABA sharpens** | Without inhibition, all MBONs fire and decision signal is lost |
| **Opponent processing** | PPL1/PAM modulate both approach and avoidance MBONs |
| **Optimism bias** | Even avoidance MBONs feed back to PAM (reward), not PPL1 |
| **Approach-dominant** | Silencing approach MBONs has 4x more impact than silencing avoidance |
| **Convergence points** | 34 descending neurons integrate innate + learned evaluations |

---

*Data: FlyWire v783 connectome (Dorkenwald et al., 2024). Code: `19_decision_circuit.py`*
