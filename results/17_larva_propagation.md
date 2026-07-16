# Analysis 17: Larval Drosophila Signal Propagation

**Question:** Does the PPL1 (aversive DAN) temporal priority over PAM (appetitive DAN) observed in the adult connectome also exist in the larval brain?

**Data:** Winding et al. (2023) *Science* — 3,016 neurons, 548K synapses.

---

## Neuron Groups

| Group | Count |
|-------|------:|
| Olfactory sensory (ORN) | 42 |
| Gustatory sensory | 238 |
| Projection neurons (PN) | 206 |
| Local neurons (LN) | 110 |
| Kenyon Cells (KC) | 144 |
| MBONs | 48 |
| DAN aversive (PPL1-like) | 8 |
| DAN appetitive (PAM-like) | 6 |
| OAN | 4 |
| Other MBIN | 10 |
| Motor (DN) | 346 |

---

## Olfactory Signal Propagation

| Step | ORN | PN | LN | KC | MBON | DAN aversive | DAN appetitive | Motor | Total |
|------|----:|---:|---:|---:|-----:|-------------:|---------------:|------:|------:|
| t+0 | 42 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 42 |
| t+1 | 42 | 71 | 32 | 0 | 0 | 0 | 0 | 8 | 162 |
| t+2 | 42 | 116 | 71 | 132 | 9 | **1** | 0 | 41 | 767 |
| t+3 | 42 | 165 | 100 | 144 | 48 | **8** | **6** | 117 | 1,560 |
| t+4 | 42 | 199 | 104 | 144 | 48 | 8 | 6 | 266 | 2,164 |
| t+5 | 42 | 199 | 110 | 144 | 48 | 8 | 6 | 321 | 2,409 |

**First activation:**
- DAN aversive (PPL1-like): **t+2**
- DAN appetitive (PAM-like): **t+3**
- Motor neurons: t+1

> **DAN aversive activates 1 step before appetitive — consistent with adult finding.**

---

## Gustatory Signal Propagation

| Step | Gust | PN | LN | KC | MBON | DAN aversive | DAN appetitive | Motor | Total |
|------|-----:|---:|---:|---:|-----:|-------------:|---------------:|------:|------:|
| t+0 | 238 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 238 |
| t+1 | 0 | 108 | 20 | 0 | 0 | 0 | 0 | 71 | 457 |
| t+2 | 31 | 173 | 77 | 32 | 15 | **6** | **2** | 102 | 1,178 |
| t+3 | 42 | 191 | 98 | 144 | 48 | 8 | 6 | 193 | 1,795 |

**First activation:**
- DAN aversive: **t+2**
- DAN appetitive: **t+2**
- Motor neurons: t+1

> Gustatory pathway: both DAN types activate simultaneously (same as adult).

---

## Parameter Sensitivity (48 combinations)

| Outcome | Count | Percentage |
|---------|------:|----------:|
| Aversive first | 27 | 56.2% |
| Appetitive first | 0 | **0.0%** |
| Tie | 21 | 43.8% |

> Across all parameter combinations, appetitive DAN **never** fires before aversive DAN.

---

## Structural Analysis: DAN Inputs

| Metric | DAN aversive | DAN appetitive |
|--------|-------------:|---------------:|
| Neuron count | 8 | 6 |
| Total input synapses | 3,926 | 3,336 |
| Per neuron | 491 | 556 |
| Input ratio (aversive/appetitive) | 0.9x | — |

### Top Input Sources

**DAN aversive:**

| Source | Synapses | Share |
|--------|--------:|------:|
| KC | 1,922 | 49.0% |
| MB-FBN | 904 | 23.0% |
| MB-FFN | 282 | 7.2% |
| PN | 203 | 5.2% |
| LN | 178 | 4.5% |

**DAN appetitive:**

| Source | Synapses | Share |
|--------|--------:|------:|
| KC | 2,118 | 63.5% |
| MB-FBN | 424 | 12.7% |
| MB-FFN | 190 | 5.7% |
| PN | 166 | 5.0% |
| MBON | 137 | 4.1% |

---

## Adult vs Larva Comparison

| | Adult (FlyWire) | Larva (Winding) |
|---|---:|---:|
| Total neurons | 139,255 | 3,016 |
| Total synapses | 50M+ | 548K |
| Kenyon Cells | 5,177 | 121 |
| MBONs | 96 | 24 |
| Aversive DAN | 16 (PPL1) | 8 (DAN-c/d/f/g) |
| Appetitive DAN | 307 (PAM) | 6 (DAN-i/j/k) |
| Aversive:Appetitive ratio | 1:19 | 8:6 |
| Olfactory → Aversive | t+4 | t+2 |
| Olfactory → Appetitive | t+7 | t+3 |
| Input/neuron (aversive) | 4,804 | 491 |
| Input/neuron (appetitive) | 414 | 556 |

---

## Key Findings

1. **PPL1 priority is conserved across life stages.** Aversive DANs activate before appetitive DANs in both adult and larval brains, despite a 46x difference in brain size.

2. **Different structural mechanism, same functional outcome.** In adults, PPL1 has 11.6x more input per neuron than PAM. In larvae, the ratio is nearly equal (0.9x). Yet the temporal priority still holds — suggesting the advantage comes from network topology (shorter path length) rather than raw synaptic weight.

3. **Gustatory pathway bypasses the priority.** In both adult and larva, taste signals reach aversive and appetitive DANs simultaneously, consistent with the "spit it out NOW" reflex strategy.

4. **Zero exceptions across parameters.** In 48 parameter combinations, appetitive DAN never fired before aversive DAN (0.0%), matching the adult result of 0/400.

---

*Data: Winding et al. (2023) Science. Code: `17_larva_propagation.py`*
