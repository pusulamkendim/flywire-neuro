# Analysis 20: Threat-First Agent — Bio-Inspired Decision Architecture

**Question:** Can decision-making principles from the Drosophila connectome improve reinforcement learning agents in dangerous environments?

**Method:** Q-learning agent augmented with three biological mechanisms vs standard Q-learning, tested in a lethal grid world (10×10, 4 food, 8 traps, 8 walls).

---

## Biological Mechanisms Implemented

| Mechanism | Biological Basis | Implementation |
|-----------|-----------------|----------------|
| **PPL1 threat modulation** | PPL1 (16 neurons) fires 4ms before PAM; avoidance MBONs activate before approach | (state, action) → danger score; proportionally penalizes Q-values of known-deadly actions |
| **LH innate caution** | Lateral Horn provides fast, hardwired responses (t+2 vs t+3 for MB) | When trap_dist ≤ 2, adds innate Q-value bonus for cautious actions — no learning required |
| **Optimism bias** | Avoidance MBONs feed back to PAM 3.2x more than PPL1 | After death, temporarily increases exploration (ε boost) to find alternative routes |

---

## Environment: FlyWorld (Lethal Grid)

```
10×10 grid
  · = empty
  F = food (+1 reward, respawns)
  X = trap (-5 reward, episode ends)
  █ = wall (impassable)

Trap:food ratio = 2:1 (dangerous world)
Max steps per episode = 200
```

---

## Results (10 independent runs × 500 episodes)

### Overall Performance

| Metric | Standard Q-Learning | Fly-Brain Agent | Improvement |
|--------|-------------------:|----------------:|-------------|
| **Total reward** | -869.5 | -703.8 | **+19.0%** |
| **Total traps hit** | 370 | 244 | **-34.1%** |
| Total food collected | 1,448 | 1,210 | -16.4% |
| **Avg traps/episode** | 0.739 | 0.488 | **-33.9%** |
| **Avg survival (steps)** | 96.8 | 141.0 | **+45.7%** |

### Learning Phases

| Phase | Metric | Standard | Fly-Brain | Delta |
|-------|--------|---------|----------|-------|
| Early (ep 1-50) | Avg reward | -3.08 | -3.04 | +0.04 |
| | Traps/ep | 0.91 | 0.80 | -0.12 |
| Mid (ep 200-250) | Avg reward | -1.36 | -0.92 | **+0.44** |
| | Traps/ep | 0.69 | 0.40 | **-0.29** |
| Late (ep 450-500) | Avg reward | -1.66 | -1.03 | **+0.64** |
| | Traps/ep | 0.73 | 0.42 | **-0.31** |

---

## Key Findings

### 1. Safety-Reward Trade-off Favors Biology

The Fly-Brain agent collects 16% less food but hits 34% fewer traps and survives 46% longer. In a lethal environment where a single mistake ends the episode, **survival is more valuable than aggression**. The net reward is 19% higher because avoiding -5 penalties outweighs missing +1 rewards.

This mirrors the Drosophila strategy: the brain defaults to caution ("avoid first, then approach" from Analysis 19) and uses GABA sharpening to filter impulsive decisions.

### 2. Biological Advantage Grows Over Time

The improvement increases across learning phases:

- Early: +0.04 reward (minimal difference — both agents are exploring)
- Mid: +0.44 reward (PPL1 threat memory begins protecting learned routes)
- Late: +0.64 reward (threat memory + optimism bias create robust policies)

The standard agent continues hitting traps at the same rate (0.73/ep late), while the Fly-Brain agent learns to avoid them (0.42/ep). Standard Q-learning can learn Q-values that penalize trap actions, but without explicit threat memory, this knowledge is fragile — Q-value updates from food rewards can overwrite trap avoidance.

### 3. PPL1 Modulation > PPL1 Override

Previous iterations (v1, v2) used PPL1 as a binary override:
- v1: Marked entire states as dangerous → agent got stuck
- v2: PPL1 overrode 84.5% of decisions → agent couldn't learn

The v3 design uses **proportional Q-value modulation** — PPL1 doesn't override decisions, it biases them. This mirrors the biology: PPL1 has only 16 neurons (vs 307 PAM), and it modulates MBONs rather than directly controlling motor output. The lesson: **biological systems influence decisions, they don't dictate them**.

### 4. LH Provides Day-Zero Safety

The LH innate caution mechanism requires no learning — it adds Q-value bonuses for cautious actions when near traps from episode 1. This is why the Fly-Brain agent already hits fewer traps in early episodes (0.80 vs 0.91). The two-speed system (fast innate LH + slow learned MB) means the agent is never completely naive about danger.

### 5. Optimism Bias Prevents Paralysis

After death, the optimism boost temporarily increases exploration (ε + 0.25, decaying at 0.85×/step). Without this, the agent would become increasingly conservative after encountering traps, potentially avoiding all exploration. The biological analogy: avoidance MBONs feed back to PAM (reward), not PPL1 (punishment), creating a structural bias toward recovery after negative experiences.

---

## Architecture Comparison

```
Standard Q-Learning:                Fly-Brain Agent:

  State → Q-table → argmax           State → Q-table → copy Q-values
                  → action                           → PPL1 modulate
                                                     → LH modulate
                                                     → argmax → action

  Reward → Q-update                   Reward → Q-update
                                      Death  → PPL1 threat memory
                                             → Optimism boost
```

---

## Design Principles (Connectome → RL)

| Drosophila Principle | RL Translation | Why It Works |
|---------------------|----------------|-------------|
| PPL1 fires before PAM | Threat memory penalizes dangerous Q-values | Prevents re-visiting lethal state-actions |
| LH is fast but imprecise | Innate bonus for caution near traps | Day-zero safety without learning |
| GABA sharpens decisions | Proportional modulation, not binary override | Preserves learning signal |
| Optimism bias (MBON→PAM) | ε boost after death | Prevents paralysis, finds alternatives |
| Approach-dominant default | Standard Q-learning as base | Only modulate, don't replace |

---

## Summary

| Metric | Standard | Fly-Brain | Winner |
|--------|---------|----------|--------|
| Safety (traps) | 0.739/ep | 0.488/ep | **Fly-Brain (-34%)** |
| Survival (steps) | 96.8 | 141.0 | **Fly-Brain (+46%)** |
| Reward (total) | -869.5 | -703.8 | **Fly-Brain (+19%)** |
| Food (total) | 1,448 | 1,210 | Standard (+16%) |
| **Overall** | | | **Fly-Brain** |

The Drosophila decision circuit's "threat-first" architecture translates effectively to RL: **survive first, optimize second**. The 139,255-neuron connectome encodes a safety-aware decision strategy that outperforms pure reward maximization in dangerous environments.

---

*Data: FlyWire v783 connectome principles. Code: `20_threat_first_agent.py`*
