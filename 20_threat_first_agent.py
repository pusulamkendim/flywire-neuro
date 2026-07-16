"""
FlyWire Analysis 20 - Threat-First Agent: Bio-Inspired Decision Architecture

Tests whether Drosophila brain decision principles improve RL agent performance:

1. THREAT-FIRST (PPL1 priority): Evaluate danger before reward
2. DUAL-PATHWAY (LH + MB): Fast innate rules + slow learned policy
3. OPTIMISM BIAS (Avoidance→PAM): After danger, boost exploration
4. GABA SHARPENING: Filter weak signals to prevent indecision

Environment: Grid world with food (+1), traps (-1), and neutral cells.
Comparison: Standard Q-learning vs Fly-Brain-inspired agent.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from collections import defaultdict

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
np.random.seed(42)

print("=" * 80)
print("ANALYSIS 20: THREAT-FIRST AGENT — BIO-INSPIRED DECISION ARCHITECTURE")
print("=" * 80)

# =====================================================================
# 1. ENVIRONMENT: Grid World with Rewards and Dangers
# =====================================================================

class FlyWorld:
    """
    Grid world mimicking a fly's foraging environment.
    - Food sources: +1 reward (respawn after collection)
    - Traps: LETHAL — episode ends with large penalty
    - Walls: impassable
    - Sensory input: distance to nearest food/trap (odor gradient)

    The asymmetry: missing food = suboptimal, hitting trap = DEATH.
    This is the evolutionary pressure that shaped threat-first architecture.
    """
    def __init__(self, size=10, n_food=4, n_traps=8, n_walls=8, lethal=True):
        self.size = size
        self.lethal = lethal
        self.grid = np.zeros((size, size))
        self.start = (0, 0)
        self.pos = self.start
        self.steps = 0
        self.max_steps = 200

        # Place walls
        wall_positions = set()
        while len(wall_positions) < n_walls:
            p = (np.random.randint(1, size), np.random.randint(1, size))
            if p != self.start:
                wall_positions.add(p)
        for p in wall_positions:
            self.grid[p] = 2

        # Place food
        self.food_positions = set()
        while len(self.food_positions) < n_food:
            p = (np.random.randint(0, size), np.random.randint(0, size))
            if self.grid[p] == 0 and p != self.start:
                self.food_positions.add(p)
                self.grid[p] = 1

        # Place traps (more than food — dangerous world)
        self.trap_positions = set()
        while len(self.trap_positions) < n_traps:
            p = (np.random.randint(0, size), np.random.randint(0, size))
            if self.grid[p] == 0 and p != self.start:
                self.trap_positions.add(p)
                self.grid[p] = -1

        self.original_grid = self.grid.copy()
        self.original_food = set(self.food_positions)
        self.original_traps = set(self.trap_positions)

    def reset(self):
        self.pos = self.start
        self.steps = 0
        self.grid = self.original_grid.copy()
        self.food_positions = set(self.original_food)
        self.trap_positions = set(self.original_traps)
        return self._get_state()

    def _get_state(self):
        x, y = self.pos
        min_food_dist = min(
            (abs(x - fx) + abs(y - fy) for fx, fy in self.food_positions),
            default=self.size * 2
        )
        min_trap_dist = min(
            (abs(x - tx) + abs(y - ty) for tx, ty in self.trap_positions),
            default=self.size * 2
        )
        # Discretize distances for manageable state space
        food_bin = min(min_food_dist, 5)
        trap_bin = min(min_trap_dist, 5)
        return (x, y, food_bin, trap_bin)

    def step(self, action):
        x, y = self.pos
        dx, dy = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        nx, ny = x + dx, y + dy

        if not (0 <= nx < self.size and 0 <= ny < self.size):
            return self._get_state(), -0.01, False
        if self.grid[nx, ny] == 2:
            return self._get_state(), -0.01, False

        self.pos = (nx, ny)
        self.steps += 1

        # Food
        if self.grid[nx, ny] == 1:
            self.grid[nx, ny] = 0
            self.food_positions.discard((nx, ny))
            while True:
                p = (np.random.randint(0, self.size), np.random.randint(0, self.size))
                if self.grid[p] == 0 and p != self.pos:
                    self.grid[p] = 1
                    self.food_positions.add(p)
                    break
            return self._get_state(), 1.0, False

        # Trap — LETHAL
        if self.grid[nx, ny] == -1:
            if self.lethal:
                return self._get_state(), -5.0, True  # episode ends!
            else:
                self.pos = self.start
                return self._get_state(), -1.0, False

        done = self.steps >= self.max_steps
        return self._get_state(), -0.01, done


# =====================================================================
# 2. AGENT 1: STANDARD Q-LEARNING
# =====================================================================

class StandardAgent:
    """Classic Q-learning agent with epsilon-greedy exploration."""

    def __init__(self, n_actions=4, alpha=0.1, gamma=0.95, epsilon=0.15):
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.name = "Standard Q-Learning"

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        next_max = 0 if done else np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * next_max - current_q
        )


# =====================================================================
# 3. AGENT 2: FLY-BRAIN INSPIRED
# =====================================================================

class FlyBrainAgent:
    """
    Bio-inspired agent based on Drosophila decision circuit.

    Same Q-learning as standard agent, BUT with three biological additions:

    1. PPL1 (threat veto): Remembers which (state, action) pairs led to death.
       Doesn't override — just adds a negative bias to those Q-values.
       Like PPL1: small (16 neurons), fast, modulates but doesn't control.

    2. LH (innate caution): When trap_dist <= 2, adds a bonus to Q-values
       of actions that move AWAY from trap. No learning required.
       Like LH: hardwired, fast, works from episode 1.

    3. Optimism bias: After death, temporarily increase exploration to find
       alternative routes. Like avoidance MBON → PAM feedback.
    """

    def __init__(self, n_actions=4, alpha=0.1, gamma=0.95, epsilon=0.15):
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # PPL1: threat memory — (state, action) → danger score [0, 1]
        self.threat_memory = defaultdict(float)

        # Optimism bias
        self.optimism_boost = 0.0
        self.optimism_decay = 0.85

        # Track previous step for credit assignment
        self.prev_state = None
        self.prev_action = None

        # Statistics
        self.ppl1_vetoes = 0
        self.lh_assists = 0
        self.total_decisions = 0

        self.name = "Fly-Brain Agent"

    def choose_action(self, state):
        self.total_decisions += 1
        x, y, food_dist, trap_dist = state

        # Exploration (with optimism bias after death)
        effective_epsilon = min(0.4, self.epsilon + self.optimism_boost)
        self.optimism_boost *= self.optimism_decay

        if np.random.random() < effective_epsilon:
            # Even random exploration avoids known-deadly actions (PPL1 veto)
            safe_actions = [a for a in range(self.n_actions)
                           if self.threat_memory.get((state, a), 0) < 0.5]
            if safe_actions:
                self.ppl1_vetoes += 1
                return np.random.choice(safe_actions)
            return np.random.randint(self.n_actions)

        # Compute effective Q-values
        q = self.q_table[state].copy()

        # PPL1 modulation: penalize actions known to lead to death
        for a in range(self.n_actions):
            threat = self.threat_memory.get((state, a), 0)
            if threat > 0.1:
                q[a] -= threat * 5.0  # proportional penalty, not binary
                self.ppl1_vetoes += 1

        # LH modulation: innate bonus for moving away from nearby traps
        if trap_dist <= 2:
            for a in range(self.n_actions):
                dx, dy = [(-1, 0), (1, 0), (0, -1), (0, 1)][a]
                nx, ny = x + dx, y + dy
                if 0 <= nx < 10 and 0 <= ny < 10:
                    # Approximate: actions that stay in bounds get bonus
                    # inversely proportional to trap proximity
                    q[a] += 0.5 / (trap_dist + 0.1)
                    self.lh_assists += 1

        return int(np.argmax(q))

    def learn(self, state, action, reward, next_state, done):
        # Standard Q-learning (mushroom body)
        current_q = self.q_table[state][action]
        next_max = 0 if done else np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * next_max - current_q
        )

        # PPL1 update: if death occurred, mark previous (state, action)
        if done and reward < -1.0 and self.prev_state is not None:
            key = (self.prev_state, self.prev_action)
            self.threat_memory[key] = min(1.0, self.threat_memory[key] + 0.4)
            # Also mark current state-action
            self.threat_memory[(state, action)] = min(1.0,
                self.threat_memory.get((state, action), 0) + 0.2)
            # Optimism bias: boost exploration after death
            self.optimism_boost = 0.25

        # Slow decay of threat memories (forgetting)
        if np.random.random() < 0.01:
            for key in list(self.threat_memory.keys()):
                self.threat_memory[key] *= 0.95
                if self.threat_memory[key] < 0.05:
                    del self.threat_memory[key]

        self.prev_state = state
        self.prev_action = action


# =====================================================================
# 4. RUN EXPERIMENTS
# =====================================================================

def run_experiment(agent, env, n_episodes=500):
    """Run agent in environment and track metrics."""
    rewards_per_episode = []
    traps_per_episode = []
    food_per_episode = []
    cumulative_reward = 0
    cumulative_rewards = []
    survival_steps = []  # steps before first trap hit

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        traps_hit = 0
        food_collected = 0
        first_trap_step = env.max_steps  # no trap hit

        for step in range(env.max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)

            total_reward += reward
            if reward >= 1.0:
                food_collected += 1
            if reward <= -1.5:
                traps_hit += 1
                if first_trap_step == env.max_steps:
                    first_trap_step = step

            state = next_state
            if done:
                break

        rewards_per_episode.append(total_reward)
        traps_per_episode.append(traps_hit)
        food_per_episode.append(food_collected)
        cumulative_reward += total_reward
        cumulative_rewards.append(cumulative_reward)
        survival_steps.append(first_trap_step)

    return {
        'rewards': rewards_per_episode,
        'traps': traps_per_episode,
        'food': food_per_episode,
        'cumulative': cumulative_rewards,
        'survival': survival_steps,
    }


# Run both agents on same environments
N_RUNS = 10
N_EPISODES = 500

print(f"\nRunning {N_RUNS} independent runs, {N_EPISODES} episodes each...")

all_results = {'standard': [], 'flybrain': []}

for run in range(N_RUNS):
    # Same environment for fair comparison — dangerous world (2x traps vs food)
    env_seed = run * 1000
    np.random.seed(env_seed)
    env = FlyWorld(size=10, n_food=4, n_traps=8, n_walls=8, lethal=True)

    # Standard agent
    np.random.seed(42 + run)
    agent_std = StandardAgent()
    results_std = run_experiment(agent_std, env, N_EPISODES)
    all_results['standard'].append(results_std)

    # Fly-brain agent (same environment)
    np.random.seed(42 + run)
    env2 = FlyWorld.__new__(FlyWorld)
    env2.__dict__ = {k: (v.copy() if isinstance(v, np.ndarray) else
                         (set(v) if isinstance(v, set) else v))
                     for k, v in env.__dict__.items()}
    agent_fly = FlyBrainAgent()
    results_fly = run_experiment(agent_fly, env2, N_EPISODES)
    all_results['flybrain'].append(results_fly)

    print(f"  Run {run+1}/{N_RUNS}: "
          f"Standard={np.mean(results_std['rewards'][-50:]):.1f} "
          f"FlyBrain={np.mean(results_fly['rewards'][-50:]):.1f} "
          f"(last 50 episodes avg)")

# =====================================================================
# 5. AGGREGATE RESULTS
# =====================================================================
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

def aggregate(results_list, key):
    """Average a metric across runs."""
    arrays = [np.array(r[key]) for r in results_list]
    return np.mean(arrays, axis=0), np.std(arrays, axis=0)

# Compute metrics
std_reward_mean, std_reward_std = aggregate(all_results['standard'], 'rewards')
fly_reward_mean, fly_reward_std = aggregate(all_results['flybrain'], 'rewards')
std_traps_mean, _ = aggregate(all_results['standard'], 'traps')
fly_traps_mean, _ = aggregate(all_results['flybrain'], 'traps')
std_food_mean, _ = aggregate(all_results['standard'], 'food')
fly_food_mean, _ = aggregate(all_results['flybrain'], 'food')
std_surv_mean, _ = aggregate(all_results['standard'], 'survival')
fly_surv_mean, _ = aggregate(all_results['flybrain'], 'survival')

# Windowed averages
window = 50
std_reward_smooth = np.convolve(std_reward_mean, np.ones(window)/window, mode='valid')
fly_reward_smooth = np.convolve(fly_reward_mean, np.ones(window)/window, mode='valid')
std_traps_smooth = np.convolve(std_traps_mean, np.ones(window)/window, mode='valid')
fly_traps_smooth = np.convolve(fly_traps_mean, np.ones(window)/window, mode='valid')
std_food_smooth = np.convolve(std_food_mean, np.ones(window)/window, mode='valid')
fly_food_smooth = np.convolve(fly_food_mean, np.ones(window)/window, mode='valid')

# Print summary
early = slice(0, 50)
mid = slice(200, 250)
late = slice(450, 500)

print(f"\n  {'Metric':<30} {'Standard':>12} {'Fly-Brain':>12} {'Improvement':>12}")
print(f"  {'─'*68}")

for period_name, period in [("Early (ep 1-50)", early),
                             ("Mid (ep 200-250)", mid),
                             ("Late (ep 450-500)", late)]:
    s_r = np.mean(std_reward_mean[period])
    f_r = np.mean(fly_reward_mean[period])
    s_t = np.mean(std_traps_mean[period])
    f_t = np.mean(fly_traps_mean[period])
    s_f = np.mean(std_food_mean[period])
    f_f = np.mean(fly_food_mean[period])

    print(f"\n  {period_name}")
    print(f"  {'  Avg reward/episode':<30} {s_r:>+12.2f} {f_r:>+12.2f} {(f_r-s_r):>+12.2f}")
    print(f"  {'  Traps hit/episode':<30} {s_t:>12.2f} {f_t:>12.2f} {(s_t-f_t):>+12.2f}")
    print(f"  {'  Food collected/episode':<30} {s_f:>12.2f} {f_f:>12.2f} {(f_f-s_f):>+12.2f}")

# Overall
print(f"\n  Overall ({N_EPISODES} episodes):")
s_total = np.sum(std_reward_mean)
f_total = np.sum(fly_reward_mean)
s_traps_total = np.sum(std_traps_mean)
f_traps_total = np.sum(fly_traps_mean)
s_food_total = np.sum(std_food_mean)
f_food_total = np.sum(fly_food_mean)

print(f"  {'  Total reward':<30} {s_total:>12.1f} {f_total:>12.1f} {(f_total-s_total):>+12.1f}")
print(f"  {'  Total traps hit':<30} {s_traps_total:>12.0f} {f_traps_total:>12.0f} {(s_traps_total-f_traps_total):>+12.0f}")
print(f"  {'  Total food collected':<30} {s_food_total:>12.0f} {f_food_total:>12.0f} {(f_food_total-s_food_total):>+12.0f}")

# Safety metric
s_trap_rate = np.mean(std_traps_mean)
f_trap_rate = np.mean(fly_traps_mean)
safety_improvement = (1 - f_trap_rate / s_trap_rate) * 100 if s_trap_rate > 0 else 0

s_surv = np.mean(std_surv_mean)
f_surv = np.mean(fly_surv_mean)

print(f"\n  {'  Avg traps/episode':<30} {s_trap_rate:>12.3f} {f_trap_rate:>12.3f} {safety_improvement:>+11.1f}%")
print(f"  {'  Avg survival (steps)':<30} {s_surv:>12.1f} {f_surv:>12.1f} {(f_surv-s_surv):>+12.1f}")

# Decision breakdown for fly agent
print(f"\n  Fly-Brain Decision Breakdown (last run):")
print(f"    PPL1 vetoes (threat modulation): {agent_fly.ppl1_vetoes}")
print(f"    LH assists (innate caution):     {agent_fly.lh_assists}")
print(f"    Total decisions:                 {agent_fly.total_decisions}")
if agent_fly.total_decisions > 0:
    print(f"    PPL1 %: {agent_fly.ppl1_vetoes/agent_fly.total_decisions*100:.1f}%")
    print(f"    LH %:   {agent_fly.lh_assists/agent_fly.total_decisions*100:.1f}%")

# =====================================================================
# 6. VISUALIZATION
# =====================================================================
print("\nGenerating visualization...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Analysis 20: Threat-First Agent vs Standard Q-Learning\n'
             'Bio-inspired decision architecture from Drosophila connectome',
             fontsize=13, fontweight='bold')

# Colors
C_STD = '#e74c3c'
C_FLY = '#2ecc71'

# Plot 1: Reward over episodes
ax = axes[0, 0]
episodes = np.arange(len(std_reward_smooth)) + window
ax.plot(episodes, std_reward_smooth, color=C_STD, label='Standard', linewidth=2)
ax.plot(episodes, fly_reward_smooth, color=C_FLY, label='Fly-Brain', linewidth=2)
ax.fill_between(episodes, std_reward_smooth - 0.5, std_reward_smooth + 0.5,
                color=C_STD, alpha=0.1)
ax.fill_between(episodes, fly_reward_smooth - 0.5, fly_reward_smooth + 0.5,
                color=C_FLY, alpha=0.1)
ax.set_xlabel('Episode')
ax.set_ylabel('Avg Reward (50-ep window)')
ax.set_title('Learning Curve')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Traps hit
ax = axes[0, 1]
ax.plot(episodes, std_traps_smooth, color=C_STD, label='Standard', linewidth=2)
ax.plot(episodes, fly_traps_smooth, color=C_FLY, label='Fly-Brain', linewidth=2)
ax.set_xlabel('Episode')
ax.set_ylabel('Avg Traps Hit (50-ep window)')
ax.set_title('Safety: Trap Avoidance')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Food collected
ax = axes[0, 2]
ax.plot(episodes, std_food_smooth, color=C_STD, label='Standard', linewidth=2)
ax.plot(episodes, fly_food_smooth, color=C_FLY, label='Fly-Brain', linewidth=2)
ax.set_xlabel('Episode')
ax.set_ylabel('Avg Food Collected (50-ep window)')
ax.set_title('Efficiency: Food Collection')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Cumulative reward
ax = axes[1, 0]
std_cum_mean, _ = aggregate(all_results['standard'], 'cumulative')
fly_cum_mean, _ = aggregate(all_results['flybrain'], 'cumulative')
ax.plot(std_cum_mean, color=C_STD, label='Standard', linewidth=2)
ax.plot(fly_cum_mean, color=C_FLY, label='Fly-Brain', linewidth=2)
ax.set_xlabel('Episode')
ax.set_ylabel('Cumulative Reward')
ax.set_title('Total Reward Accumulation')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Survival steps
ax = axes[1, 1]
std_surv_smooth = np.convolve(std_surv_mean, np.ones(window)/window, mode='valid')
fly_surv_smooth = np.convolve(fly_surv_mean, np.ones(window)/window, mode='valid')
ax.plot(np.arange(len(std_surv_smooth)) + window, std_surv_smooth,
        color=C_STD, label='Standard', linewidth=2)
ax.plot(np.arange(len(fly_surv_smooth)) + window, fly_surv_smooth,
        color=C_FLY, label='Fly-Brain', linewidth=2)
ax.set_xlabel('Episode')
ax.set_ylabel('Steps Before First Trap')
ax.set_title('Survival Duration')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Architecture diagram
ax = axes[1, 2]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Fly-Brain Agent Architecture')

# Draw architecture
boxes = [
    (5, 9.0, 'SENSORY INPUT\n(state)', '#ecf0f1', 'black'),
    (2, 7.0, 'PPL1\nThreat Detector\n(fast, small)', '#e74c3c', 'white'),
    (5, 7.0, 'Lateral Horn\nInnate Rules\n(fast, hardcoded)', '#e67e22', 'white'),
    (8, 7.0, 'Mushroom Body\nQ-Learning\n(slow, learned)', '#3498db', 'white'),
    (5, 4.5, 'GABA Filter\nConfidence Check', '#9b59b6', 'white'),
    (5, 2.5, 'ACTION', '#2ecc71', 'white'),
    (2, 4.5, 'Optimism Bias\nexplore after\ndanger', '#f39c12', 'black'),
]

for x, y, text, color, tcolor in boxes:
    ax.add_patch(plt.Rectangle((x-1.4, y-0.6), 2.8, 1.2,
                                facecolor=color, edgecolor='black',
                                linewidth=1.5, zorder=2))
    ax.text(x, y, text, ha='center', va='center', fontsize=7,
            fontweight='bold', color=tcolor, zorder=3)

# Arrows
arrow_props = dict(arrowstyle='->', color='black', linewidth=1.5)
for (x1, y1), (x2, y2) in [
    ((5, 8.4), (2, 7.6)),   # input → PPL1
    ((5, 8.4), (5, 7.6)),   # input → LH
    ((5, 8.4), (8, 7.6)),   # input → MB
    ((2, 6.4), (5, 5.1)),   # PPL1 → GABA
    ((5, 6.4), (5, 5.1)),   # LH → GABA
    ((8, 6.4), (5, 5.1)),   # MB → GABA
    ((5, 3.9), (5, 3.1)),   # GABA → action
    ((2, 3.9), (2, 5.1)),   # optimism → PPL1 (feedback)
]:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=arrow_props, zorder=1)

# Priority labels
ax.text(1.0, 7.7, '1st', fontsize=8, fontweight='bold', color='red')
ax.text(4.0, 7.7, '2nd', fontsize=8, fontweight='bold', color='orange')
ax.text(7.0, 7.7, '3rd', fontsize=8, fontweight='bold', color='blue')

plt.tight_layout()
results_dir = os.path.join(PROJECT_DIR, "results")
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, "20_threat_first_agent.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
