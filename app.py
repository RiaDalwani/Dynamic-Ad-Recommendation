# app.py — RL Ads Recommendation v3
# State = click history vector (what we've learned about THIS user so far)
# Run: python app.py  →  open http://localhost:5000

from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────
# Ads and hidden user personas
# ─────────────────────────────────────────────────────────────────────
ADS        = ["Sports Ad", "Tech Ad", "Fashion Ad"]
N_ACTIONS  = len(ADS)

# Hidden user personas — agent never sees these labels.
# A real user arrives as one of these types, but we only observe their clicks.
PERSONAS = {
    "Sports fan":    [0.80, 0.20, 0.15],   # click prob per ad type
    "Tech enthusiast": [0.15, 0.80, 0.20],
    "Fashion lover": [0.10, 0.20, 0.85],
    "Generalist":    [0.40, 0.40, 0.40],   # clicks everything moderately
    "Ad skeptic":    [0.15, 0.15, 0.15],   # rarely clicks anything
}
PERSONA_NAMES = list(PERSONAS.keys())
PERSONA_PROBS = [PERSONAS[p] for p in PERSONA_NAMES]

# ─────────────────────────────────────────────────────────────────────
# State encoding
#
# State = discretised click history: for each ad type, how many times
# has the user clicked it so far?  Bucketed into 0 / 1 / 2+ clicks.
#
# Buckets per ad: [0, 1, 2+]  → 3 values
# 3 ad types  →  3^3 = 27 possible states
#
# State index = clicks_sports*9 + clicks_tech*3 + clicks_fashion
#   where each click count is capped at 2.
# ─────────────────────────────────────────────────────────────────────
BUCKET_MAX = 2          # cap click count at this value
N_BUCKETS  = BUCKET_MAX + 1                     # 3 buckets per ad
N_STATES   = N_BUCKETS ** N_ACTIONS             # 27

def history_to_state(click_history):
    """Convert [clicks_sports, clicks_tech, clicks_fashion] → state int."""
    idx = 0
    for i, c in enumerate(click_history):
        idx += min(c, BUCKET_MAX) * (N_BUCKETS ** (N_ACTIONS - 1 - i))
    return int(idx)

def state_label(state_idx):
    """Human-readable description of a state."""
    parts = []
    tmp = state_idx
    for i in range(N_ACTIONS):
        power = N_BUCKETS ** (N_ACTIONS - 1 - i)
        bucket = tmp // power
        tmp %= power
        label = f"{ADS[i].replace(' Ad','')}: {bucket if bucket < BUCKET_MAX else str(BUCKET_MAX)+'+'}✓"
        parts.append(label)
    return " | ".join(parts)

STATE_LABELS = [state_label(s) for s in range(N_STATES)]

# ─────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────
class AdEnvironment:
    def __init__(self):
        self.n_states  = N_STATES
        self.n_actions = N_ACTIONS

    def sample_persona(self):
        """Pick a random hidden persona for a new user."""
        idx = np.random.randint(len(PERSONA_NAMES))
        return idx, PERSONA_PROBS[idx]

    def get_reward(self, persona_probs, action):
        return int(np.random.rand() < persona_probs[action])

    def initial_history(self):
        return [0] * N_ACTIONS

# ─────────────────────────────────────────────────────────────────────
# Q-Learning Agent
# ─────────────────────────────────────────────────────────────────────
class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha, gamma, epsilon):
        self.n_states  = n_states
        self.n_actions = n_actions
        self.alpha     = alpha
        self.gamma     = gamma
        self.epsilon   = epsilon
        self.q_table   = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)    # explore
        return int(np.argmax(self.q_table[state]))       # exploit

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        td_error  = reward + self.gamma * best_next - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self, decay=0.999, min_eps=0.05):
        self.epsilon = max(min_eps, self.epsilon * decay)

    def best_action(self, state):
        return int(np.argmax(self.q_table[state]))


# ─────────────────────────────────────────────────────────────────────
# Training
#
# Each "episode" = one user session with up to MAX_ADS_PER_SESSION ads shown.
# The agent observes only clicks — never the persona label.
# ─────────────────────────────────────────────────────────────────────
MAX_ADS_PER_SESSION = 10

def train(episodes, alpha, gamma, epsilon):
    env   = AdEnvironment()
    agent = QLearningAgent(N_STATES, N_ACTIONS, alpha, gamma, epsilon)

    rl_ctr_history   = []   # avg CTR per episode
    rand_ctr_history = []

    for ep in range(episodes):
        _, persona_probs = env.sample_persona()
        history = env.initial_history()
        ep_clicks = 0
        ep_rand_clicks = 0

        for step in range(MAX_ADS_PER_SESSION):
            state  = history_to_state(history)
            action = agent.choose_action(state)
            reward = env.get_reward(persona_probs, action)

            # Update click history → new state
            new_history = history[:]
            if reward == 1:
                new_history[action] += 1
            next_state = history_to_state(new_history)

            agent.update(state, action, reward, next_state)
            history = new_history
            ep_clicks += reward

            # Random baseline (separate run, same persona)
            rand_action = np.random.randint(N_ACTIONS)
            ep_rand_clicks += env.get_reward(persona_probs, rand_action)

        agent.decay_epsilon()
        rl_ctr_history.append(ep_clicks / MAX_ADS_PER_SESSION)
        rand_ctr_history.append(ep_rand_clicks / MAX_ADS_PER_SESSION)

    return agent, rl_ctr_history, rand_ctr_history


# ─────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/train", methods=["POST"])
def api_train():
    data     = request.get_json()
    episodes = int(data.get("episodes", 3000))
    alpha    = float(data.get("alpha",  0.1))
    gamma    = float(data.get("gamma",  0.9))
    epsilon  = float(data.get("epsilon",1.0))

    agent, rl_hist, rand_hist = train(episodes, alpha, gamma, epsilon)

    def smooth(arr, w=100):
        step = max(1, len(arr) // 120)
        return [round(float(np.mean(arr[max(0,i-w):i+1])), 4)
                for i in range(0, len(arr), step)]

    q_table_out = [[round(float(v), 4) for v in row] for row in agent.q_table]

    # Which action does the agent recommend for each state?
    state_best = [
        {"state": STATE_LABELS[s], "best_ad": ADS[agent.best_action(s)],
         "q_values": [round(float(v),4) for v in agent.q_table[s]]}
        for s in range(N_STATES)
    ]

    return jsonify({
        "rl_curve":   smooth(rl_hist),
        "rand_curve": smooth(rand_hist),
        "q_table":    q_table_out,
        "state_best": state_best,
        "final_rl_ctr":   round(float(np.mean(rl_hist[-200:])), 4),
        "final_rand_ctr": round(float(np.mean(rand_hist[-200:])), 4),
        "n_states": N_STATES,
        "ads": ADS,
        "state_labels": STATE_LABELS,
    })


@app.route("/api/simulate_user", methods=["POST"])
def api_simulate_user():
    """
    Simulate showing ads to a new unknown user one at a time.
    Each call shows ONE ad, gets a click/no-click, updates history.

    Request:  { q_table, history, epsilon }
    Response: { action, reward, new_history, new_state, state_label,
                q_values, persona_revealed (only after session ends) }
    """
    data     = request.get_json()
    q_table  = np.array(data["q_table"])
    history  = data["history"]           # [clicks_sports, clicks_tech, clicks_fashion]
    epsilon  = float(data.get("epsilon", 0.0))
    persona_idx = int(data.get("persona_idx", -1))

    # Assign hidden persona on first step
    if persona_idx == -1:
        persona_idx = int(np.random.randint(len(PERSONA_NAMES)))
    persona_probs = PERSONA_PROBS[persona_idx]

    # Current state from history
    state = history_to_state(history)

    # Choose action (epsilon-greedy using provided epsilon)
    if np.random.rand() < epsilon:
        action = int(np.random.randint(N_ACTIONS))
        mode = "explore"
    else:
        action = int(np.argmax(q_table[state]))
        mode = "exploit"

    # Simulate click
    reward = int(np.random.rand() < persona_probs[action])

    # Update history
    new_history = history[:]
    if reward == 1:
        new_history[action] += 1
    new_state = history_to_state(new_history)

    return jsonify({
        "action":       action,
        "ad_shown":     ADS[action],
        "reward":       reward,
        "history":      new_history,
        "state":        new_state,
        "state_label":  STATE_LABELS[new_state],
        "q_values":     [round(float(v), 4) for v in q_table[new_state]],
        "mode":         mode,
        "persona_idx":  persona_idx,
        "persona_name": PERSONA_NAMES[persona_idx],   # hidden during live session, revealed at end
        "persona_probs": persona_probs,
    })


if __name__ == "__main__":
    app.run(debug=True)
