# RL Ads Recommendation System

A Reinforcement Learning based advertisement recommendation simulator built using Q-Learning, Flask, and an interactive frontend dashboard.

This project demonstrates how an RL agent can learn user preferences dynamically using click history instead of predefined user labels.

---

# Features

- Q-Learning based ad recommendation
- History-based state representation
- Dynamic Q-table learning
- Epsilon-greedy exploration vs exploitation
- Interactive training dashboard
- Live user session simulation
- CTR comparison against random baseline
- Real-time visualization using Chart.js
- Flask backend with REST APIs

---

# Tech Stack

- Python
- Flask
- NumPy
- HTML
- CSS
- JavaScript
- Chart.js

---

# Project Idea

Traditional recommendation systems often rely on predefined user profiles.

This project instead uses Reinforcement Learning where:

- The agent does NOT know the user type
- The agent only observes clicks and skips
- User preference is inferred gradually through interaction history

The RL agent learns which advertisement to show in order to maximize click-through rate (CTR).

---

# Reinforcement Learning Setup

## Environment

The environment simulates hidden user personas.

Possible personas:

- Sports Fan
- Tech Enthusiast
- Fashion Lover
- Generalist
- Ad Skeptic

Each persona has different click probabilities for each ad category.

---

## State Representation

The state is based on user click history.

For each ad category:
- 0 clicks
- 1 click
- 2 or more clicks

Since there are 3 ad categories:

```text
3 × 3 × 3 = 27 states
```

State encoding:

```text
state = sports_clicks × 9 + tech_clicks × 3 + fashion_clicks
```

---

## Actions

The agent can choose among:

- Sports Ad
- Tech Ad
- Fashion Ad

---

## Reward

```text
Click  -> +1 reward
Skip   ->  0 reward
```

---

## Learning Algorithm

The project uses Q-Learning.

Q-value update equation:

```text
Q(s,a) = Q(s,a) + α[r + γ max(Q(s',a')) − Q(s,a)]
```

Where:

- α = learning rate
- γ = discount factor
- ε = exploration rate

---

# Project Structure

```text
rl-ads-recommendation/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── templates/
    └── index.html
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/rl-ads-recommendation.git
cd rl-ads-recommendation
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Project

```bash
python app.py
```

Open in browser:

```text
http://127.0.0.1:5000
```

---

# Application Workflow

## Step 1 — Train Agent

The RL agent is trained over multiple simulated user sessions.

During training:
- the agent explores different ads
- observes clicks/skips
- updates Q-values
- improves CTR gradually

---

## Step 2 — Live User Simulation

After training:
- a hidden user persona is selected
- ads are shown one by one
- click history updates state
- agent adapts recommendations dynamically

---

# Dashboard Features

## Training Tab

- Hyperparameter controls
- Training progress
- CTR graph
- Q-table visualization
- State encoding explanation

---

## Test Tab

- Live recommendation simulation
- Click history bars
- Session timeline
- Exploration vs exploitation view
- Q-value comparison
- Session statistics

---

# API Endpoints

## Train Agent

```http
POST /api/train
```

### Parameters

```json
{
  "episodes": 3000,
  "alpha": 0.1,
  "gamma": 0.9,
  "epsilon": 1.0
}
```

---

## Simulate User

```http
POST /api/simulate_user
```

Simulates a single interaction step with a hidden user.

---

# Example Concepts Demonstrated

- Reinforcement Learning
- Q-Learning
- Markov Decision Process
- Recommendation Systems
- Epsilon-Greedy Policy
- User Behaviour Modeling
- Adaptive Decision Making

---

# Future Improvements

Possible enhancements:

- Deep Q Networks (DQN)
- Neural-network based policy learning
- Contextual bandits
- Real user datasets
- Persistent training storage
- User authentication
- Deployment on cloud platforms
- Database integration
- Analytics dashboard

---

# Screenshots

You can add screenshots here later.

Example:

```md
![Training Dashboard](screenshots/train.png)

![Live Simulation](screenshots/test.png)
```

---

# .gitignore

Recommended:

```gitignore
__pycache__/
*.pyc
.env
```

---

# requirements.txt

```txt
Flask==3.0.3
numpy==1.26.4
```

---

# Author

Ria Dalwani
