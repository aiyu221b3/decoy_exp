# Decoy: Multi-Agent Environment for Hidden-Role Strategic Learning

**Author:** Ayushi Bhattacharya  
**Paper:** `ssrn-7345498` | **Code:** [aiyu221b3/decoy_exp](https://github.com/aiyu221b3/decoy_exp)

> **Abstract Summary:** Decoy is a compact four-agent, general-sum, finite-horizon partially observable stochastic game (POSG) designed to study hidden-role strategic learning under asymmetric information, Ornstein-Uhlenbeck financial dynamics, and socially reactive behaviour.

---

## 1. POSG Formulation & Agent Space

Each episode consists of four agents: an Investigator ($I$) and three Traders ($\mathcal{T} = \{T_1, T_2, T_3\}$) operating over a horizon of $T = 20$ timesteps with an Investigator attention budget of $B = 10$.

```math
\mathcal{G} = \{I, T_1, T_2, T_3\}
```

At the start of each episode, the three hidden roles—**Innocent**, **Fraudster**, and **Trickster**—are randomly permuted and assigned to the trader identities $\mathcal{T}$ so role identity cannot be inferred from a fixed player index. Role-conditioned learners are selected externally while excluding the explicit role label from each learner's state representation.

### State & Action Space Architecture

Legal-action masking restricts action selection and value bootstrapping strictly to valid actions at timestep $t$:

```math
a_i^t \in \mathcal{A}_i^{\text{legal}}(s_i^t) \subseteq \mathcal{A}_i
```

| Agent / Module | Representation | State Components | Action Space ($\mathcal{A}_i$) |
| :--- | :--- | :---: | :--- |
| **Investigator ($I$)** | Diagnostic State | $32$ | `OBSERVE`, `FOCUS`, `ACCUSE`, `VOTE` |
| **Investigator ($I$)** | Compact Q-State | $16$ | `OBSERVE`, `FOCUS`, `ACCUSE`, `VOTE` |
| **Trader ($T_j$)** | Market Learner State | $6$ | `NORMAL`, `CONCEAL`, `SIGNAL` |
| **Trader ($T_j$)** | Social Learner State | $6$ | `NEUTRAL`, `DEFEND`, `ACCUSE`, `SUPPORT` |

* **Compact Investigator Q-State ($16$ components):** Per-trader market bucket, inspection flag, and inspection-mean bucket ($3 \times 3 = 9$), latest social statement and target ($2$), current accusation target ($1$), budget and timestep buckets ($2$), remaining focus charges ($1$), and focus target ($1$).
* **Trader Market State ($6$ components):** Local market value, current accusation target, budget and timestep buckets, and most recent social statement and target.
* **Trader Social State ($6$ components):** Replaces the budget component with a binary indicator of whether the trader is currently accused.

---

## 2. Stochastic Financial Dynamics

Each trader's underlying latent market signal $X_t$ evolves via an independent **Ornstein-Uhlenbeck (OU)** mean-reverting stochastic process:

```math
dX_t = \theta(\mu - X_t)dt + \sigma\,dW_t
```

The environment advances the latent signal at each timestep via **Euler-Maruyama discretization**:

```math
X_{t+\Delta t} = X_t + \theta(\mu - X_t)\Delta t + \sigma\sqrt{\Delta t}\,\epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, 1)
```

### Financial Observation Model & Parameters

Before the Investigator observes the market, each trader $i$ applies a market action $a_{i,t}^{\text{mkt}}$ that scales the latent state prior to additive Gaussian observation noise:

```math
Y_{i,t} = m\left(a_{i,t}^{\text{mkt}}\right) X_{i,t} + \eta_{i,t}, \quad \eta_{i,t} \sim \mathcal{N}(0, \sigma_{\text{obs}}^2)
```

where the action-dependent signal modifier $m(a)$ is defined as:

```math
m(a) = \begin{cases} 
1.0, & a = \text{NORMAL}, \\ 
0.5, & a = \text{CONCEAL}, \\ 
1.5, & a = \text{SIGNAL}. 
\end{cases}
```

| Parameter | Symbol | Value | Notes |
| :--- | :---: | :---: | :--- |
| Mean-reversion rate | $\theta$ | $0.3$ | Controls pullback toward $\mu$ |
| Long-run mean | $\mu$ | $0.0$ | Baseline latent equilibrium |
| Volatility | $\sigma$ | $1.0$ | Wiener process scaling |
| Discretization step | $\Delta t$ | $0.01$ | Euler-Maruyama step size |
| Base observation noise | $\sigma_{\text{obs}}$ | $0.5$ | Standard deviation of $\eta_{i,t}$ |
| Focused observation noise | $\sigma_{\text{focus}}$ | $0.25$ | Halves $\sigma_{\text{obs}}$ for targeted trader on next step |

---

## 3. General-Sum Payoff Structure & Trickster WinRegister

### Terminal Outcome Rewards ($R(o)$)

An `ACCUSE` action is interactive and non-terminal, whereas `VOTE` commits to a terminal identification. The asymmetric terminal reward functions across roles establish the general-sum structure of the game:

```math
R_I(o) = \begin{cases} 
+10, & o = \text{fraudster\_caught}, \\ 
-10, & o = \text{innocent\_caught}, \\ 
-3, & o = \text{trickster\_caught}, \\ 
-1, & o = \text{timeout}. 
\end{cases}
```

```math
R_{\text{Innocent}}(o) = \begin{cases} 
+5, & o = \text{fraudster\_caught}, \\ 
-10, & o = \text{innocent\_caught}, \\ 
+1, & o = \text{trickster\_caught}, \\ 
0, & o = \text{timeout}. 
\end{cases}
```

```math
R_{\text{Fraudster}}(o) = \begin{cases} 
-10, & o = \text{fraudster\_caught}, \\ 
+10, & o = \text{innocent\_caught}, \\ 
+2, & o = \text{trickster\_caught}, \\ 
+1, & o = \text{timeout}. 
\end{cases}
```

* **Stepwise Social Shaping:**
  * **Innocent:** `ACCUSE` receives $0$; other legal non-neutral social actions receive $-1$.
  * **Fraudster:** `ACCUSE` receives $+1$; other legal social actions receive $0$.
  * **Trickster:** `ACCUSE` and `SUPPORT` receive $+1$; `NEUTRAL` and `DEFEND` receive $0$.

### Trickster Adaptive WinRegister Dynamics

The Trickster dynamically selects an allegiance (Investigator side $I$ vs. Criminal side $C$) using an online `WinRegister` that blends all-history decisive win rates ($p_{\text{all}}$) and recent-window decisive win rates ($p_{\text{recent}}$, up to $500$ outcomes):

```math
p_I = (1 - \lambda)p_{\text{all}} + \lambda p_{\text{recent}}, \quad \lambda = 0.65
```

```math
p_C = 1 - p_I, \quad \hat{w} = \arg\max_{w \in \{I, C\}} p_w
```

Commitment to side $\hat{w}$ occurs only when minimum episode history, confidence, and expected-margin conditions are simultaneously satisfied:

```math
\begin{aligned}
N_{\text{seen}} &\ge N_{\text{min}} && (N_{\text{min}} = 2000) \\
p_{\hat{w}} &\ge p_{\text{min}} && (p_{\text{min}} = 0.60) \\
|p_{\hat{w}} - 0.5| N_{\text{rem}} &\ge M_{\text{min}} && (M_{\text{min}} = 10)
\end{aligned}
```

where $N_{\text{rem}} = \max(N_{\text{total}} - n - 1, 0)$.
* **Pre-Commitment Shaping:** $+1$ for either a Fraudster or Innocent catch, $-1$ if accused during the episode, plus an entropy bonus every $100$ episodes equal to $2 \times$ the mean binary entropy over the last $100$ decisive updates.
* **Post-Commitment Shaping:** Legal social actions are restricted to defending/supporting the predicted ally and accusing the predicted opponent. Terminal reward yields $+5$ when $\hat{w}$ wins, $-5$ when the opposing side wins or the Trickster is caught, and $-1$ if accused.

---

## 4. Learning Algorithms & Hyperparameters

### Tabular Q-Learning (Online TD)
With legal-action masking, the continuation value, target $y_t$, and update rule are:

```math
V_Q(s) = \max_{a' \in \mathcal{A}^{\text{legal}}(s)} Q(s, a')
```

```math
y_t = \begin{cases} 
r_{t+1}, & \text{at terminal transitions}, \\ 
r_{t+1} + \gamma V_Q(s_{t+1}), & \text{otherwise}. 
\end{cases}
```

```math
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ y_t - Q(s_t, a_t) \right]
```

### Monte Carlo Learning (Every-Visit Sample Average)
Using discounted episodic return $G_t$ updated at the end of each episode:

```math
G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_{k+1}
```

```math
\begin{aligned}
N(s_t, a_t) &\leftarrow N(s_t, a_t) + 1 \\
Q(s_t, a_t) &\leftarrow Q(s_t, a_t) + \frac{G_t - Q(s_t, a_t)}{N(s_t, a_t)}
\end{aligned}
```

* **Training Configuration:** Learning rate $\alpha = 0.1$, discount factor $\gamma = 0.99$, training exploration $\epsilon = 0.1$, greedy evaluation $\epsilon = 0.0$. Frozen-opponent seeds: $\{11, 42, 99\}$; Joint-training seeds: $\{71, 74, 77\}$.

---

## 5. Empirical Benchmarks

### 5.1 Frozen-Opponent Convergence ($I$ Learning vs. Stationary $T$)
Evaluated over $2,000$ greedy episodes ($\epsilon = 0$) per seed across $3$ seeds ($\text{mean} \pm \text{SD}$):

| Algorithm | $10\text{k}$ Detection | $10\text{k}$ Reward | $500\text{k}$ Detection | $500\text{k}$ Reward |
| :--- | :---: | :---: | :---: | :---: |
| **Tabular Q-Learning** | $74.8 \pm 0.9\%$ | $6.50 \pm 0.10$ | $98.4 \pm 0.3\%$ | $9.81 \pm 0.04$ |
| **Monte Carlo** | $58.6 \pm 3.4\%$ | $4.20 \pm 0.47$ | $98.7 \pm 0.1\%$ | $9.84 \pm 0.01$ |

### 5.2 Joint-Training Co-Adaptation Sweep ($3$-Seed Means)
To isolate terminal identification accuracy from timeout avoidance, the **conditional Fraudster identification rate** is defined over non-timeout episodes:

```math
P(F \mid \text{no timeout}) = \frac{N_F}{N_F + N_I + N_T}
```

| Budget | Mean Reward | Fraudster ($N_F$) | Innocent ($N_I$) | Trickster ($N_T$) | Timeout | $P(F \mid \text{no timeout})$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$50\text{k}$** | $-0.448$ | $26.0\%$ | $20.9\%$ | $21.4\%$ | $31.7\%$ | **$38.1\%$** |
| **$100\text{k}$** | $-0.326$ | $25.5\%$ | $19.3\%$ | $19.9\%$ | $35.3\%$ | **$39.4\%$** |
| **$250\text{k}$** | $-0.082$ | $25.2\%$ | $16.6\%$ | $17.8\%$ | $40.4\%$ | **$42.3\%$** |
| **$500\text{k}$** | $+0.629$ | $29.2\%$ | $14.1\%$ | $15.9\%$ | $40.8\%$ | **$49.3\%$** |

### 5.3 Behavioural & Trickster Dynamics ($50\text{k} \rightarrow 500\text{k}$ Joint Training)

| Metric Category | Metric | @ $50\text{k}$ Episodes | @ $500\text{k}$ Episodes | Strategic Shift |
| :--- | :--- | :---: | :---: | :--- |
| **Investigator Behaviour** | Mean Observations / Ep | $2.03$ | $1.68$ | Less passive observation |
| **Investigator Behaviour** | Mean Accusations / Ep | $4.50$ | $5.73$ | More interactive probing |
| **Investigator Behaviour** | Mean Episode Length | $8.77$ | $9.87$ | Longer horizon survival |
| **Investigator Behaviour** | Remaining Budget | $2.88$ | $2.00$ | Higher budget utilization |
| **Trickster WinRegister** | All-History $I$ Win Prob ($p_{\text{all}}$) | $55.4\%$ | $67.5\%$ | Tracks declining false catches |
| **Trickster WinRegister** | Recent-Window $I$ Win Prob ($p_{\text{recent}}$) | $58.1\%$ | $75.2\%$ | Rapid adaptation to $I$ accuracy |
| **Trickster WinRegister** | Prediction Confidence ($p_{\hat{w}}$) | $57.1\%$ | $72.5\%$ | Exceeds $p_{\text{min}} = 0.60$ threshold |

### 5.4 Cross-Evaluation Diagnostics ($10\text{k}$ Episodes, $3$-Seed Means)

| Evaluation Regime | Fraudster | Innocent | Trickster | Timeout | Mean Reward |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Frozen $I$ vs. Staged $T$** | $24.0\%$ | $24.2\%$ | $26.2\%$ | $25.8\%$ | $-1.058$ |
| **Frozen $I$ vs. Joint $T$** | $28.8\%$ | $25.5\%$ | $29.1\%$ | $16.5\%$ | $-0.709$ |
| **Joint $I$ vs. Staged $T$** | $24.7\%$ | $24.7\%$ | $24.8\%$ | $25.8\%$ | $-1.002$ |
| **Joint $I$ vs. Joint $T$** | $27.0\%$ | $23.2\%$ | $22.1\%$ | $27.8\%$ | $-0.558$ |

---

## Citation

```bibtex
@article{bhattacharya2026decoy,
  title   = {Decoy: Multi-Agent Environment for Hidden-Role Strategic Learning},
  author  = {Bhattacharya, Ayushi},
  journal = {SSRN Electronic Journal},
  note    = {SSRN 7345498},
  year    = {2026},
  url     = {[https://github.com/aiyu221b3/decoy_exp](https://github.com/aiyu221b3/decoy_exp)}
}
```
