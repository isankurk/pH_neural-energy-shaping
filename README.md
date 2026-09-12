# Co-Learning Port-Hamiltonian Systems and Optimal Energy-Shaping Control

[![Journal](https://img.shields.io/badge/Journal-ASME%20J.%20Dyn.%20Sys.%20Meas.%20Control-blue.svg)](https://asmedigitalcollection.asme.org/)

This repository contains the implementation and documentation for the paper:

> **Co-Learning Port-Hamiltonian Systems and Optimal Energy-Shaping Control**  
> *Ankur Kamboj, Biswadip Dey, and Vaibhav Srivastava*  
> *ASME Journal of Dynamic Systems, Measurement and Control*

---

## Abstract

We develop a physics-informed learning framework for energy-shaping control of fully-actuated port-Hamiltonian (pH) systems from trajectory data. The proposed approach co-learns a pH system model and an optimal energy-balancing passivity-based controller (EB-PBC) through alternating optimization with policy-aware data collection. At each iteration, the system model is refined using trajectory data collected under the current control policy, and the controller is re-optimized on the updated model. Both components are parameterized by neural networks that embed the pH dynamics and EB-PBC structure, ensuring interpretability in terms of energy interactions. The learned controller renders the closed-loop system inherently passive, with the target equilibrium provably locally asymptotically stable nominally and uniformly practically stable under model mismatch, and exploits passive plant dynamics without canceling the natural potential. Dissipation regularization promotes strict energy decay on the learned closed-loop model, providing robustness margins against model mismatch. The proposed framework is validated on state-regulation and swing-up tasks for planar and torsional pendulum systems.

---

## Key Contributions

1. **Alternating Optimization with Policy-Aware Data Collection**: A co-design framework where pH system identification and EB-PBC policy optimization alternate. Trajectory data collected under the evolving policy concentrates model capacity in the region of the state space traversed by the closed-loop system.
2. **Physics-Informed Architecture Preserving Passivity**: Structured neural network parameterization that guarantees port-Hamiltonian dynamics for the learned model and an inherently passive EB-PBC policy at every training iterate. The policy learns an added potential $V_\phi^*(\mathbf{q})$ that exploits natural passive plant dynamics rather than canceling them.
3. **Dissipation Regularization**: A penalty term enforcing strict closed-loop energy dissipation during training, providing explicit robustness margins against model approximation errors and sim-to-real gaps.
4. **Deterministic Practical Stability Guarantees**: Rigorous Lyapunov analysis establishing uniform practical asymptotic stability (uniform ultimate boundedness) of the true system under bounded model mismatch, with an explicit ultimate bound $\mathcal{O}\left(\sqrt{\frac{\xi + \varepsilon_{\mathrm{diss}}}{\rho}}\right)$.

---

## Framework Overview

```
                        ┌─────────────────────────────────────────────────────────┐
                        │             Phase 0: Warm-up Model Training             │
                        │    Train initial pH model f_θ on step-excited data      │
                        └───────────────────────────┬─────────────────────────────┘
                                                    │
                                                    ▼
                       ┌───────────────────────────────────────────────────────────┐
                       │        Phase 1: Alternating Optimization Iterations       │
                       │                                                           │
                       │  ┌───────────────────────┐     ┌───────────────────────┐  │
                       │  │   (a) θ-Step (Model)  │     │  (b) ϕ-Step (Policy)  │  │
                       │  │ Refine f_θ with both  │ ──► │ Optimize u_ϕ on f_θ   │  │
                       │  │ step-excited & policy-│     │ via NODE adjoint with │  │
                       │  │ excited trajectory    │ ◄── │ dissipation           │  │
                       │  │ data; ϕ frozen        │     │ regularization;       │  │
                       │  │                       │     │ θ frozen              │  │
                       │  └───────────────────────┘     └───────────────────────┘  │
                       └───────────────────────────────────────────────────────────┘
```

### 1. Port-Hamiltonian Dynamics Formulation
The physical system is modeled as a port-Hamiltonian system with generalized coordinates $\mathbf{q} \in \mathbb{R}^n$ and momenta $\mathbf{p} \in \mathbb{R}^n$:

$$\begin{bmatrix} \dot{\mathbf{q}} \\ \dot{\mathbf{p}} \end{bmatrix} = \begin{bmatrix} \mathbf{0} & \mathbf{I} \\ -\mathbf{I} & -\mathbf{D}(\mathbf{q}) \end{bmatrix} \begin{bmatrix} \nabla_{\mathbf{q}} H \\ \nabla_{\mathbf{p}} H \end{bmatrix} + \begin{bmatrix} \mathbf{0} \\ \mathbf{g}(\mathbf{q}) \end{bmatrix} \mathbf{u} \iff \dot{\mathbf{z}} = \mathbf{F}(\mathbf{q}) \nabla_{\mathbf{z}} H + \mathbf{G}(\mathbf{q}) \mathbf{u}$$

where the Hamiltonian is:

$$H(\mathbf{q}, \mathbf{p}) = \frac{1}{2} \mathbf{p}^\top \mathbf{M}^{-1}(\mathbf{q}) \mathbf{p} + V(\mathbf{q})$$

The learned dynamics $\mathbf{f}_\theta$ parameterize individual physical components using separate neural networks:

$$\mathbf{f}_\theta(\mathbf{z}, \mathbf{u}) = \begin{bmatrix} \mathbf{0} & \mathbf{I} \\ -\mathbf{I} & -\mathbf{D}_{\theta_4}(\mathbf{q}) \end{bmatrix} \begin{bmatrix} \nabla_{\mathbf{q}} H_\theta \\ \nabla_{\mathbf{p}} H_\theta \end{bmatrix} + \begin{bmatrix} \mathbf{0} \\ \mathbf{g}_{\theta_3}(\mathbf{q}) \end{bmatrix} \mathbf{u}$$

with $H_\theta(\mathbf{z}) = \frac{1}{2} \mathbf{p}^\top \mathbf{M}_{\theta_1}^{-1}(\mathbf{q}) \mathbf{p} + V_{\theta_2}(\mathbf{q})$.

### 2. Neural Optimal Energy Shaping (OES) Control
The parameterization of the EB-PBC policy incorporates an added potential $V_\phi^\*(\mathbf{q})$ and a state- and time-dependent damping injection matrix $\mathbf{K}_\phi^*(t, \mathbf{z})$:

$$\mathbf{u}_\phi(t, \mathbf{z}) = -\mathbf{G}_\theta^\dagger(\mathbf{q}) \mathbf{F}_\theta^\top(\mathbf{q}) \begin{bmatrix} \nabla_{\mathbf{q}} V_\phi^\*(\mathbf{q}) \\ \mathbf{0} \end{bmatrix} - \mathbf{K}_\phi^*(t, \mathbf{z}) \mathbf{g}_\theta^\top(\mathbf{q}) \nabla_{\mathbf{p}} H_\theta(\mathbf{z})$$

For fully-actuated systems ($\text{rank}(\mathbf{g}(\mathbf{q})) = n$), the matching partial differential equations are satisfied automatically without requiring cancellation of the natural potential $V(\mathbf{q})$.

### 3. Dissipation Regularization
To impart robustness against model mismatch (e.g., sim-to-real gap), the policy objective is augmented with a dissipation penalty:

$$\mathcal{L}_u = \ell_u(\mathbf{z}_0) + \mathcal{L}_{\mathrm{diss}}$$

$$\mathcal{L}_{\mathrm{diss}} = \lambda_{\mathrm{diss}} \mathbb{E}_{\mathbf{z}} \left[ \text{ReLU}\left( \dot{H}_{\theta,\phi_d}(\mathbf{z}) + \rho \|\mathbf{z} - \mathbf{z}_d\|^2 \right) \right]$$

where $\dot{H}_{\theta,\phi_d}(\mathbf{z}) = \nabla H_{\theta,\phi_d}^\top \mathbf{f}_\theta(\mathbf{z}, \mathbf{u}_\phi(\mathbf{z}))$, $\rho > 0$ is the prescribed dissipation rate, and $\lambda_{\mathrm{diss}}$ is the regularization weight.

---

## Theoretical Guarantees

- **Nominal Stability (Lemmas 1 & 2)**: Under the parameterized EB-PBC law with $V_\phi^* \in \mathcal{C}^2$ and $\mathbf{K}_\phi^*(t, \mathbf{z}) \succeq \kappa \mathbf{I}$ ($\kappa > 0$), the target equilibrium $\mathbf{z}_d$ = ($\mathbf{q}_d, \mathbf{0}$) is an isolated strict local minimum of $H_{\phi_d}$ and is uniformly locally asymptotically stable (established via Barbalat's Lemma and Matrosov's Theorem).
- **Practical Stability under Model Mismatch (Theorem 1)**: Let model approximation errors over a compact domain $\mathcal{D}$ satisfy $\|\mathbf{f}_\theta - \mathbf{f}\| \le \epsilon$, $\|\mathbf{G}_\theta - \mathbf{G}\| \le \epsilon_G$, and $\|\mathbf{R}_\theta - \mathbf{R}\| \le \epsilon_R$. Along trajectories of the true closed-loop system:

$$\dot{H}_d(t, \mathbf{z}) \le -\rho \|\mathbf{z} - \mathbf{z}_d\|^2 + \xi + \varepsilon_{\mathrm{diss}}$$

where $\xi = \mathcal{O}(\epsilon, \epsilon_R, \epsilon_G)$. System trajectories are uniformly practically asymptotically stable (uniformly ultimately bounded) and enter the compact sublevel set containing:

$$\{ \mathbf{z} \in \Omega_h : \|\mathbf{z} - \mathbf{z}_d\| \le \sqrt{\frac{\xi + \varepsilon_{\mathrm{diss}}}{\rho}} \}$$

---

## Experimental Evaluations and Summary of Results

### 1. Planar Pendulum
- Evaluated on state-regulation to $(\mathbf{q}^*, \mathbf{p}^*) = (0, 0)$ and swing-up to $(\pm\pi, 0)$.
- The learned pH model accurately reproduces true system dynamics, predicting system trajectories over long evaluation horizons ($2\,\text{s}$ to $3\,\text{s}$) when trained on trajectories of only $0.15\,\text{s}$.
- The learned added potential $V_\phi^*$ reshapes the energy landscape to place the global minimum at the target equilibrium.

### 2. Single-Link Torsional Pendulum
- Evaluated on swing-up with Hamiltonian $H = \frac{1}{2}J\dot{q}^2 + mgr(1 - \cos q) + \frac{1}{2}k q^2$.
- Compared against a gain-optimized classical PD controller with potential compensation (PD+):
  - **Mean control effort**: $4.03$ (OES) vs. $7.10$ (PD+).
  - OES takes advantage of gravitational and elastic spring energy rather than canceling them.
  - The learned damping $\mathbf{K}_\phi^*(t, \mathbf{z})$ exhibits dynamic state- and time-dependent adaptation (low damping early to pump energy; increased damping near target).

### 3. Two-Link Torsional Pendulum with Joint Elasticity
- Higher-dimensional nonlinear system with parallel torsional springs at both joints.
- Compared against a gain-optimized standard EB-PBC that strictly cancels natural potentials and adds a quadratic potential:
  - **Mean total control effort**: $1129.07$ (OES) vs. $1646.14$ (standard EB-PBC) — a **31.4% reduction** in control effort.
  - **Mean terminal position error**: $2.83 \times 10^{-2}$ (OES) vs. $3.72 \times 10^{-3}$ (standard EB-PBC).

### 4. Ablation on Damping Matrix Parameterization
Evaluated on the 2-link torsional pendulum swing-up task ($200$ test trajectories, $T = 5.0\,\text{s}$):

| Damping Matrix Type | Mean Terminal State Error [rad, rad/s] | Mean Total Control Effort [$\text{N}^2\text{m}^2\text{s}$] |
|:---|:---:|:---:|
| $\mathbf{K}_\phi^*(t, \mathbf{z})$ (Proposed) | $\mathbf{1.9279 \times 10^{-4}}$ | $\mathbf{1178.1}$ |
| $\mathbf{K}_\phi^*(\mathbf{z})$ | $5.6659 \times 10^{-4}$ | $1245.5$ |
| $\mathbf{K}_\phi^*$ (Constant) | $1.1668 \times 10^{-3}$ | $1302.1$ |

### 5. Sensitivity Analysis of Dissipation Rate $\rho$ under Model Mismatch
Swing-up transfer performance on the true 2-link torsional pendulum when trained on an approximate model ($\epsilon = 0.005$ with $12.56\%$ stiffness error, $8.17\%$ damping error, $11.31\%$ inertia error, $16.44\%$ potential error):

| $\rho$ | Median $\|\mathbf{z}_e(T)\|$ | Mean $\|\mathbf{z}_e(T)\|$ | Mean $\int \|\mathbf{u}\|^2 \mathrm{d}t$ | Success Rate ($\|\mathbf{q}_e(T)\|_2 < 0.05\,\text{rad}$) |
|:---:|:---:|:---:|:---:|:---:|
| $0.001$ | $1.70$ | $0.87$ | $1192.5$ | $16.2\%$ |
| $0.01$ | $1.62$ | $0.92$ | $1198.5$ | $17.8\%$ |
| $0.03$ | $1.13$ | $0.50$ | $1136.8$ | $50.7\%$ |
| $0.10$ | $0.86$ | $0.58$ | $1224.1$ | $48.5\%$ |
| $\mathbf{0.30}$ | $\mathbf{1.17}$ | $\mathbf{0.28}$ | $\mathbf{1301.2}$ | $\mathbf{84.5\%}$ |
| $1.00$ | $5.23$ | $0.51$ | $1948.6$ | $50.5\%$ |

### 6. Controller and Model Paradigm Benchmarks
Performance comparison for swing-up of the 2-link torsional pendulum across exact and learned models:

| Controller Pipeline / Architecture | Mean Terminal State Error | Mean Total Control Effort |
|:---|:---:|:---:|
| Gain-Optimized EB-PBC on Exact Model | $1.6089 \times 10^{-2}$ | $1617.38$ |
| Gain-Optimized EB-PBC on Learned pH Model | $1.4147 \times 10^{-2}$ | $1711.39$ |
| Neural OES on Exact Model | $6.4036 \times 10^{-2}$ | $1008.33$ |
| Proposed Co-Learning (Neural OES + learned pH) | $1.1773 \times 10^{-1}$ | $1134.07$ |

---

## Neural Network Specifications and Hyperparameters

### Network Topologies
All networks are fully connected with the format $d_{\mathrm{in}} - n_1 f_{\mathrm{act}_1} - \dots - n_N f_{\mathrm{act}_N} - d_{\mathrm{out}}$:

- $\mathbf{M}_{\theta_1}^{-1} = \mathbf{L}_{\theta_1}^\top \mathbf{L}_{\theta_1}$: $2n - 300\,\tanh - 300\,\tanh - 300\,\tanh - n$ (with $+\epsilon \mathbf{I}$, $\epsilon > 0$ for positive definiteness)
- $V_{\theta_2}$: $n - 50\,\tanh - 50\,\tanh - 1$
- $\mathbf{g}_{\theta_3}$: $2n - 400\,\tanh - 400\,\tanh - n$
- $V_\phi^*$: $n - 64\,\text{softplus} - 64\,\text{softplus} - 64\,\tanh - n$
- $\mathbf{K}_\phi^*$: $(2n+1) - 64\,\text{softplus} - 64\,\text{softplus} - n\,\text{softplus}$ (with $+\kappa \mathbf{I}$, $\kappa > 0$ for positive definiteness)

### Training Hyperparameters
- **ODE Solver**: Fourth-order Runge-Kutta (`rk4`)
- **Integration Step**: $10^{-2}$ for system model training; $2 \times 10^{-2}$ for controller training
- **Optimizer**: RAdam
- **Learning Rates**: $10^{-3}$ for neural network parameters; $0.1$ for scalar parameters; weight decay $10^{-4}$
- **Learning Rate Schedule**: Cosine annealing down to minimum learning rate $10^{-6}$ for model training
- **Cost Weights**: $\eta = 10^{-3}$, target variance $\sigma^2 = 10^{-3}$, $\lambda_{\mathrm{diss}} = 1$ (nominal) / $2.0$ (sensitivity study), $\rho = 10^{-3}$ (nominal)
- **Training Trajectory Length**: $T = 0.15\,\text{s}$ (evaluated over $2\,\text{s}$ to $5\,\text{s}$)
- **Initial Condition Distribution**: $\mathbb{P}_{\mathbf{z}_0} \sim \mathbb{U}([-2\pi, 2\pi] \times [-\pi, \pi])$, $512$ trajectories

---

## Citation

```bibtex
@article{kamboj2026colearning,
  author    = {Kamboj, Ankur and Dey, Biswadip and Srivastava, Vaibhav},
  title     = {Co-Learning Port-Hamiltonian Systems and Optimal Energy-Shaping Control},
  journal   = {ASME Journal of Dynamic Systems, Measurement and Control},
  year      = {2026}
}
```
