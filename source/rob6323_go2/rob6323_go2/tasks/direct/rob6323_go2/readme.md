# Reward-Shaped PPO for Symmetric and Stable Quadrupedal Locomotion

While the tutorial was not sufficient for the robot dogs to move around, it was jittery and the baseline tutorial was not optimised for the robot to learn accurately.

This project extends a basic DirectRLEnv implementation for the Unitree Go2 robot in Isaac Lab. The goal is to transform a simple kinematic learner into a robust, research-grade locomotion controller.

Moving beyond simple velocity tracking, this environment implements custom low-level control, history-based smoothness penalties, and the Raibert Heuristic for precise gait generation.

# Key Features & Implementation Details

1. Symmetry-Enforced Learning
Why: Standard RL often learns asymmetric or "limping" gaits that are optimal in simulation but undesirable in reality. We enforce symmetry to produce natural, robust trotting.

2. Realistic Actuator Dynamics (Friction)
Why: The basic PD controller assumes ideal motors. Real actuators have significant friction, leading to the "reality gap." Implementation:

3. Enhanced Gait Shaping
Why: The basic Raibert heuristic provides foot targets, but doesn't guarantee the robot actually lifts its legs or commits to a schedule.

# Rob6323 GO2 Locomotion Environment — Extended Implementation

This repository builds on the **Isaac Lab GO2 baseline environment** and extends it with physically grounded actuation, gait-aware rewards, contact scheduling, and symmetry regularization. The goal is to move from a minimal velocity-tracking baseline to a **stable, expressive quadruped locomotion controller** suitable for research and coursework (ROB 6323).


## 1. Baseline (Provided)

The baseline environment (`Rob6323Go2Env`) implements:

* Position-controlled joints via Isaac Lab’s implicit PD controller
* Commanded base velocity tracking (XY linear + yaw)
* Simple exponential tracking rewards
* Episode termination on base collision or upside-down orientation
* Debug visualization of commanded vs. actual velocity

This baseline serves as a clean reference point for all modifications.


# Major Changes Overview

Below is summary of the changes

### 1. Explicit Torque Control with Custom PD

**What changed**

* Disabled Isaac Lab’s built-in joint PD gains
* Implemented explicit torque control:
  $$tau = K_p (q_d - q) - K_d \dot{q} $$
* Applied torques directly via `set_joint_effort_target`

**Why**

* Gives full control over low-level dynamics
* Required for modeling actuator friction
* Matches hardware-oriented control pipelines

**Where**

* `final_cfg.py`: implicit actuator gains set to zero
* `final_env.py`: `_apply_action()`

---

### 2. Actuator Friction Model (New File)

**What changed**

* Added viscous + stiction friction model per actuator
* Randomized friction parameters per environment

**Why**

* Improves realism
* Prevents overly optimistic policies
* Encourages robust locomotion strategies

**Where**

* `actuator_friction.py`
* Applied inside `_apply_action()`

---

### 3. Expanded Observation Space (Gait-Aware)

**What changed**

* Added **clock / phase inputs** (sinusoidal gait encoding)
* Observation now includes gait timing information

**Why**

* Helps the policy learn phase-dependent behaviors
* Enables foot-specific swing/stance coordination

**Where**

* `final_env.py`: `_get_observations()`

---

### 4. Contact Sensor Integration

**What changed**

* Explicit indexing of foot bodies in both:

  * Articulation (kinematics)
  * ContactSensor (forces & contact state)

**Why**

* Enables contact-aware rewards
* Required for airtime, contact scheduling, and symmetry terms

**Where**

* `__init__()` in `final_env.py`

---

### 5. Gait Phase & Contact Scheduling

**What changed**

* Implemented internal gait phase variable
* Generated desired contact states per foot
* Smooth phase transitions using probabilistic shaping

**Why**

* Encourages structured locomotion
* Avoids random footfall patterns
* Provides weak supervision without hard constraints

**Where**

* `_step_contact_targets()`

---

### 6. Raibert Heuristic Reward

**What changed**

* Penalizes deviation from Raibert-style foot placement
* Foot targets depend on commanded velocity and gait phase

**Why**

* Classic stabilizing heuristic for legged locomotion
* Helps early-stage learning and reduces falls

**Where**

* `_reward_raibert_heuristic()`

---

### 7. Contact-Consistent Force Shaping

**What changed**

* Reward high forces during stance
* Penalize forces during swing
* Uses smooth saturation (exponential shaping)

**Why**

* Prevents foot scuffing
* Encourages clean liftoff and landing

**Where**

* `_reward_tracking_contacts_shaped_force()`

---

### 8. Feet Clearance & Airtime Rewards

**What changed**

* Phase-shaped foot height targets during swing
* Reward airtime at touchdown

**Why**

* Encourages stepping instead of shuffling
* Improves obstacle robustness and stability

**Where**

* `_reward_feet_clearance()`
* `_reward_air_time()`

---

### 9. Orientation & Motion Regularization

**What changed**

* Penalized:

  * Roll/pitch deviation
  * Vertical velocity
  * Excess joint velocity
  * Roll/pitch angular velocity

**Why**

* Keeps motion physically plausible
* Reduces jitter and instability

**Where**

* `_get_rewards()`

---

### 10. Symmetry Rewards (Advanced)

#### 10.1 Temporal Symmetry

**What**

* Enforces stance/swing consistency under time reversal
* Uses GRF during swing and foot velocity during stance

**Why**

* Encourages periodic, reversible gaits
* Inspired by recent symmetry-based locomotion papers

**Where**

* `_reward_sym_temporal()`

#### 10.2 Morphological Symmetry

**What**

* Penalizes asymmetric joint configurations
* Applies only to phase-compatible leg pairs

**Why**

* Encourages left–right balance
* Reduces limping and biased gaits

**Where**

* `_reward_sym_morphological()`

---

### 11. Termination Conditions

**Added**

* Base height termination (robot crouching or collapsing)

**Why**

* Prevents exploiting low-height configurations

**Where**

* `_get_dones()`

---

### 12. Configuration Changes Summary

Key additions in `final_cfg.py`:

* Explicit PD gains (`Kp`, `Kd`)
* Torque limits
* Friction randomization ranges
* Expanded reward scales
* Debug visualization enabled by default

---



Ensure Isaac Lab is installed and sourced correctly.

---


(Exact command depends on your Isaac Lab training entrypoint.)


### 13. Debug Visualization


You will see:

* Green arrow: commanded velocity
* Blue arrow: actual base velocity

---

## 14. Expected Outcomes

Compared to the baseline, this implementation:

* Learns faster and more stably
* Produces smoother, more symmetric gaits
* Exhibits clear stance–swing structure
* Is robust to actuator friction and contact noise

---

## 15. Acknowledgements

* Isaac Lab Project Developers
* Unitree GO2 model
* Raibert legged locomotion heuristics
* Symmetry-based locomotion literature

---