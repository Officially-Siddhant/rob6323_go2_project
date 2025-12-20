# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import gymnasium as gym
import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
import isaaclab.utils.math as math_utils

from .rob6323_go2_env_cfg import Rob6323Go2EnvCfg
from .actuator_friction import apply_actuator_friction


class Rob6323Go2Env(DirectRLEnv):
    cfg: Rob6323Go2EnvCfg

    def __init__(self, cfg: Rob6323Go2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space),
                                             device=self.device)

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                # Tutorial rewards
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "rew_action_rate",
                "raibert_heuristic",
                "orient",
                "lin_vel_z",
                "dof_vel",
                "ang_vel_xy",
                "feet_clearance",
                "tracking_contacts_shaped_force",

                # Contact Schedule Tracking
                "face_command",
                "contact_schedule",
                "air_time",
                "pitch_nose_down",

                # Symmetry Related
                "sym_tem",
                "sym_mor",
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        # self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
        # self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*thigh")

        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

        # Robot indexing (positions/kinematics)
        self._feet_ids = []
        for name in foot_names:
            ids, _ = self.robot.find_bodies(name)
            self._feet_ids.append(ids[0])

        # ContactSensor indexing (forces/contact history)
        self._feet_ids_sensor = []
        for name in foot_names:
            ids, _ = self._contact_sensor.find_bodies(name)
            self._feet_ids_sensor.append(ids[0])

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        # keep a track of the last few actions
        self.last_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), 3,
                                        dtype=torch.float, device=self.device, requires_grad=False)

        # PD control parameters
        self.Kp = torch.tensor([cfg.Kp] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.Kd = torch.tensor([cfg.Kd] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.motor_offsets = torch.zeros(self.num_envs, 12, device=self.device)
        self.torque_limits = cfg.torque_limits

        # Actuator friction params

        # Store per-env scalars (broadcast across 12 joints)
        self.mu_v = torch.zeros(self.num_envs, 1, device=self.device)  # viscous coeff
        self.Fs = torch.zeros(self.num_envs, 1, device=self.device)  # stiction coeff

        # Keeping last computed friction for debugging/logging
        self._tau_friction = torch.zeros(self.num_envs, 12, device=self.device)

        # Constants from the assignment
        self._stiction_vel_scale = cfg.friction_vel_scale

        # Variables needed for the raibert heuristic
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                  requires_grad=False)

        # For foot contact and airtime buffers
        self.feet_swing_time = torch.zeros(self.num_envs, 4, device=self.device)  # seconds
        self.prev_contact = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self.contacts_touchdown = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.last_dof_vel = torch.zeros(self.num_envs, 12, device=self.device)  # last dof velocities

        # ----------------------------
        #       Symmetry reward
        # ----------------------------

        self._beta = 0.5  # duty factor β
        self._eps_sigma = 0.10  # εσ threshold for "same-phase" pairing
        self._w_tem = 0.15  # paper coefficient (you can move to cfg)
        self._w_mor = 0.15

        # phase offsets θ_i[FL, FR, RL, RR]
        self._theta = torch.tensor([0.0, 0.5, 0.5, 0.0], device=self.device).unsqueeze(0)  # (1,4)
        # Lateral Sequence: [0.00, 0.25, 0.50, 0.75]
        # Trotting Sequence: [0.50, 0.00, 0.00, 0.50]
        # Bounding Sequence: [0.50, 0.50, 0.00, 0.00]
        # Half Bounding Left: [0.63, 0.37, 0.00, 0.00]
        # Half Bounding Right: [0.37, 0.63, 0.00, 0.00]
        # Rotary Gallop: [0.37, 0.63, 0.13, 0.87]
        # Transverse Gallop: [0.63, 0.37, 0.13, 0.87]

        # For foot velocity if body_lin_vel_w isn't available (finite difference fallback)
        self._prev_foot_pos_w = torch.zeros(self.num_envs, 4, 3, device=self.device)

        # Joint indices per leg for morphological symmetry
        leg_names = ["FL", "FR", "RL", "RR"]
        joint_triplets = []
        for leg in leg_names:
            hip_ids, _ = self.robot.find_joints(f"{leg}_hip_joint")
            thigh_ids, _ = self.robot.find_joints(f"{leg}_thigh_joint")
            calf_ids, _ = self.robot.find_joints(f"{leg}_calf_joint")
            joint_triplets.append([hip_ids[0], thigh_ids[0], calf_ids[0]])

        # 4 legs, 3 joints
        self._leg_joint_ids = torch.tensor(joint_triplets, device=self.device, dtype=torch.long)

        # For temporal symmetry reward
        self._k_f = 2e-4  # start softer than 0.001
        self._k_v = 0.5  # start softer than 2.0

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        # self._processed_actions = self.cfg.action_scale * self._actions + self.robot.data.default_joint_pos
        # Compute desired joint positions from policy actions
        self.desired_joint_pos = (
                self.cfg.action_scale * self._actions
                + self.robot.data.default_joint_pos
        )

    def _apply_action(self) -> None:
        # PD torques
        tau_pd = (self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos) - self.Kd * self.robot.data.joint_vel)

        # Actuator friction
        tau_cmd, tau_friction = apply_actuator_friction(tau_pd=tau_pd, qd=self.robot.data.joint_vel, Fs=self.Fs,
                                                        mu_v=self.mu_v, vel_scale=self._stiction_vel_scale, )

        # For logging and debugging
        self._tau_friction[:] = tau_friction

        # Clip and Apply
        torques = torch.clip(tau_cmd, -self.torque_limits, self.torque_limits)
        self.robot.set_joint_effort_target(torques)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        obs = torch.cat(
            [
                tensor
                for tensor in (
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_ang_vel_b,
                self.robot.data.projected_gravity_b,
                self._commands,
                self.robot.data.joint_pos - self.robot.data.default_joint_pos,
                self.robot.data.joint_vel,
                self._actions,
                self.clock_inputs,  # added gait inputs
            )
                if tensor is not None
            ],
            dim=-1,
        )
        self.last_dof_vel[:] = self.robot.data.joint_vel[:]  # Update last dof velocities
        observations = {"policy": obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:

        # -----------------------------
        # Command tracking rewards
        # -----------------------------
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)

        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

        # -----------------------------
        # Action smoothness penalty
        # -----------------------------
        # action rate penalization
        # First derivative (Current - Last)
        rew_action_rate = torch.sum(torch.square(self._actions - self.last_actions[:, :, 0]), dim=1) * (
                self.cfg.action_scale ** 2)

        # Second derivative (Current - 2*Last + 2ndLast)
        rew_action_rate += torch.sum(
            torch.square(self._actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1) * (
                                       self.cfg.action_scale ** 2)

        # Update the prev action hist (roll buffer and insert new action)
        self.last_actions = torch.roll(self.last_actions, 1, 2)
        self.last_actions[:, :, 0] = self._actions[:]

        self._step_contact_targets()
        self._update_foot_contact_timing()  # updating foot contact timing

        # Calculating air time before
        rew_air_time = self._reward_air_time()
        rew_contact_schedule = self._reward_contact_schedule_tracking()

        # -----------------------------
        # Raibert heuristic penalty
        # -----------------------------
        rew_raibert_heuristic = self._reward_raibert_heuristic()

        # Foot clearance penalty (swing feet should lift)
        rew_feet_clearance = self._reward_feet_clearance()

        # Face command reward (robot should face the commanded heading)
        rew_face_cmd = self._reward_face_command()

        # "Keep Head Up" penalty
        rew_pitch_nose_down = torch.clamp(self.robot.data.projected_gravity_b[:, 0], min=0.0) ** 2

        # Contact force tracking reward (stance feet should load, swing feet should unload)
        rew_contact_forces = self._reward_tracking_contacts_shaped_force()

        # -----------------------------
        # Pose / motion shaping penalties
        # -----------------------------
        # 1. Penalize non-vertical orientation
        rew_orient = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, 0:2]), dim=1)

        # 2. Penalize vertical velocity
        rew_lin_vel_z = torch.square(self.robot.data.root_lin_vel_b[:, 2])

        # 3. Penalize high joint velocities
        rew_dof_vel = torch.sum(torch.square(self.robot.data.joint_vel), dim=1)

        # 4. Penalize angular velocity in XY plane (roll/pitch)
        rew_ang_vel_xy = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, 0:2]), dim=1)

        # 5. Symmetry reward
        rew_sym_tem = self._reward_symmetry_temporal()
        rew_sym_mor = self._reward_symmetry_morphological()

        rewards = {
            # Tutorial rewards
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale,
            "rew_action_rate": rew_action_rate * self.cfg.action_rate_reward_scale,
            "raibert_heuristic": rew_raibert_heuristic * self.cfg.raibert_heuristic_reward_scale,
            "orient": rew_orient * self.cfg.orient_reward_scale,
            "lin_vel_z": rew_lin_vel_z * self.cfg.lin_vel_z_reward_scale,
            "dof_vel": rew_dof_vel * self.cfg.dof_vel_reward_scale,
            "ang_vel_xy": rew_ang_vel_xy * self.cfg.ang_vel_xy_reward_scale,
            "feet_clearance": rew_feet_clearance * self.cfg.feet_clearance_reward_scale,
            "tracking_contacts_shaped_force": rew_contact_forces * self.cfg.tracking_contacts_shaped_force_reward_scale,

            # Contact Schedule Tracking
            "face_command": rew_face_cmd * self.cfg.face_command_reward_scale,
            "contact_schedule": rew_contact_schedule * self.cfg.contact_schedule_reward_scale,
            "air_time": rew_air_time * self.cfg.air_time_reward_scale,
            "pitch_nose_down": rew_pitch_nose_down * self.cfg.pitch_nose_down_reward_scale,

            # Symmetry Related
            "sym_tem": rew_sym_tem * self.cfg.sym_tem_reward_scale,
            "sym_mor": rew_sym_mor * self.cfg.sym_mor_reward_scale,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        cstr_termination_contacts = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        cstr_upsidedown = self.robot.data.projected_gravity_b[:, 2] > 0

        base_height = self.robot.data.root_pos_w[:, 2]
        cstr_base_height_min = base_height < self.cfg.base_height_min

        # Applying the terminations
        died = cstr_termination_contacts | cstr_upsidedown | cstr_base_height_min
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0

        self.last_actions[env_ids] = 0.

        self._previous_actions[env_ids] = 0.0

        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)

        # Reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.feet_swing_time[env_ids] = 0.0
        self.prev_contact[env_ids] = False
        self.contacts_touchdown[env_ids] = 0.0
        self.mu_v[env_ids] = 0.0
        self.Fs[env_ids] = 0.0
        self.gait_indices[env_ids] = 0  # Reset raibert quantity
        self._prev_foot_pos_w[env_ids] = self.foot_positions_w[env_ids]  # Reset foot position buffer for symmetry reward

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._commands[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

    @property
    def foot_positions_w(self) -> torch.Tensor:
        """Returns the feet positions in the world frame.
        Shape: (num_envs, num_feet, 3)
        """
        return self.robot.data.body_pos_w[:, self._feet_ids]

    # Defines contact plan
    def _step_contact_targets(self):
        frequencies = 3.
        phases = 0.5
        offsets = 0.
        bounds = 0.
        durations = 0.5 * torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        self.gait_indices = torch.remainder(self.gait_indices + self.step_dt * frequencies, 1.0)

        foot_indices = [self.gait_indices + phases + offsets + bounds,
                        self.gait_indices + offsets,
                        self.gait_indices + bounds,
                        self.gait_indices + phases]

        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations

            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                    0.5 / (1 - durations[swing_idxs]))

        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

        # von mises distribution
        kappa = 0.07
        smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                                kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

        smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                   smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                           1 - smoothing_cdf_start(
                                       torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
        smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                   smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                           1 - smoothing_cdf_start(
                                       torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                   smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                           1 - smoothing_cdf_start(
                                       torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                   smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                           1 - smoothing_cdf_start(
                                       torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

        self.desired_contact_states[:, 0] = smoothing_multiplier_FL
        self.desired_contact_states[:, 1] = smoothing_multiplier_FR
        self.desired_contact_states[:, 2] = smoothing_multiplier_RL
        self.desired_contact_states[:, 3] = smoothing_multiplier_RR

    # Raibert Heuristic

    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.foot_positions_w - self.robot.data.root_pos_w.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = math_utils.quat_apply_yaw(
                math_utils.quat_conjugate(self.robot.data.root_quat_w),
                cur_footsteps_translated[:, i, :])

        # nominal positions: [FR, FL, RR, RL]
        desired_stance_width = 0.25
        desired_ys_nom = torch.tensor(
            [desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2],
            device=self.device).unsqueeze(0)

        desired_stance_length = 0.45
        desired_xs_nom = torch.tensor([desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2,
                                       -desired_stance_length / 2], device=self.device).unsqueeze(0)

        # Raibert offsets
        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = torch.tensor([3.0], device=self.device)
        x_vel_des = self._commands[:, 0:1]
        yaw_vel_des = self._commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward

    def _reward_feet_clearance(self) -> torch.Tensor:
        """
        Phase-shaped foot height target (legacy CaT-style):
        target_z = A * phase + z0, only during swing (1 - desired_contact_states).
        Penalize (target_z - foot_z)^2.
        """
        # (num_envs, 4)
        foot_z = self.foot_positions_w[:, :, 2]

        # foot phase in [0,1] per foot (you already compute this in _step_contact_targets)
        # self.foot_indices: (num_envs, 4)
        fi = self.foot_indices

        # legacy phase shaping:
        # phases = 1 - |1 - clip((fi*2 - 1),0,1)*2|
        x = torch.clamp(fi * 2.0 - 1.0, 0.0, 1.0) * 2.0
        phases = 1.0 - torch.abs(1.0 - x)  # (num_envs, 4), peaks mid-swing

        # target foot height profile (tune these)
        A = 0.10  # swing amplitude (m) # was 0.08
        z0 = 0.03  # offset (m) ~ “foot radius” margin # was 0.02
        target_z = A * phases + z0  

        # only care during swing (desired_contact_states ~1 in stance)
        swing = 1.0 - self.desired_contact_states

        # err = target_z - foot_z
        # err = torch.clamp(target_z - foot_z, min=0.0)

        # penalize low more than high, but don't ignore high entirely
        low_err = torch.clamp(target_z - foot_z, min=0.0)
        high_err = torch.clamp(foot_z - (target_z + 0.03), min=0.0)  # 3cm headroom

        # per_foot = (err * err) * swing
        per_foot = (low_err ** 2 + 0.2 * high_err ** 2) * swing
        return torch.sum(per_foot, dim=1) 

    def _reward_tracking_contacts_shaped_force(self) -> torch.Tensor:
        """Reward: high contact forces in stance, low forces in swing."""
        # net forces in world frame: (num_envs, num_sensor_bodies, 3)
        forces_w = self._contact_sensor.data.net_forces_w

        # pick feet using CONTACT SENSOR indices -> (num_envs, 4, 3)
        foot_forces_w = forces_w[:, self._feet_ids_sensor, :]
        foot_force_mag = torch.linalg.norm(foot_forces_w, dim=-1)  

        # desired contact in [0,1] (1 = stance, 0 = swing)
        c_des = self.desired_contact_states

        # shaping scale: how quickly "high force" saturates
        f_scale = 25.0  # tune; units depend on sim/contact scaling

        # smooth "contact present" in [0,1]
        contact_present = 1.0 - torch.exp(-(foot_force_mag / f_scale) ** 2)

        # stance: want contact_present -> 1
        # swing:  want contact_present -> 0
        per_foot = c_des * contact_present + (1.0 - c_des) * (1.0 - contact_present)

        # sum over feet -> (num_envs,)
        return torch.sum(per_foot, dim=1)

    def _reward_stride_length(self) -> torch.Tensor:
        """
        Penalty: during swing, foot x-position in yaw-aligned body frame should follow
        a simple phase-shaped stride target (larger command -> larger target stride).
        We directly ask the foot to move forward in body x, not just lift in z, so hips/thighs have incentive to participate.
        Returns (num_envs,) positive cost.
        """
        # foot positions relative to base in world
        rel_w = self.foot_positions_w - self.robot.data.root_pos_w.unsqueeze(1)  

        # rotate into yaw-aligned body frame (so "x" means forward of robot)
        rel_b = torch.zeros_like(rel_w)
        qyaw_inv = math_utils.quat_conjugate(self.robot.data.root_quat_w)  
        for i in range(4):
            rel_b[:, i, :] = math_utils.quat_apply_yaw(qyaw_inv, rel_w[:, i, :])

        x_foot = rel_b[:, :, 0]  # (N,4)

        # phase in [0,1] per foot (already computed in _step_contact_targets)
        fi = self.foot_indices
        x = torch.clamp(fi * 2.0 - 1.0, 0.0, 1.0) * 2.0
        phase_peak = 1.0 - torch.abs(1.0 - x)  # 0..1, peaks mid-swing

        swing = 1.0 - self.desired_contact_states  

        # stride target grows with commanded speed magnitude
        cmd_xy = self._commands[:, :2]
        speed = torch.linalg.norm(cmd_xy, dim=1, keepdim=True)  
        speed = torch.clamp(speed, 0.0, 1.0)

        # tune these
        stride_min = 0.03  # m
        stride_max = 0.12  # m
        stride = stride_min + (stride_max - stride_min) * speed  

        # nominal stance x targets (FR, FL, RR, RL) matching your Raibert ordering
        # If your foot ordering is [FL, FR, RL, RR], swap this array accordingly.
        x_nom = torch.tensor([0.225, 0.225, -0.225, -0.225], device=self.device).unsqueeze(0)  

        # aim for forward peak during swing; sign uses commanded forward component
        sgn = torch.sign(self._commands[:, 0:1])  
        target_x = x_nom + sgn * stride * phase_peak  

        err = target_x - x_foot
        return torch.sum((err * err) * swing, dim=1)

    def _wrap_to_pi(self, ang: torch.Tensor) -> torch.Tensor:
        return (ang + torch.pi) % (2 * torch.pi) - torch.pi

    def _reward_face_command(self) -> torch.Tensor:
        """
        Penalty: base yaw should align with the *world* direction of the commanded velocity.
        Returns (num_envs,) positive cost.
        """
        cmd_b = torch.zeros(self.num_envs, 3, device=self.device)
        cmd_b[:, :2] = self._commands[:, :2]

        # command direction expressed in world (using current base yaw)
        cmd_w = math_utils.quat_apply_yaw(self.robot.data.root_quat_w, cmd_b)  
        cmd_dir = cmd_w[:, :2]
        cmd_norm = torch.linalg.norm(cmd_dir, dim=1)

        # desired heading in world
        des_yaw = torch.atan2(cmd_dir[:, 1], cmd_dir[:, 0])

        # current heading = base forward axis in world
        fwd_b = torch.zeros(self.num_envs, 3, device=self.device)
        fwd_b[:, 0] = 1.0
        fwd_w = math_utils.quat_apply_yaw(self.robot.data.root_quat_w, fwd_b)
        cur_yaw = torch.atan2(fwd_w[:, 1], fwd_w[:, 0])

        yaw_err = self._wrap_to_pi(des_yaw - cur_yaw)

        # Ignore tiny commands
        mask = (cmd_norm > 0.1).float()
        return (yaw_err * yaw_err) * mask

    def _update_foot_contact_timing(self):
        # Updates feet_swing_time and contacts_touchdown using ContactSensor forces.
        forces_w = self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, :] 
        foot_force_mag = torch.linalg.norm(forces_w, dim=-1)  # (N,4)

        # contact if above threshold (tune threshold if needed)
        contact = foot_force_mag > 1.0  

        # touchdown event: 0 to 1 transition
        touchdown = (~self.prev_contact) & contact  
        self.contacts_touchdown[:] = touchdown.float()

        # swing = not in contact
        swing = ~contact  # boolean

        # accumulate swing time, reset on contact
        self.feet_swing_time[swing] += self.step_dt
        self.feet_swing_time[contact] = 0.0

        # update prev
        self.prev_contact[:] = contact

    def _reward_contact_schedule_tracking(self) -> torch.Tensor:
        """
        Penalize mismatch between desired_contact_states (0..1) and measured contact (0/1).
        Returns (num_envs,) positive cost.
        """
        forces_w = self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, :]
        foot_force_mag = torch.linalg.norm(forces_w, dim=-1)  
        contact = (foot_force_mag > 1.0).float()  

        c_des = self.desired_contact_states  
        err = contact - c_des
        return torch.sum(err * err, dim=1)

    def _reward_air_time(self) -> torch.Tensor:
        """
        Reward longer swing durations, but only when a touchdown happens.
        Returns (num_envs,) reward (can be positive).
        """
        # Get current contact status directly from sensors
        forces_w = self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, :]
        contact = torch.linalg.norm(forces_w, dim=-1) > 1.0  

        # Detect landing: Currently touching ground AND wasn't touching previously
        # We use self.prev_contact because it hasn't been updated yet!
        first_contact = contact & (~self.prev_contact)

        # Reward accumulated time for the feet that just landed
        return torch.sum((self.feet_swing_time - 0.25) * first_contact.float(), dim=1)

    # Symmetry Rewards 

    def _leg_phase_time_reversal(self) -> torch.Tensor:
        """
        Implements paper's time-reversal phase remapping:
          φ_i = (φ + θ_i) mod 1            if v_x_cmd >= 0
          φ_i = -(φ + θ_i) mod 1           if v_x_cmd < 0
        where φ is the global gait phase in [0,1).
        """
        phi = self.gait_indices.unsqueeze(1)  
        phi_i = torch.remainder(phi + self._theta, 1.0)  

        backward = (self._commands[:, 0:1] < 0.0) 
        # "−(φ+θ) mod 1" is equivalent to remainder( -(...), 1 )
        phi_i_back = torch.remainder(-phi_i, 1.0)

        return torch.where(backward, phi_i_back, phi_i)

    def _stance_swing_indicators(self, phi_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (I_stance, I_swing) in {0,1} using duty factor β: stance if φ_i < β else swing
        """
        I_stance = (phi_i < self._beta).float()
        I_swing = 1.0 - I_stance
        return I_stance, I_swing

    def _reward_sym_temporal(self) -> torch.Tensor:
        """
        R_tem = -0.15 * sum_i ( E[I_swing(φ_i)](1-exp(-0.001||f_i||))
                             + E[I_stance(φ_i)](1-exp(-2||v_i||)) )
        Here:
          f_i = GRF magnitude for leg i (from ContactSensor net forces)
          v_i = foot speed magnitude (from body lin vel if available; else finite diff)
        """
        # time-reversal phase remap + stance/swing indicators
        phi_i = self._leg_phase_time_reversal()
        I_stance, I_swing = self._stance_swing_indicators(phi_i)

        # --- GRF magnitude (N,4)
        forces_w = self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, :] 
        f_mag = torch.linalg.norm(forces_w, dim=-1)

        # --- foot speed magnitude (N,4)
        if hasattr(self.robot.data, "body_lin_vel_w"):
            foot_vel_w = self.robot.data.body_lin_vel_w[:, self._feet_ids, :]  
            v_mag = torch.linalg.norm(foot_vel_w, dim=-1)
        else:
            # fallback: finite-difference foot positions
            foot_pos = self.foot_positions_w  
            foot_vel = (foot_pos - self._prev_foot_pos_w) / max(self.step_dt, 1e-6)
            self._prev_foot_pos_w[:] = foot_pos
            v_mag = torch.linalg.norm(foot_vel, dim=-1)

        term_swing = I_swing * (1.0 - torch.exp(-self._k_f * f_mag))
        term_stance = 0.3 * I_stance * (1.0 - torch.exp(-self._k_v * v_mag))

        cost = torch.sum(term_swing + term_stance, dim=1)

        return -self._w_tem * cost

    def _reward_sym_morphological(self) -> torch.Tensor:
        """
        Paper-style morphological symmetry term:
        R_mor = -0.15 * (1 - exp(-5 * d(Gσ)))
        d(Gσ) = sum_{(i,j) in Gσ} f(σ(i,j)) * sum_{k in {hip,thigh,calf}} |q_{i,k} - q_{j,k}|
        f(σ(i,j)) = 1 if |θ_i - θ_j| <= εσ else 0

        We use fixed candidate symmetric pairs; gating decides if they're "active".
        """
        # joint positions 
        q = self.robot.data.joint_pos

        # gather per-leg joint vectors (N,4,3): [hip, thigh, calf] for each leg
        # self._leg_joint_ids: (4,3)
        q_leg = q[:, self._leg_joint_ids]  

        # candidate symmetry graph pairs (choose a simple set)
        # diagonal pairs: (FL,RR) and (FR,RL)
        pairs = [(0, 3), (1, 2)]

        # gate f(σ(i,j)) using |θ_i - θ_j| <= εσ (wrap-aware in [0,1))
        theta = self._theta  
        d = torch.zeros(self.num_envs, device=self.device)

        for (i, j) in pairs:
            dtheta = torch.abs(theta[:, i] - theta[:, j])  
            dtheta = torch.minimum(dtheta, 1.0 - dtheta)  
            gate = (dtheta <= self._eps_sigma).float()  
            gate = gate.expand(self.num_envs)  

            diff = torch.abs(q_leg[:, i, :] - q_leg[:, j, :])  
            d += gate * torch.sum(diff, dim=1)

        return -self._w_mor * (1.0 - torch.exp(-5.0 * d))  