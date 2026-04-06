import math
import numpy as np
import os
from .utils.mygym import convert_to_gym
import gym
import opensim
import random
from .osim import OsimEnv

# ---------------------------------------------------------------------------
# Physiological joint angle limits (radians)
# Shoulder: roughly -30° to 120° of flexion in the 2-D sagittal plane
# Elbow:    0° (full extension) to ~135° of flexion
# ---------------------------------------------------------------------------
SHOULDER_MIN, SHOULDER_MAX = -0.5236, 2.0944   # -30° to 120°
ELBOW_MIN,    ELBOW_MAX    =  0.0,    2.3562   #   0° to 135°

# Reward shaping weights — tune as needed
W_EFFORT     = 0.01    # penalty per unit of total squared muscle activation
W_SMOOTHNESS = 0.001   # penalty per unit of total squared joint velocity
W_JOINT_LIM  = 0.5     # penalty per unit of joint limit violation (squared)


def _range_violation(val: float, low: float, high: float, margin: float = 0.1) -> float:
    """
    Returns a squared soft-barrier penalty when `val` is within `margin`
    of (or outside) the joint limit.  Zero when comfortably inside range.
    """
    if val < low + margin:
        return (val - low - margin) ** 2
    if val > high - margin:
        return (val - high + margin) ** 2
    return 0.0


class Arm2DEnv(OsimEnv):
    model_path = os.path.join(os.path.dirname(__file__), '../models/arm2dof6musc.osim')
    time_limit = 200
    target_x = 0
    target_y = 0

    # ------------------------------------------------------------------
    # State dictionary (used for logging / analysis)
    # ------------------------------------------------------------------
    def get_d_state(self, action):
        state_desc = self.get_state_desc()
        d = {}

        for body_part in ["r_ulna_radius_hand"]:
            d[f'{body_part}_pos_0']  = state_desc["body_pos"][body_part][0]
            d[f'{body_part}_pos_1']  = state_desc["body_pos"][body_part][1]
            d[f'{body_part}_vel_0']  = state_desc["body_vel"][body_part][0]
            d[f'{body_part}_vel_1']  = state_desc["body_vel"][body_part][1]
            d[f'{body_part}_acc_0']  = state_desc["body_acc"][body_part][0]
            d[f'{body_part}_acc_1']  = state_desc["body_acc"][body_part][1]

        for joint in ["r_shoulder", "r_elbow"]:
            d[f'{joint}_pos'] = state_desc["joint_pos"][joint]
            d[f'{joint}_vel'] = state_desc["joint_vel"][joint]
            d[f'{joint}_acc'] = state_desc["joint_acc"][joint]

        for muscle in sorted(state_desc["muscles"].keys()):
            d[f'{muscle}_activation']    = state_desc["muscles"][muscle]["activation"]
            d[f'{muscle}_fiber_length']  = state_desc["muscles"][muscle]["fiber_length"]
            d[f'{muscle}_fiber_velocity']= state_desc["muscles"][muscle]["fiber_velocity"]

        d["markers_0"] = state_desc["markers"]["r_radius_styloid"]["pos"][0]
        d["markers_1"] = state_desc["markers"]["r_radius_styloid"]["pos"][1]

        return d

    # ------------------------------------------------------------------
    # Observation vector fed to the RL agent
    # Includes fiber length + velocity so the agent can sense
    # muscle state and learn coordinated, physiologically realistic
    # recruitment rather than brute-force co-contraction.
    #
    # Observation layout (28 values total):
    #   [0-1]   target x, y
    #   [2-7]   r_shoulder pos/vel/acc  (each a 1-element list)
    #   [8-13]  r_elbow    pos/vel/acc
    #   [14-31] 6 muscles × 3 (activation, fiber_length, fiber_velocity)
    #   [32-33] wrist marker x, y
    # ------------------------------------------------------------------
    def get_observation(self):
        state_desc = self.get_state_desc()

        res = [self.target_x, self.target_y]

        for joint in ["r_shoulder", "r_elbow"]:
            res += state_desc["joint_pos"][joint]
            res += state_desc["joint_vel"][joint]
            res += state_desc["joint_acc"][joint]

        for muscle in sorted(state_desc["muscles"].keys()):
            res += [state_desc["muscles"][muscle]["activation"]]
            res += [state_desc["muscles"][muscle]["fiber_length"]]
            res += [state_desc["muscles"][muscle]["fiber_velocity"]]

        res += state_desc["markers"]["r_radius_styloid"]["pos"][:2]

        return res

    def get_observation_space_size(self):
        # 2 target + 2 joints×3 features + 6 muscles×3 features + 2 marker = 34
        return 34

    # ------------------------------------------------------------------
    # Target generation
    # FIX: removed hardcoded overrides that forced every episode to the
    # same fixed point, preventing the agent from learning to generalise.
    # Set env var FIXED_TARGET=1 to restore the old single-target mode
    # for quick debugging.
    # ------------------------------------------------------------------
    def generate_new_target(self):
        if os.getenv("FIXED_TARGET"):
            self.target_x = 0.16056636337579086
            self.target_y = 0.49340151308159397
        else:
            theta          = random.uniform(0, math.pi * 2 / 3)
            radius         = random.uniform(0.3, 0.65)
            self.target_x  = math.cos(theta) * radius
            self.target_y  = -math.sin(theta) * radius + 0.8

        print('\ntarget: [{:.4f} {:.4f}]'.format(self.target_x, self.target_y))

        state = self.osim_model.get_state()
        self.target_joint.getCoordinate(1).setValue(state, self.target_x, False)
        self.target_joint.getCoordinate(2).setLocked(state, False)
        self.target_joint.getCoordinate(2).setValue(state, self.target_y, False)
        self.target_joint.getCoordinate(2).setLocked(state, True)
        self.osim_model.set_state(state)

    def reset(self, random_target=True, obs_as_dict=True):
        obs = super(Arm2DEnv, self).reset(obs_as_dict=obs_as_dict)
        if random_target:
            self.generate_new_target()
        self.osim_model.reset_manager()
        return obs

    def __init__(self, *args, **kwargs):
        super(Arm2DEnv, self).__init__(*args, **kwargs)
        blockos = opensim.Body('target', 0.0001, opensim.Vec3(0), opensim.Inertia(1, 1, .0001, 0, 0, 0))
        self.target_joint = opensim.PlanarJoint(
            'target-joint',
            self.osim_model.model.getGround(),
            opensim.Vec3(0, 0, 0),
            opensim.Vec3(0, 0, 0),
            blockos,
            opensim.Vec3(0, 0, -0.25),
            opensim.Vec3(0, 0, 0)
        )

        self.noutput = self.osim_model.noutput

        geometry = opensim.Ellipsoid(0.02, 0.02, 0.02)
        geometry.setColor(opensim.Green)
        blockos.attachGeometry(geometry)

        self.osim_model.model.addJoint(self.target_joint)
        self.osim_model.model.addBody(blockos)
        self.osim_model.model.initSystem()

    # ------------------------------------------------------------------
    # Shaped reward
    #
    # Components:
    #   1. Distance penalty    — primary task signal (reach the target)
    #   2. Effort penalty      — discourages full co-contraction of all
    #                            muscles simultaneously; encourages the
    #                            agent to use only the muscles it needs
    #   3. Smoothness penalty  — discourages high joint velocities /
    #                            jerky motion; promotes physiological
    #                            movement trajectories
    #   4. Joint limit penalty — soft barrier at anatomical ROM limits;
    #                            prevents unrealistic joint angles
    # ------------------------------------------------------------------
    def reward(self):
        state_desc = self.get_state_desc()

        # 1. Distance to target (primary)
        dx = state_desc["markers"]["r_radius_styloid"]["pos"][0] - self.target_x
        dy = state_desc["markers"]["r_radius_styloid"]["pos"][1] - self.target_y
        dist_penalty = dx ** 2 + dy ** 2

        # 2. Metabolic effort — penalise squared activations across all 6 muscles
        activations   = [state_desc["muscles"][m]["activation"]
                         for m in state_desc["muscles"]]
        effort_penalty = np.sum(np.square(activations))

        # 3. Smoothness — penalise high joint angular velocities
        joint_vels        = (state_desc["joint_vel"]["r_shoulder"] +
                             state_desc["joint_vel"]["r_elbow"])
        smoothness_penalty = np.sum(np.square(joint_vels))

        # 4. Joint range-of-motion soft barrier
        shoulder_pos  = state_desc["joint_pos"]["r_shoulder"][0]
        elbow_pos     = state_desc["joint_pos"]["r_elbow"][0]
        joint_penalty = (_range_violation(shoulder_pos, SHOULDER_MIN, SHOULDER_MAX) +
                         _range_violation(elbow_pos,    ELBOW_MIN,    ELBOW_MAX))

        total = (1.0
                 - dist_penalty
                 - W_EFFORT      * effort_penalty
                 - W_SMOOTHNESS  * smoothness_penalty
                 - W_JOINT_LIM   * joint_penalty)

        return float(np.nan_to_num(total))

    def get_reward(self):
        return self.reward()


class Arm2DVecEnv(Arm2DEnv):
    def reset(self, obs_as_dict=False):
        obs = super(Arm2DVecEnv, self).reset(obs_as_dict=obs_as_dict)
        if np.isnan(obs).any():
            obs = np.nan_to_num(obs)
        return obs

    def step(self, action, obs_as_dict=False):
        if np.isnan(action).any():
            action = np.nan_to_num(action)
        obs, reward, done, info = super(Arm2DVecEnv, self).step(action, obs_as_dict=obs_as_dict)
        if np.isnan(obs).any():
            obs    = np.nan_to_num(obs)
            done   = True
            reward -= 10   # FIX: was `reward - 10` (no-op); now correctly subtracts penalty
        return obs, reward, done, info
