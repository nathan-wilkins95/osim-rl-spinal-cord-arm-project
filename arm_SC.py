"""arm_SC.py — Spinal Cord-Augmented (SC) Arm Environment

Extends the baseline arm environment (``arm.py``) with a biologically-inspired
spinal cord interneuron layer that sits between the RL policy output and the
OpenSim muscle activations.  This is the experimental condition in the spinal
cord feedback study.

Spinal Cord Pipeline (per step)
--------------------------------
1. **Prochazka Ia afferent rates** — ``Prochazka_Ia_rates()`` computes
   normalised Ia afferent firing rates from current fiber length and velocity.
2. **Sigmoid compression** — raw Ia rates passed through sigmoid -> r_Ia_s.
3. **Reciprocal inhibition** — W_SC (6x6) encodes antagonist inhibition.
4. **Motor neuron output** — r_mn = clip(sigmoid(W_SC @ r_Ia_s) + action, 0, 1).

References
----------
    Prochazka (1999). Prog. Brain Res., 123, 133-142.
    Eccles et al. (1956). J. Physiol.
    Delp et al. (2007). IEEE Trans. Biomed. Eng.
"""

import math
import numpy as np
import os
from .utils.mygym import convert_to_gym
import gym
import opensim
import random
from .osim import OsimEnv

SHOULDER_MIN, SHOULDER_MAX = -0.5236, 2.0944
ELBOW_MIN,    ELBOW_MAX    =  0.0,    2.3562

W_EFFORT     = 0.005
W_SMOOTHNESS = 0.0005
W_JOINT_LIM  = 0.5

IA_A = 4.3
IA_B = 2.0
IA_C = 10.0

FLEXOR_MUSCLES   = ['BIClong', 'BICshort', 'BRA']
EXTENSOR_MUSCLES = ['TRIlong', 'TRIlat',   'TRImed']


def _range_violation(val: float, low: float, high: float, margin: float = 0.1) -> float:
    """Compute a soft-barrier joint limit penalty."""
    if val < low + margin:
        return (val - low - margin) ** 2
    if val > high - margin:
        return (val - high + margin) ** 2
    return 0.0


class Arm2DEnv(OsimEnv):
    """2-DOF, 6-muscle OpenSim arm environment with spinal cord feedback layer."""

    model_path = os.path.join(os.path.dirname(__file__), '../models/arm2dof6musc.osim')
    time_limit = 200
    target_x = 0
    target_y = 0

    def get_aux_info(self):
        """Return the most recent spinal cord intermediate signals (r_Ia, r_mn)."""
        r_Ia = self._r_Ia if self._r_Ia is not None else np.zeros(6)
        r_mn = self._r_mn if self._r_mn is not None else np.zeros(6)
        return r_Ia, r_mn

    def get_d_state(self, action):
        """Build a flat state dictionary for logging and analysis.

        Joint pos/vel/acc are stored as scalars (index [0]) since
        state_desc returns 1-element lists for each coordinate.
        """
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
            d[f'{joint}_pos'] = state_desc["joint_pos"][joint][0]
            d[f'{joint}_vel'] = state_desc["joint_vel"][joint][0]
            d[f'{joint}_acc'] = state_desc["joint_acc"][joint][0]

        for muscle in sorted(state_desc["muscles"].keys()):
            d[f'{muscle}_activation']     = state_desc["muscles"][muscle]["activation"]
            d[f'{muscle}_fiber_length']   = state_desc["muscles"][muscle]["fiber_length"]
            d[f'{muscle}_fiber_velocity'] = state_desc["muscles"][muscle]["fiber_velocity"]

        d["markers_0"] = state_desc["markers"]["r_radius_styloid"]["pos"][0]
        d["markers_1"] = state_desc["markers"]["r_radius_styloid"]["pos"][1]

        flexor_act   = np.mean([state_desc["muscles"][m]["activation"] for m in FLEXOR_MUSCLES])
        extensor_act = np.mean([state_desc["muscles"][m]["activation"] for m in EXTENSOR_MUSCLES])
        d["CCI"] = float(min(flexor_act, extensor_act))

        all_acts = [state_desc["muscles"][m]["activation"] for m in sorted(state_desc["muscles"].keys())]
        d["recruitment_diversity"] = float(np.std(all_acts))

        return d

    def get_observation(self):
        """Construct the 28-element observation vector for the RL policy."""
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
        """Return the observation vector length (28)."""
        return 28

    def generate_new_target(self):
        """Sample a new random reach target and update the OpenSim scene."""
        if os.getenv("FIXED_TARGET"):
            self.target_x = 0.16056636337579086
            self.target_y = 0.49340151308159397
        else:
            theta         = random.uniform(0, math.pi * 2 / 3)
            radius        = random.uniform(0.3, 0.65)
            self.target_x = math.cos(theta) * radius
            self.target_y = -math.sin(theta) * radius + 0.8

        print('\ntarget: [{:.4f} {:.4f}]'.format(self.target_x, self.target_y))

        state = self.osim_model.get_state()
        self.target_joint.getCoordinate(1).setValue(state, self.target_x, False)
        self.target_joint.getCoordinate(2).setLocked(state, False)
        self.target_joint.getCoordinate(2).setValue(state, self.target_y, False)
        self.target_joint.getCoordinate(2).setLocked(state, True)
        self.osim_model.set_state(state)

    def reset(self, random_target=True, obs_as_dict=True):
        """Reset the environment to its initial state."""
        obs = super(Arm2DEnv, self).reset(obs_as_dict=obs_as_dict)
        if random_target:
            self.generate_new_target()
        self.osim_model.reset_manager()
        return obs

    def __init__(self, *args, **kwargs):
        """Initialise the SC arm environment."""
        self._r_Ia = np.zeros(6)
        self._r_mn = np.zeros(6)
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

        self.W_SC = np.array([
            [ 1.0,  0.0,  0.0, -0.5,  0.0,  0.0],
            [ 0.0,  1.0,  0.0,  0.0, -0.5, -0.5],
            [ 0.0,  0.0,  1.0,  0.0, -0.5, -0.5],
            [-0.5,  0.0,  0.0,  1.0,  0.0,  0.0],
            [ 0.0, -0.5, -0.5,  0.0,  1.0,  0.0],
            [ 0.0, -0.5, -0.5,  0.0,  0.0,  1.0],
        ], dtype=float)

        self.noutput = self.osim_model.noutput
        geometry = opensim.Ellipsoid(0.02, 0.02, 0.02)
        geometry.setColor(opensim.Green)
        blockos.attachGeometry(geometry)
        self.osim_model.model.addJoint(self.target_joint)
        self.osim_model.model.addBody(blockos)
        self.osim_model.model.initSystem()

    def reward(self):
        """Compute the shaped per-step reward and component breakdown."""
        state_desc = self.get_state_desc()

        dx = state_desc["markers"]["r_radius_styloid"]["pos"][0] - self.target_x
        dy = state_desc["markers"]["r_radius_styloid"]["pos"][1] - self.target_y
        dist_penalty = dx ** 2 + dy ** 2

        activations    = [state_desc["muscles"][m]["activation"] for m in state_desc["muscles"]]
        effort_penalty = np.sum(np.square(activations))

        joint_vels         = (state_desc["joint_vel"]["r_shoulder"] +
                              state_desc["joint_vel"]["r_elbow"])
        smoothness_penalty = np.sum(np.square(joint_vels))

        shoulder_pos  = state_desc["joint_pos"]["r_shoulder"][0]
        elbow_pos     = state_desc["joint_pos"]["r_elbow"][0]
        joint_penalty = (_range_violation(shoulder_pos, SHOULDER_MIN, SHOULDER_MAX) +
                         _range_violation(elbow_pos,    ELBOW_MIN,    ELBOW_MAX))

        total = (1.0
                 - dist_penalty
                 - W_EFFORT     * effort_penalty
                 - W_SMOOTHNESS * smoothness_penalty
                 - W_JOINT_LIM  * joint_penalty)

        info = {
            'reward_dist':    float(dist_penalty),
            'reward_effort':  float(W_EFFORT     * effort_penalty),
            'reward_smooth':  float(W_SMOOTHNESS * smoothness_penalty),
            'reward_joint':   float(W_JOINT_LIM  * joint_penalty),
            'reward_total':   float(np.nan_to_num(total)),
        }
        return float(np.nan_to_num(total)), info

    def get_reward(self):
        """Return only the scalar reward."""
        total, _ = self.reward()
        return total


class Arm2DVecEnv(Arm2DEnv):
    """Vectorised SC arm environment with Prochazka Ia model and NaN guards."""

    def Prochazka_Ia_rates(self, a=IA_A, b=IA_B, c=IA_C):
        """Compute normalised Ia afferent firing rates (Prochazka 1999)."""
        state_desc = self.get_state_desc()
        norm_rate  = np.zeros(6)

        for i in range(self.osim_model.muscleSet.getSize()):
            muscle  = self.osim_model.muscleSet.get(i)
            name    = muscle.getName()

            fiber_l = state_desc["muscles"][name]["fiber_length"]  * 1000
            fiber_v = state_desc["muscles"][name]["fiber_velocity"] * 1000
            opt_l   = muscle.getOptimalFiberLength()     * 1000
            max_v   = muscle.getMaxContractionVelocity() * opt_l

            abs_v_clamped   = max(min(abs(fiber_v), max_v), 1e-6)
            max_v_clamped   = max(max_v, 1e-6)
            signed_vel_term = (np.sign(fiber_v) if fiber_v != 0 else 1.0) * \
                              np.exp(0.6 * np.log(abs_v_clamped))

            rate  = (a * signed_vel_term
                     + b * (min(fiber_l, 1.5 * opt_l) - opt_l)
                     + c)
            denom = a * np.exp(0.6 * np.log(max_v_clamped)) + b * 0.5 * opt_l + c
            norm_rate[i] = max(rate / denom, 0.0)

        return norm_rate

    def sigmoid(self, x, alpha=8, beta=0.5):
        """Element-wise sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-alpha * (x - beta)))

    def reset(self, obs_as_dict=False):
        """Reset the environment and sanitise the initial observation."""
        obs = super(Arm2DVecEnv, self).reset(obs_as_dict=obs_as_dict)
        if np.isnan(obs).any():
            obs = np.nan_to_num(obs)
        return obs

    def step(self, action, obs_as_dict=False):
        """Step the environment with the full spinal cord pipeline."""
        if np.isnan(action).any():
            action = np.nan_to_num(action)

        r_Ia       = self.Prochazka_Ia_rates()
        self._r_Ia = r_Ia
        r_Ia_s     = self.sigmoid(r_Ia)
        r_mn       = np.clip(self.sigmoid(np.matmul(self.W_SC, r_Ia_s)) + action, 0.0, 1.0)
        self._r_mn = r_mn

        obs, reward, done, info = super(Arm2DVecEnv, self).step(r_mn, obs_as_dict=obs_as_dict)

        if np.isnan(obs).any():
            obs     = np.nan_to_num(obs)
            done    = True
            reward -= 10

        return obs, reward, done, info
