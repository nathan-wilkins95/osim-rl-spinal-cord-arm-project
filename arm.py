"""arm.py — Baseline Musculoskeletal (MS) Arm Environment

Defines the OpenSim-based 2-DOF, 6-muscle arm reaching environment used as the
control condition in the spinal cord feedback study. A DDPG agent controls muscle
activations directly — no spinal cord layer is present.

Classes
-------
Arm2DEnv
    Core gym-compatible environment wrapping the OpenSim arm model. Handles
    observation construction, shaped reward computation, target generation, and
    state logging. Inherits from OsimEnv (osim-rl).

Arm2DVecEnv
    Vectorised wrapper around Arm2DEnv. Adds NaN safety guards on observations
    and actions for use with keras-rl's DDPG agent.

Model
-----
    arm2dof6musc.osim  —  2 degrees of freedom (shoulder flexion, elbow flexion)
                          6 Hill-type muscles: BIClong, BICshort, BRA (flexors)
                                               TRIlong, TRIlat, TRImed (extensors)

Usage
-----
    # Training (see train_arm.py)
    from osim.env.arm import Arm2DVecEnv
    env = Arm2DVecEnv(visualize=False)
    obs = env.reset()
    obs, reward, done, info = env.step(action)

Notes
-----
    Observation space: 34 values
        [0-1]   target x, y (m)
        [2-13]  r_shoulder + r_elbow: pos, vel, acc (3 values each)
        [14-31] 6 muscles × 3: activation [-], fiber_length [m], fiber_velocity [m/s]
        [32-33] wrist marker (r_radius_styloid) x, y (m)

    Reward is shaped across four components:
        r = 1.0 - dist_penalty - effort_penalty - smoothness_penalty - joint_limit_penalty

References
----------
    Delp et al. (2007). OpenSim: Open-source software to create and analyze
        dynamic simulations of movement. IEEE Trans. Biomed. Eng.
    Crowninshield & Brand (1981). A physiologically based criterion of muscle
        force prediction in locomotion. J. Biomechanics.
"""

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
# Shoulder: -30deg to 120deg of flexion in the 2-D sagittal plane
# Elbow:      0deg (full extension) to ~135deg of flexion
# ---------------------------------------------------------------------------
SHOULDER_MIN, SHOULDER_MAX = -0.5236, 2.0944
ELBOW_MIN,    ELBOW_MAX    =  0.0,    2.3562

# ---------------------------------------------------------------------------
# Reward shaping weights
# Conservative starting values — halved from initial to prevent effort penalty
# from overwhelming the distance signal in early training (agent collapsing
# to zero activation).  Tune upward once agent reliably reaches targets.
#
# Sanity check: at max co-contraction (all 6 activations = 1.0)
#   effort term  = W_EFFORT * 6 = 0.03  << 1.0 base reward  ✓
# ---------------------------------------------------------------------------
W_EFFORT     = 0.005   # penalty per unit squared muscle activation (sum over 6)
W_SMOOTHNESS = 0.0005  # penalty per unit squared joint angular velocity
W_JOINT_LIM  = 0.5     # soft-barrier penalty for exceeding anatomical ROM

# Muscle groupings for Co-Contraction Index (CCI) computation
FLEXOR_MUSCLES   = ['BIClong', 'BICshort', 'BRA']
EXTENSOR_MUSCLES = ['TRIlong', 'TRIlat',   'TRImed']


def _range_violation(val: float, low: float, high: float, margin: float = 0.1) -> float:
    """Compute a soft-barrier joint limit penalty.

    Returns a squared deviation when *val* encroaches within *margin* of a
    physiological limit, and zero when comfortably inside the range.  This
    discourages hyperextension / hyperlexion without creating a hard wall
    that would destabilise the RL gradient.

    Parameters
    ----------
    val : float
        Current joint angle (radians).
    low : float
        Minimum physiological joint angle (radians).
    high : float
        Maximum physiological joint angle (radians).
    margin : float, optional
        Safety buffer inside each limit at which the penalty activates
        (default 0.1 rad ≈ 5.7°).

    Returns
    -------
    float
        Squared angular deviation from the nearest limit boundary, or 0.0
        if the joint is within the safe operating range.
    """
    if val < low + margin:
        return (val - low - margin) ** 2
    if val > high - margin:
        return (val - high + margin) ** 2
    return 0.0


class Arm2DEnv(OsimEnv):
    """2-DOF, 6-muscle OpenSim arm reaching environment (baseline MS condition).

    Wraps the OpenSim musculoskeletal model ``arm2dof6musc.osim`` as a
    gym-compatible environment.  The RL policy outputs muscle activations
    directly — there is no spinal cord processing layer (see ``arm_SC.py``
    for the SC-augmented condition).

    Attributes
    ----------
    model_path : str
        Absolute path to the .osim musculoskeletal model file.
    time_limit : int
        Maximum number of simulation steps per episode (default 200).
    target_x : float
        Current reach target x-coordinate (metres).
    target_y : float
        Current reach target y-coordinate (metres).
    target_joint : opensim.PlanarJoint
        Planar joint used to position the visual reach target sphere in the
        OpenSim scene.
    noutput : int
        Number of muscle actuators (= 6).
    """

    model_path = os.path.join(os.path.dirname(__file__), '../models/arm2dof6musc.osim')
    time_limit = 200
    target_x = 0
    target_y = 0

    def get_d_state(self, action):
        """Build a flat state dictionary for logging and analysis.

        Collects per-step kinematic, muscle, and co-contraction data into a
        single dict suitable for writing to a pickle file and later plotting
        with the ``pickle_*.py`` scripts.

        Parameters
        ----------
        action : array-like
            The muscle activation vector applied at this step (unused in
            computation but included for API consistency with subclasses).

        Returns
        -------
        dict
            Keys include:
            - ``r_ulna_radius_hand_pos/vel/acc_{0,1}`` : hand segment kinematics
            - ``r_shoulder/r_elbow_pos/vel/acc``        : joint kinematics
            - ``{muscle}_activation/fiber_length/fiber_velocity`` : 6 muscles
            - ``markers_0/1``          : wrist marker x, y (m)
            - ``CCI``                  : Co-Contraction Index (0–0.5)
            - ``recruitment_diversity``: std of muscle activations across 6 muscles
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
            d[f'{joint}_pos'] = state_desc["joint_pos"][joint]
            d[f'{joint}_vel'] = state_desc["joint_vel"][joint]
            d[f'{joint}_acc'] = state_desc["joint_acc"][joint]

        for muscle in sorted(state_desc["muscles"].keys()):
            d[f'{muscle}_activation']     = state_desc["muscles"][muscle]["activation"]
            d[f'{muscle}_fiber_length']   = state_desc["muscles"][muscle]["fiber_length"]
            d[f'{muscle}_fiber_velocity'] = state_desc["muscles"][muscle]["fiber_velocity"]

        d["markers_0"] = state_desc["markers"]["r_radius_styloid"]["pos"][0]
        d["markers_1"] = state_desc["markers"]["r_radius_styloid"]["pos"][1]

        # Co-contraction index: min(mean_flexor_act, mean_extensor_act)
        # 0 = no co-contraction, 0.5 = full simultaneous activation
        flexor_act   = np.mean([state_desc["muscles"][m]["activation"] for m in FLEXOR_MUSCLES])
        extensor_act = np.mean([state_desc["muscles"][m]["activation"] for m in EXTENSOR_MUSCLES])
        d["CCI"] = float(min(flexor_act, extensor_act))

        # Recruitment diversity: std of mean activation across all 6 muscles
        # Low = one or two muscles dominate; high = coordinated selective recruitment
        all_acts = [state_desc["muscles"][m]["activation"] for m in sorted(state_desc["muscles"].keys())]
        d["recruitment_diversity"] = float(np.std(all_acts))

        return d

    def get_observation(self):
        """Construct the 34-element observation vector for the RL policy.

        Returns
        -------
        list of float
            Ordered observation values:
            [0-1]   target x, y (m)
            [2-7]   r_shoulder: pos, vel, acc
            [8-13]  r_elbow:    pos, vel, acc
            [14-31] 6 muscles (alphabetical) × [activation, fiber_length, fiber_velocity]
            [32-33] wrist marker (r_radius_styloid) x, y (m)
        """
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
        """Return the number of elements in the observation vector.

        Returns
        -------
        int
            34 (2 target + 2 joints × 3 kinematics + 6 muscles × 3 + 2 marker).
        """
        # 2 target + 2 joints*3 + 6 muscles*3 + 2 marker = 34
        return 34

    def generate_new_target(self):
        """Sample or set a new reach target and update the OpenSim scene.

        In normal training, draws a random target position within a reachable
        arc (theta in [0, 2π/3], radius in [0.3, 0.65] m) in the sagittal
        plane.  When the environment variable ``FIXED_TARGET=1`` is set, uses
        a hard-coded position for reproducible debugging.

        Side Effects
        ------------
        Updates ``self.target_x``, ``self.target_y``, and the position of
        ``self.target_joint`` in the running OpenSim simulation state.
        """
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
        """Reset the environment to its initial state.

        Parameters
        ----------
        random_target : bool, optional
            If True (default), generates a new random reach target.
            Set to False to keep the previous target position.
        obs_as_dict : bool, optional
            Passed through to the parent OsimEnv reset (default True).

        Returns
        -------
        array-like
            Initial observation vector.
        """
        obs = super(Arm2DEnv, self).reset(obs_as_dict=obs_as_dict)
        if random_target:
            self.generate_new_target()
        self.osim_model.reset_manager()
        return obs

    def __init__(self, *args, **kwargs):
        """Initialise the arm environment and add the reach target to the model.

        Calls the parent OsimEnv constructor, then attaches a massless target
        body (rendered as a green ellipsoid) to the model via a PlanarJoint.
        The target's x/y coordinates are updated each episode by
        ``generate_new_target``.
        """
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

    def reward(self):
        """Compute the shaped per-step reward and its component breakdown.

        The reward balances four competing objectives:

        1. **Distance** — minimise squared Euclidean wrist-to-target distance.
        2. **Effort**   — penalise total squared muscle activation (anti-
                          co-contraction; metabolic cost proxy).
        3. **Smoothness** — penalise squared joint angular velocities (anti-jerk).
        4. **Joint limits** — soft-barrier penalty for approaching anatomical ROM
                              bounds (see ``_range_violation``).

        Formula::

            r = 1.0
                - dist_penalty
                - W_EFFORT     * sum(activation_i^2)
                - W_SMOOTHNESS * sum(joint_vel_j^2)
                - W_JOINT_LIM  * joint_barrier_penalty

        Returns
        -------
        total : float
            Scalar reward for this step (NaN-safe).
        info : dict
            Per-component breakdown with keys:
            ``reward_dist``, ``reward_effort``, ``reward_smooth``,
            ``reward_joint``, ``reward_total``.
        """
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

        # Return component breakdown alongside scalar for logging
        info = {
            'reward_dist':    float(dist_penalty),
            'reward_effort':  float(W_EFFORT     * effort_penalty),
            'reward_smooth':  float(W_SMOOTHNESS * smoothness_penalty),
            'reward_joint':   float(W_JOINT_LIM  * joint_penalty),
            'reward_total':   float(np.nan_to_num(total)),
        }
        return float(np.nan_to_num(total)), info

    def get_reward(self):
        """Return only the scalar reward (keras-rl compatible interface).

        Returns
        -------
        float
            Total shaped reward for this step.
        """
        total, _ = self.reward()
        return total


class Arm2DVecEnv(Arm2DEnv):
    """Vectorised, NaN-safe wrapper around Arm2DEnv for use with keras-rl DDPG.

    Adds input/output safety guards that sanitise NaN values in observations
    and actions — necessary when OpenSim occasionally produces non-finite
    states during highly perturbed or degenerate simulation steps.

    Inherits all environment logic from ``Arm2DEnv``.  This class is the one
    passed directly to the DDPG agent in ``train_arm.py``.
    """

    def reset(self, obs_as_dict=False):
        """Reset the environment and sanitise the initial observation.

        Parameters
        ----------
        obs_as_dict : bool, optional
            If False (default), returns a flat numpy array.

        Returns
        -------
        numpy.ndarray
            Initial observation with any NaN values replaced by 0.
        """
        obs = super(Arm2DVecEnv, self).reset(obs_as_dict=obs_as_dict)
        if np.isnan(obs).any():
            obs = np.nan_to_num(obs)
        return obs

    def step(self, action, obs_as_dict=False):
        """Step the environment with NaN guards on both action and observation.

        If the incoming action contains NaN values they are zeroed before being
        passed to OpenSim.  If the resulting observation is non-finite the
        episode is terminated immediately and a penalty of -10 is applied to
        the reward to discourage the policy from entering degenerate states.

        Parameters
        ----------
        action : array-like, shape (6,)
            Desired muscle activation vector in [0, 1].
        obs_as_dict : bool, optional
            If False (default), returns observations as a flat numpy array.

        Returns
        -------
        obs : numpy.ndarray
            Next observation (NaN-safe).
        reward : float
            Shaped reward (with -10 penalty on NaN termination).
        done : bool
            True if the episode has ended.
        info : dict
            Reward component breakdown from ``Arm2DEnv.reward``.
        """
        if np.isnan(action).any():
            action = np.nan_to_num(action)
        obs, reward, done, info = super(Arm2DVecEnv, self).step(action, obs_as_dict=obs_as_dict)
        if np.isnan(obs).any():
            obs    = np.nan_to_num(obs)
            done   = True
            reward -= 10
        return obs, reward, done, info
