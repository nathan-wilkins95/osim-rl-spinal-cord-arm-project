"""arm_SC.py — Spinal Cord-Augmented (SC) Arm Environment

Extends the baseline arm environment (``arm.py``) with a biologically-inspired
spinal cord interneuron layer that sits between the RL policy output and the
OpenSim muscle activations.  This is the experimental condition in the spinal
cord feedback study.

Spinal Cord Pipeline (per step)
--------------------------------
1. **Prochazka Ia afferent rates** — ``Prochazka_Ia_rates()`` computes
   normalised Ia afferent firing rates from current fiber length and velocity
   using the Prochazka (1999) model.
2. **Sigmoid compression** — raw Ia rates are passed through a sigmoid to
   produce bounded interneuron signals (r_Ia_s).
3. **Reciprocal inhibition** — ``W_SC`` (6×6 matrix) encodes Ia inhibitory
   interneuron projections from each muscle onto its functional antagonists.
4. **Motor neuron output** — ``r_mn = clip(sigmoid(W_SC @ r_Ia_s) + action, 0, 1)``
   combines spinal modulation with the policy action before passing to OpenSim.

Classes
-------
Arm2DEnv
    SC-augmented version of the base environment.  Stores r_Ia and r_mn
    arrays for per-step logging. Inherits from OsimEnv.

Arm2DVecEnv
    Vectorised, NaN-safe wrapper. Houses the Prochazka model and step logic
    that injects the SC layer into the action pipeline.

Model
-----
    arm2dof6musc.osim  —  2 DOF, 6 Hill-type muscles (same as arm.py).
    W_SC (6×6)         —  Reciprocal inhibition connectivity matrix.
                          Rows/cols: [BIClong, BICshort, BRA, TRIlong, TRIlat, TRImed]
                          Diagonal = self-excitation (1.0);
                          Off-diagonal antagonist pairs = -0.5 (inhibitory).

Usage
-----
    # Training (see train_arm_SC.py)
    from osim.env.arm_SC import Arm2DVecEnv
    env = Arm2DVecEnv(visualize=False)
    obs = env.reset()
    obs, reward, done, info = env.step(action)  # action modulated by SC layer

References
----------
    Prochazka, A. (1999). Quantifying proprioception. Prog. Brain Res., 123, 133-142.
    Eccles, J.C. et al. (1956). Central pathways for Ia inhibitory interneurons.
        J. Physiol.
    Delp et al. (2007). OpenSim. IEEE Trans. Biomed. Eng.
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
# ---------------------------------------------------------------------------
SHOULDER_MIN, SHOULDER_MAX = -0.5236, 2.0944
ELBOW_MIN,    ELBOW_MAX    =  0.0,    2.3562

# ---------------------------------------------------------------------------
# Reward shaping weights (conservative starting values)
# ---------------------------------------------------------------------------
W_EFFORT     = 0.005
W_SMOOTHNESS = 0.0005
W_JOINT_LIM  = 0.5

# Prochazka Ia afferent model parameters
# rate = IA_A * signed_vel_term + IA_B * (length - opt_length) + IA_C
IA_A = 4.3   # velocity sensitivity coefficient
IA_B = 2.0   # length sensitivity coefficient
IA_C = 10.0  # baseline firing rate offset

# Muscle groupings for CCI
FLEXOR_MUSCLES   = ['BIClong', 'BICshort', 'BRA']
EXTENSOR_MUSCLES = ['TRIlong', 'TRIlat',   'TRImed']


def _range_violation(val: float, low: float, high: float, margin: float = 0.1) -> float:
    """Compute a soft-barrier joint limit penalty.

    Returns a squared deviation when *val* encroaches within *margin* of a
    physiological limit, and zero when comfortably inside the range.

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
        if within the safe operating range.
    """
    if val < low + margin:
        return (val - low - margin) ** 2
    if val > high - margin:
        return (val - high + margin) ** 2
    return 0.0


class Arm2DEnv(OsimEnv):
    """2-DOF, 6-muscle OpenSim arm environment with spinal cord feedback layer.

    Identical in structure to the baseline ``arm.Arm2DEnv`` but initialises
    the Ia afferent (r_Ia) and motor neuron (r_mn) state arrays, and exposes
    them via ``get_aux_info`` for per-step logging by the test harness.

    Attributes
    ----------
    model_path : str
        Absolute path to the .osim musculoskeletal model file.
    time_limit : int
        Maximum simulation steps per episode (default 200).
    target_x : float
        Current reach target x-coordinate (metres).
    target_y : float
        Current reach target y-coordinate (metres).
    _r_Ia : numpy.ndarray, shape (6,)
        Most recent Ia afferent firing rates (normalised, [0, 1]).
    _r_mn : numpy.ndarray, shape (6,)
        Most recent motor neuron output activations ([0, 1]) sent to OpenSim.
    W_SC : numpy.ndarray, shape (6, 6)
        Reciprocal inhibition connectivity matrix.
    target_joint : opensim.PlanarJoint
        Planar joint used to position the visual target in the OpenSim scene.
    """

    model_path = os.path.join(os.path.dirname(__file__), '../models/arm2dof6musc.osim')
    time_limit = 200
    target_x = 0
    target_y = 0

    def get_aux_info(self):
        """Return the most recent spinal cord intermediate signals.

        Used by the test harness (``test_SC_agents.py``) to log r_Ia and
        r_mn arrays alongside the standard state for SC-specific analysis.

        Returns
        -------
        r_Ia : numpy.ndarray, shape (6,)
            Normalised Ia afferent firing rates from the last step.
        r_mn : numpy.ndarray, shape (6,)
            Motor neuron output activations from the last step.
        """
        r_Ia = self._r_Ia if self._r_Ia is not None else np.zeros(6)
        r_mn = self._r_mn if self._r_mn is not None else np.zeros(6)
        return r_Ia, r_mn

    def get_d_state(self, action):
        """Build a flat state dictionary for logging and analysis.

        Identical to ``arm.Arm2DEnv.get_d_state``. Collects kinematics,
        muscle states, wrist marker position, CCI, and recruitment diversity
        into a single dict per step.

        Parameters
        ----------
        action : array-like
            Motor neuron activations applied this step (r_mn from SC layer).

        Returns
        -------
        dict
            Same keys as ``arm.Arm2DEnv.get_d_state``, plus SC-specific data
            is captured separately via ``get_aux_info``.
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

        # Co-contraction index
        flexor_act   = np.mean([state_desc["muscles"][m]["activation"] for m in FLEXOR_MUSCLES])
        extensor_act = np.mean([state_desc["muscles"][m]["activation"] for m in EXTENSOR_MUSCLES])
        d["CCI"] = float(min(flexor_act, extensor_act))

        # Recruitment diversity
        all_acts = [state_desc["muscles"][m]["activation"] for m in sorted(state_desc["muscles"].keys())]
        d["recruitment_diversity"] = float(np.std(all_acts))

        return d

    def get_observation(self):
        """Construct the 34-element observation vector for the RL policy.

        Returns
        -------
        list of float
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
        """Return the observation vector length (34)."""
        return 34

    def generate_new_target(self):
        """Sample a new random reach target and update the OpenSim scene.

        See ``arm.Arm2DEnv.generate_new_target`` for full documentation.
        Behaviour is identical in the SC condition.
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
        obs_as_dict : bool, optional
            Passed through to OsimEnv reset (default True).

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
        """Initialise the SC arm environment.

        Extends the base initialisation by:
        - Zeroing r_Ia and r_mn arrays before the parent __init__ call
          (prevents AttributeError on the first logging step).
        - Building the 6×6 reciprocal inhibition weight matrix W_SC that
          encodes Ia interneuron projections between antagonist muscle pairs.

        W_SC structure::

            Rows/cols: [BIClong, BICshort, BRA, TRIlong, TRIlat, TRImed]
            Diagonal  (+1.0): self-excitation
            [flexor, extensor] pairs (-0.5): reciprocal inhibition
        """
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

        # Reciprocal inhibition weight matrix
        # Rows/cols: [BIClong, BICshort, BRA, TRIlong, TRIlat, TRImed]
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
        """Compute the shaped per-step reward and component breakdown.

        Identical reward formulation to ``arm.Arm2DEnv.reward``. The SC layer
        affects the reward indirectly through the muscle activations it
        produces — the reward function itself does not differ between conditions,
        enabling a fair MS vs SC comparison.

        Returns
        -------
        total : float
            Scalar reward (NaN-safe).
        info : dict
            Keys: ``reward_dist``, ``reward_effort``, ``reward_smooth``,
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

        info = {
            'reward_dist':   float(dist_penalty),
            'reward_effort': float(W_EFFORT     * effort_penalty),
            'reward_smooth': float(W_SMOOTHNESS * smoothness_penalty),
            'reward_joint':  float(W_JOINT_LIM  * joint_penalty),
            'reward_total':  float(np.nan_to_num(total)),
        }
        return float(np.nan_to_num(total)), info

    def get_reward(self):
        """Return only the scalar reward (keras-rl compatible interface)."""
        total, _ = self.reward()
        return total


class Arm2DVecEnv(Arm2DEnv):
    """Vectorised SC arm environment with Prochazka Ia model and NaN guards.

    This is the class used directly by the DDPG agent in ``train_arm_SC.py``.
    On each ``step`` call it:

    1. Computes Ia afferent rates via ``Prochazka_Ia_rates``.
    2. Passes them through the sigmoid and W_SC reciprocal inhibition matrix.
    3. Adds the RL policy action and clips to [0, 1] to produce r_mn.
    4. Passes r_mn to OpenSim as the actual muscle activation command.

    This means the RL agent's action is interpreted as a *descending drive*
    modulated by spinal feedback rather than a direct activation command.
    """

    def Prochazka_Ia_rates(self, a=IA_A, b=IA_B, c=IA_C):
        """Compute normalised Ia afferent firing rates using the Prochazka (1999) model.

        For each muscle, the Ia firing rate depends on fiber velocity (dynamic
        sensitivity) and fiber length (static sensitivity)::

            rate_i = a * sign(v) * |v|^0.6  +  b * (l - l_opt)  +  c

        Rates are normalised to [0, 1] by dividing by the maximum possible rate
        at peak contraction velocity and 150% optimal fiber length.

        Numerical safeguards applied:
        - abs(fiber_velocity) clamped to [1e-6, max_v] to prevent log(0).
        - max_v clamped to ≥ 1e-6 to prevent division by zero.
        - Normalised rate floored at 0 to discard negative artefacts.

        Parameters
        ----------
        a : float, optional
            Velocity sensitivity coefficient (default IA_A = 4.3).
        b : float, optional
            Length sensitivity coefficient (default IA_B = 2.0).
        c : float, optional
            Baseline firing rate offset (default IA_C = 10.0).

        Returns
        -------
        numpy.ndarray, shape (6,)
            Normalised Ia firing rates for each muscle in muscleSet order.
        """
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
        """Element-wise sigmoid activation function.

        Used to compress Ia rates and W_SC-weighted interneuron signals into
        the [0, 1] range before combining with the RL policy action.

        Parameters
        ----------
        x : array-like
            Input values.
        alpha : float, optional
            Steepness of the sigmoid (default 8 — relatively sharp).
        beta : float, optional
            Midpoint / threshold value (default 0.5).

        Returns
        -------
        numpy.ndarray
            Sigmoid-transformed values in (0, 1).
        """
        return 1.0 / (1.0 + np.exp(-alpha * (x - beta)))

    def reset(self, obs_as_dict=False):
        """Reset the environment and sanitise the initial observation.

        Parameters
        ----------
        obs_as_dict : bool, optional
            If False (default), returns a flat numpy array.

        Returns
        -------
        numpy.ndarray
            Initial observation with NaN values replaced by 0.
        """
        obs = super(Arm2DVecEnv, self).reset(obs_as_dict=obs_as_dict)
        if np.isnan(obs).any():
            obs = np.nan_to_num(obs)
        return obs

    def step(self, action, obs_as_dict=False):
        """Step the environment with the full spinal cord pipeline.

        Translates the RL policy's raw action (descending drive) into a
        spinal-cord-modulated motor neuron command before passing to OpenSim::

            r_Ia   = Prochazka_Ia_rates()                  # afferent signal
            r_Ia_s = sigmoid(r_Ia)                          # interneuron signal
            r_mn   = clip(sigmoid(W_SC @ r_Ia_s) + action, 0, 1)  # motor output

        NaN guards:
        - NaN actions are zeroed before the SC computation.
        - NaN observations terminate the episode with a -10 reward penalty.

        Parameters
        ----------
        action : array-like, shape (6,)
            RL policy output (descending drive), nominally in [0, 1].
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
