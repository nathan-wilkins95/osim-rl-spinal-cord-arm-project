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

# Reward shaping weights
W_EFFORT     = 0.01
W_SMOOTHNESS = 0.001
W_JOINT_LIM  = 0.5

# Prochazka Ia afferent model parameters
IA_A = 4.3   # velocity-sensitive gain
IA_B = 2.0   # length-sensitive gain
IA_C = 10.0  # baseline firing rate


def _range_violation(val: float, low: float, high: float, margin: float = 0.1) -> float:
    """
    Soft-barrier joint limit penalty.
    Returns squared deviation when val is within margin of (or outside) the limit.
    Zero when comfortably inside the physiological range.
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

    def get_aux_info(self):
        """
        Returns (r_Ia, r_mn) arrays for logging/analysis.
        FIX: returns zero arrays before the first step instead of (None, None),
        which previously caused crashes in any logging code expecting arrays.
        """
        r_Ia = self._r_Ia if self._r_Ia is not None else np.zeros(6)
        r_mn = self._r_mn if self._r_mn is not None else np.zeros(6)
        return r_Ia, r_mn

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
            d[f'{muscle}_activation']     = state_desc["muscles"][muscle]["activation"]
            d[f'{muscle}_fiber_length']   = state_desc["muscles"][muscle]["fiber_length"]
            d[f'{muscle}_fiber_velocity'] = state_desc["muscles"][muscle]["fiber_velocity"]

        d["markers_0"] = state_desc["markers"]["r_radius_styloid"]["pos"][0]
        d["markers_1"] = state_desc["markers"]["r_radius_styloid"]["pos"][1]
        return d

    # ------------------------------------------------------------------
    # Observation vector (34 values)
    #   [0-1]    target x, y
    #   [2-13]   r_shoulder + r_elbow: pos/vel/acc
    #   [14-31]  6 muscles x3: activation, fiber_length, fiber_velocity
    #   [32-33]  wrist marker x, y
    #
    # FIX: fiber_length and fiber_velocity restored (were commented out).
    # The SC model needs these to compute meaningful Ia afferent signals;
    # without them the agent cannot sense muscle state for coordination.
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
        # 2 target + 2 joints*3 features + 6 muscles*3 features + 2 marker = 34
        return 34

    # ------------------------------------------------------------------
    # Target generation
    # FIX: removed hardcoded overrides that pinned every episode to one
    # fixed target coordinate, preventing generalisation.
    # Set env var FIXED_TARGET=1 to restore single-target debug mode.
    # ------------------------------------------------------------------
    def generate_new_target(self):
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
        obs = super(Arm2DEnv, self).reset(obs_as_dict=obs_as_dict)
        if random_target:
            self.generate_new_target()
        self.osim_model.reset_manager()
        return obs

    def __init__(self, *args, **kwargs):
        # FIX: initialise to zero arrays (not None) so get_aux_info() is
        # safe to call before the first step without crashing.
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

        # Spinal cord reciprocal inhibition weight matrix.
        # Diagonal = self-excitation (1.0).
        # Off-diagonal -0.5 entries = Ia inhibitory interneuron projections
        # from each muscle's afferent onto its functional antagonists.
        # Layout: [BIClong, BICshort, BRA, TRIlong, TRIlat, TRImed]
        self.W_SC = np.array([
            [ 1.0,  0.0,  0.0, -0.5,  0.0,  0.0],   # BIClong  inhibits TRIlong
            [ 0.0,  1.0,  0.0,  0.0, -0.5, -0.5],   # BICshort inhibits TRIlat, TRImed
            [ 0.0,  0.0,  1.0,  0.0, -0.5, -0.5],   # BRA      inhibits TRIlat, TRImed
            [-0.5,  0.0,  0.0,  1.0,  0.0,  0.0],   # TRIlong  inhibits BIClong
            [ 0.0, -0.5, -0.5,  0.0,  1.0,  0.0],   # TRIlat   inhibits BICshort, BRA
            [ 0.0, -0.5, -0.5,  0.0,  0.0,  1.0],   # TRImed   inhibits BICshort, BRA
        ], dtype=float)

        self.noutput = self.osim_model.noutput

        geometry = opensim.Ellipsoid(0.02, 0.02, 0.02)
        geometry.setColor(opensim.Green)
        blockos.attachGeometry(geometry)

        self.osim_model.model.addJoint(self.target_joint)
        self.osim_model.model.addBody(blockos)
        self.osim_model.model.initSystem()

    # ------------------------------------------------------------------
    # Shaped reward (identical structure to arm.py)
    # ------------------------------------------------------------------
    def reward(self):
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

        return float(np.nan_to_num(total))

    def get_reward(self):
        return self.reward()


class Arm2DVecEnv(Arm2DEnv):

    def Prochazka_Ia_rates(self, a=IA_A, b=IA_B, c=IA_C):
        """
        Compute normalised Ia afferent firing rates using the Prochazka model.

        FIX 1: replaced np.sign(fiber_v) with a safe signed-power formulation.
                np.sign(0) == 0 which zeroed the velocity term at simulation
                start (fiber_v is 0 on the first step of every episode).
        FIX 2: inner np.log() now clamps its argument to >= 1e-6 (was 0.01
                but only on the velocity term; the denominator's log had no
                guard and could silently produce -inf / NaN for tiny max_v).
        FIX 3: outer max(..., 0) clamping preserved to keep rates non-negative.
        """
        state_desc = self.get_state_desc()
        norm_rate  = np.zeros(6)

        for i in range(self.osim_model.muscleSet.getSize()):
            muscle  = self.osim_model.muscleSet.get(i)
            name    = muscle.getName()

            fiber_l = state_desc["muscles"][name]["fiber_length"]  * 1000
            fiber_v = state_desc["muscles"][name]["fiber_velocity"] * 1000
            opt_l   = muscle.getOptimalFiberLength()    * 1000
            max_v   = muscle.getMaxContractionVelocity() * opt_l

            # Safe signed power: preserves direction without collapsing to 0
            # when fiber_v == 0 (as np.sign would).
            abs_v_clamped    = max(min(abs(fiber_v), max_v), 1e-6)
            max_v_clamped    = max(max_v, 1e-6)
            signed_vel_term  = (np.sign(fiber_v) if fiber_v != 0 else 1.0) * \
                               np.exp(0.6 * np.log(abs_v_clamped))

            rate = (a * signed_vel_term
                    + b * (min(fiber_l, 1.5 * opt_l) - opt_l)
                    + c)

            denom = a * np.exp(0.6 * np.log(max_v_clamped)) + b * 0.5 * opt_l + c
            norm_rate[i] = max(rate / denom, 0.0)

        return norm_rate

    def sigmoid(self, x, alpha=8, beta=0.5):
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-alpha * (x - beta)))

    def reset(self, obs_as_dict=False):
        obs = super(Arm2DVecEnv, self).reset(obs_as_dict=obs_as_dict)
        if np.isnan(obs).any():
            obs = np.nan_to_num(obs)
        return obs

    def step(self, action, obs_as_dict=False):
        if np.isnan(action).any():
            action = np.nan_to_num(action)

        # --- Spinal cord model -------------------------------------------
        # 1. Compute Ia afferent rates from current muscle state
        r_Ia       = self.Prochazka_Ia_rates()
        self._r_Ia = r_Ia

        # 2. Sigmoid-compress Ia rates -> inhibitory interneuron signals
        r_Ia_s = self.sigmoid(r_Ia)

        # 3. Combine SC interneuron output with RL policy action
        #    FIX: clamp r_mn to [0, 1] before passing to OpenSim.
        #    Previously sigmoid(...) + action was unbounded; values outside
        #    [0, 1] are invalid muscle activations and cause sim instability.
        r_mn       = np.clip(self.sigmoid(np.matmul(self.W_SC, r_Ia_s)) + action, 0.0, 1.0)
        self._r_mn = r_mn
        # ----------------------------------------------------------------

        obs, reward, done, info = super(Arm2DVecEnv, self).step(r_mn, obs_as_dict=obs_as_dict)

        if np.isnan(obs).any():
            obs     = np.nan_to_num(obs)
            done    = True
            reward -= 10   # FIX: was `reward - 10` (no-op expression)

        return obs, reward, done, info
