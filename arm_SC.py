import math
import numpy as np
import os
from .utils.mygym import convert_to_gym
import gym
import opensim
import random
from .osim import OsimEnv


class Arm2DEnv(OsimEnv):
    model_path = os.path.join(os.path.dirname(__file__), '../models/arm2dof6musc.osim')
    time_limit = 200
    target_x = 0
    target_y = 0

    def get_aux_info(self):
        return self._r_Ia, self._r_mn

    def get_d_state(self, action):
        state_desc = self.get_state_desc()
        d = {}
        # d["ball_x"] = self.target_x
        # d["ball_y"] = self.target_y


        # for body_part in ["r_humerus", "r_ulna_radius_hand"]:
        #     res += state_desc["body_pos"][body_part][0:2]
        #     res += state_desc["body_vel"][body_part][0:2]
        #     res += state_desc["body_acc"][body_part][0:2]
        #     res += state_desc["body_pos_rot"][body_part][2:]
        #     res += state_desc["body_vel_rot"][body_part][2:]
        #     res += state_desc["body_acc_rot"][body_part][2:]

        #for body_part in ["r_humerus", "r_ulna_radius_hand"]:
        for body_part in ["r_ulna_radius_hand"]:
            d[f'{body_part}_pos_0'] = state_desc["body_pos"][body_part][0]
            d[f'{body_part}_pos_1'] = state_desc["body_pos"][body_part][1]
            d[f'{body_part}_vel_0'] = state_desc["body_vel"][body_part][0]
            d[f'{body_part}_vel_1'] = state_desc["body_vel"][body_part][1]
            d[f'{body_part}_acc_0'] = state_desc["body_acc"][body_part][0]
            d[f'{body_part}_acc_1'] = state_desc["body_acc"][body_part][1]
        #     res += state_desc["body_pos_rot"][body_part][2]
        #     res += state_desc["body_vel_rot"][body_part][2]
        #     res += state_desc["body_acc_rot"][body_part][2]


        for joint in ["r_shoulder", "r_elbow",]:
            d[f'{joint}_pos'] = state_desc["joint_pos"][joint]
            d[f'{joint}_vel'] = state_desc["joint_vel"][joint]
            d[f'{joint}_acc'] = state_desc["joint_acc"][joint]

        for muscle in sorted(state_desc["muscles"].keys()):
            d[f'{muscle}_activation'] = state_desc["muscles"][muscle]["activation"]
            d[f'{muscle}_fiber_length'] = state_desc["muscles"][muscle]["fiber_length"]
            d[f'{muscle}_fiber_velocity'] = state_desc["muscles"][muscle]["fiber_velocity"]

        d["markers_0"] = state_desc["markers"]["r_radius_styloid"]["pos"][0]
        d["markers_1"] = state_desc["markers"]["r_radius_styloid"]["pos"][1]

        return d

    def get_observation(self):
        state_desc = self.get_state_desc()

        res = [self.target_x, self.target_y]

        # for body_part in ["r_humerus", "r_ulna_radius_hand"]:
        #     res += state_desc["body_pos"][body_part][0:2]
        #     res += state_desc["body_vel"][body_part][0:2]
        #     res += state_desc["body_acc"][body_part][0:2]
        #     res += state_desc["body_pos_rot"][body_part][2:]
        #     res += state_desc["body_vel_rot"][body_part][2:]
        #     res += state_desc["body_acc_rot"][body_part][2:]

        for joint in ["r_shoulder", "r_elbow", ]:
            res += state_desc["joint_pos"][joint]
            res += state_desc["joint_vel"][joint]
            res += state_desc["joint_acc"][joint]

        for muscle in sorted(state_desc["muscles"].keys()):
            res += [state_desc["muscles"][muscle]["activation"]]
           # res += [state_desc["muscles"][muscle]["fiber_length"]]
           # res += [state_desc["muscles"][muscle]["fiber_velocity"]]

        res += state_desc["markers"]["r_radius_styloid"]["pos"][:2]

        return res

    def get_observation_space_size(self):
        return  16  # 46

    def generate_new_target(self):
        theta = random.uniform(math.pi * 0, math.pi * 2 / 3)
        radius = random.uniform(0.3, 0.65)
        self.target_x = math.cos(theta) * radius
        self.target_y = -math.sin(theta) * radius + 0.8
        self.target_x = 0.16056636337579086
        self.target_y = 0.49340151308159397

        print('\ntarget: [{} {}]'.format(self.target_x, self.target_y))

        state = self.osim_model.get_state()

        #        self.target_joint.getCoordinate(0).setValue(state, self.target_x, False)
        self.target_joint.getCoordinate(1).setValue(state, self.target_x, False)

        self.target_joint.getCoordinate(2).setLocked(state, False)
        self.target_joint.getCoordinate(2).setValue(state, self.target_y, False)
        self.target_joint.getCoordinate(2).setLocked(state, True)
        self.osim_model.set_state(state)

    def reset(self, random_target=True, obs_as_dict=True):
        obs = super(Arm2DEnv, self).reset(obs_as_dict=obs_as_dict)
        #self.r_mn_state = 0.05 * np.ones(6)

        if random_target:
            self.generate_new_target()
        self.osim_model.reset_manager()
        return obs

    def __init__(self, *args, **kwargs):
        self._r_Ia = None
        self._r_mn = None
        super(Arm2DEnv, self).__init__(*args, **kwargs)
        blockos = opensim.Body('target', 0.0001, opensim.Vec3(0), opensim.Inertia(1, 1, .0001, 0, 0, 0));
        self.target_joint = opensim.PlanarJoint('target-joint',
                                                self.osim_model.model.getGround(),  # PhysicalFrame
                                                opensim.Vec3(0, 0, 0),
                                                opensim.Vec3(0, 0, 0),
                                                blockos,  # PhysicalFrame
                                                opensim.Vec3(0, 0, -0.25),
                                                opensim.Vec3(0, 0, 0))


        self.W_SC   = np.array([[1, 0, 0, -0.5, 0, 0],
                                [0, 1, 0, 0, -0.5, -0.5],
                                [0, 0, 1, 0, -0.5, -0.5],
                                [-0.5, 0, 0, 1, 0, 0],
                                [0, -0.5, -0.5, 0, 1, 0],
                                [0, -0.5, -0.5, 0, 0, 1]])

        #self.W_SC_float = self.W_SC.astype(float)




        self.noutput = self.osim_model.noutput

        geometry = opensim.Ellipsoid(0.02, 0.02, 0.02);
        geometry.setColor(opensim.Green);
        blockos.attachGeometry(geometry)

        self.osim_model.model.addJoint(self.target_joint)
        self.osim_model.model.addBody(blockos)

        self.osim_model.model.initSystem()

        # initialize motor neuron activation r
        #self.r_mn_state = 0.05*np.ones(6)

    def reward(self):
        state_desc = self.get_state_desc()
        penalty = (state_desc["markers"]["r_radius_styloid"]["pos"][0] - self.target_x) ** 2 + (
                    state_desc["markers"]["r_radius_styloid"]["pos"][1] - self.target_y) ** 2
        # print(state_desc["markers"]["r_radius_styloid"]["pos"])
        # print((self.target_x, self.target_y))
        if np.isnan(penalty):
            penalty = 1
        return 1. - penalty

    def get_reward(self):
        return self.reward()


class Arm2DVecEnv(Arm2DEnv):

    def Prochazka_Ia_rates(self, a=4.3, b=2, c=10):
        """ Compute Prochazka Ia rates """
        self.prev_state_desc = self.get_state_desc()

        norm_rate = np.zeros(6)

        for i in range(self.osim_model.muscleSet.getSize()):
            muscle = self.osim_model.muscleSet.get(i)
            name = muscle.getName()

            fiber_l = self.prev_state_desc["muscles"][name]["fiber_length"] * 1000
            fiber_v = self.prev_state_desc["muscles"][name]["fiber_velocity"] * 1000
            opt_l = muscle.getOptimalFiberLength() * 1000
            max_v = muscle.getMaxContractionVelocity() * opt_l
    #        if self.past_fiber_l is None:
    #            self.past_fiber_l = opt_l
    #            fiber_v = 0.001
    #        else:
            #fiber_v = (fiber_l - self.past_fiber_l) / model.step_size
            #self.past_fiber_l = fiber_l
            rate = a * np.sign(fiber_v) * np.exp(0.6 * np.log(max(min(abs(fiber_v), max_v), 0.01))) + \
                   b * (min(fiber_l, 1.5 * opt_l) - opt_l) + c
            norm_rate[i] = max(rate / (a * np.exp(0.6 * np.log(max_v)) + b * 0.5 * opt_l + c), 0)

        #: Update past rates
        #self.past_Ia_rates[0:-1] = self.past_Ia_rates[1:]
        #self.past_Ia_rates[-1] = norm_rate
        return norm_rate

    def sigmoid(self, x, alpha=8, beta=0.5):
        """ Sigmoid function """
        return 1 / (1 + np.exp(-alpha * (x - beta)))

    def reset(self, obs_as_dict=False):
        obs = super(Arm2DVecEnv, self).reset(obs_as_dict=obs_as_dict)
        if np.isnan(obs).any():
            obs = np.nan_to_num(obs)
        return obs

    def step(self, action, obs_as_dict=False):
        if np.isnan(action).any():
            action = np.nan_to_num(action)
        # spinal cord model
        #r_mn        = self.r_mn_state
        r_Ia        = self.Prochazka_Ia_rates()
        self._r_Ia = r_Ia
        r_Ia_s      = self.sigmoid(r_Ia)
        #r_Ia_       = np.array([[r_Ia[0], r_Ia_s[0], r_Ia_s[0], r_Ia_s[0], r_Ia_s[0], r_Ia_s[0]],
                                #[r_Ia_s[1], r_Ia[1], r_Ia_s[1], r_Ia_s[1], r_Ia_s[1], r_Ia_s[1]],
                                #[r_Ia_s[2], r_Ia_s[2], r_Ia[2], r_Ia_s[2], r_Ia_s[2], r_Ia_s[2]],
                                #[r_Ia_s[3], r_Ia_s[3], r_Ia_s[3], r_Ia[3], r_Ia_s[3], r_Ia_s[3]],
                                #[r_Ia_s[4], r_Ia_s[4], r_Ia_s[4], r_Ia_s[4], r_Ia[4], r_Ia_s[4]],
                                #[r_Ia_s[5], r_Ia_s[5], r_Ia_s[5], r_Ia_s[5], r_Ia_s[5], r_Ia[5]]])

        #r_Ia_float = r_Ia_.astype(float)


        #sc_dot = np.matmul(self.W_SC_vect, r_Ia_vect)
        #r_mn_dot = (- r_mn + self.sigmoid((np.diag(np.matmul(self.W_SC, r_Ia_))) + action))/0.01
        r_mn = self.sigmoid(np.matmul(self.W_SC, r_Ia_s))+action
        self._r_mn = r_mn
        # print(r_mn)
        obs, reward, done, info = super(Arm2DVecEnv, self).step(r_mn, obs_as_dict=obs_as_dict)
        #obs += list(r_mn)

        #self.r_mn_state = r_mn
        if np.isnan(obs).any():
            obs = np.nan_to_num(obs)
            done = True
            reward - 10
        return obs, reward, done, info