# Derived from keras-rl
import opensim as osim
import numpy as np
import sys
import os
import datetime as dt
import pickle
import argparse
import math

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from osim.env.arm_SC import Arm2DVecEnv
from examples.test_SC_agents import test_agent, test_agents
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# NOTE: observation space = 34. Old obs=16 weights must be retrained.
# To train:  python train_arm_SC.py --train --steps 100000
# ---------------------------------------------------------------------------

number_of_steps      = 200
number_of_iterations = 5
print_best           = False
convert_rad_to_deg   = True

BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
PATH_FIG_BASE       = os.path.join(BASE_DIR, 'figures', 'SC_figures')
PATH_PICKLE_STATES  = os.path.join(PATH_FIG_BASE, 'd_combined_states.pkl')
PATH_PICKLE_RESULTS = os.path.join(PATH_FIG_BASE, 'list_results.pkl')

parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train',     dest='train',     action='store_true',  default=False)
parser.add_argument('--test',      dest='train',     action='store_false')
parser.add_argument('--steps',     dest='steps',     action='store',       default=100000, type=int)
parser.add_argument('--visualize', dest='visualize', action='store_true')
parser.add_argument('--model',     dest='model',     action='store',       default=os.path.join(BASE_DIR, 'train_SC21.h5f'))
args = parser.parse_args()

env = Arm2DVecEnv(visualize=args.visualize)
env.reset()

nb_actions = env.action_space.shape[0]
nallsteps  = args.steps

# ---------------------------------------------------------------------------
# DDPG networks — scaled up to 3x64 actor / 3x128 critic
# ---------------------------------------------------------------------------
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(64))
actor.add(Activation('relu'))
actor.add(Dense(64))
actor.add(Activation('relu'))
actor.add(Dense(64))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('sigmoid'))
print(actor.summary())

action_input      = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_obs     = Flatten()(observation_input)
x = concatenate([action_input, flattened_obs])
x = Dense(128)(x)
x = Activation('relu')(x)
x = Dense(128)(x)
x = Activation('relu')(x)
x = Dense(128)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

memory         = SequentialMemory(limit=500000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=env.noutput)
agent = DDPGAgent(
    nb_actions=nb_actions, actor=actor, critic=critic,
    critic_action_input=action_input,
    memory=memory,
    nb_steps_warmup_critic=1000,
    nb_steps_warmup_actor=1000,
    random_process=random_process,
    gamma=.99, target_model_update=1e-3, delta_clip=1.
)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

if args.train:
    agent.fit(env, nb_steps=nallsteps, visualize=False, verbose=1,
              nb_max_episode_steps=200, log_interval=10000)
    agent.save_weights(args.model, overwrite=True)

else:
    try:
        agent.load_weights(args.model)
    except Exception as e:
        print(f"[WARNING] Could not load weights from {args.model}: {e}")
        print("  Observation space changed from 16 -> 34. Retrain with --train --steps 100000")
        sys.exit(1)

    list_results, list_d_states = test_agents(agent, env, number_of_steps, number_of_iterations)

    d_combined_states = {key: [] for key in list_d_states[0].keys()}
    idx_best_run      = 0
    reward_best       = -1e99

    for k, (results, d_states) in enumerate(zip(list_results, list_d_states)):
        print(f'[{k + 1}/{len(list_results)}]')
        for key, item in d_states.items():
            d_combined_states[key] = d_combined_states[key] + item
        reward_mean = np.mean(results['rewards'])
        if reward_mean > reward_best:
            idx_best_run = k
            reward_best  = reward_mean

    for key, item in d_combined_states.items():
        print(f'{key} = {np.mean(item):.4f}+/-{np.std(item, ddof=1):.4f}')

    # Print validation summary
    cci_vals  = d_combined_states.get('CCI', [])
    rdiv_vals = d_combined_states.get('recruitment_diversity', [])
    if cci_vals:
        print(f'\n--- Validation Metrics ---')
        print(f'Mean CCI:                  {np.mean(cci_vals):.4f}  (target < 0.1)')
        print(f'Mean recruitment_diversity:{np.mean(rdiv_vals):.4f}  (target > 0.1)')
        print(f'Max  CCI:                  {np.max(cci_vals):.4f}')

    d_states_best = list_d_states[idx_best_run]
    results_best  = list_results[idx_best_run]

    os.makedirs(PATH_FIG_BASE, exist_ok=True)
    with open(PATH_PICKLE_STATES, 'wb') as fp:
        pickle.dump(d_states_best, fp)
    with open(PATH_PICKLE_RESULTS, 'wb') as fp:
        pickle.dump(results_best, fp)

    y_label_dict = {
        "pos": "deg", "velocity": "m/s", "vel": "deg/s",
        "acc": "deg/s^2", "length": "m", "activation": "[-]",
        "CCI": "[-]", "diversity": "[-]",
    }

    rew  = results_best['rewards']
    obs  = results_best['obs']
    r_mn = results_best.get('r_mn', None)
    r_Ia = results_best.get('r_Ia', None)

    if convert_rad_to_deg:
        for key, item in d_combined_states.items():
            item_label = ""
            for key_label, val_label in y_label_dict.items():
                if key_label in key:
                    item_label = val_label
                    break
            if "deg" in item_label:
                d_combined_states[key] = list(np.rad2deg(item))

    folder_date = dt.datetime.strftime(dt.datetime.now(), "%Y%m%dT%H%M")
    path_fig    = os.path.join(PATH_FIG_BASE, folder_date)
    os.makedirs(path_fig, exist_ok=True)

    nn = len(rew)
    t  = np.linspace(1, nn, nn)
    plt.figure()
    plt.plot(t, rew)
    plt.grid()
    plt.ylim((0, 1))
    plt.xlabel('time [-]')
    plt.ylabel('rew [-]')
    plt.savefig(os.path.join(path_fig, 'SC_model_rew.png'))
    plt.close()

    obs     = np.array(obs)
    targets = obs[:, 0:2]
    act_pos = obs[:, -2:]
    c_time  = np.linspace(0, 1, targets.shape[0])
    plt.figure()
    plt.scatter(targets[:, 0], targets[:, 1], marker='x', label='target')
    plt.scatter(act_pos[:, 0], act_pos[:, 1], s=10, c=c_time, cmap='jet', label='wrist')
    plt.grid()
    plt.ylim((-1, 1))
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.savefig(os.path.join(path_fig, 'SC_model_obs.png'))
    plt.close()

    # CCI over time
    if cci_vals:
        plt.figure()
        plt.plot(cci_vals)
        plt.axhline(0.1, color='r', linestyle='--', label='target < 0.1')
        plt.grid()
        plt.ylim((0, 0.6))
        plt.xlabel('time [-]')
        plt.ylabel('CCI [-]')
        plt.title('Co-Contraction Index')
        plt.legend()
        plt.savefig(os.path.join(path_fig, 'SC_model_CCI.png'))
        plt.close()

    for key, item in d_combined_states.items():
        fig, ax = plt.subplots()
        ax.plot(item)
        ax.grid()
        ax.set_title(key)
        ax.set_xlabel('time [-]')
        y_label_text = 'y'
        for key_label, val_label in y_label_dict.items():
            if key_label in key:
                y_label_text = val_label
                break
        ax.set_ylabel(y_label_text)
        plt.savefig(os.path.join(path_fig, f'SC_model_{key}.png'))
        plt.close()

    muscle_names = ['BIClong', 'BICshort', 'BRA', 'TRIlong', 'TRIlat', 'TRImed']

    if r_mn is not None:
        r_mn = np.array(r_mn)
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        for i, ax in enumerate(axes.flat):
            ax.plot(r_mn[:, i] if r_mn.ndim > 1 else [r_mn[i]])
            ax.set_title(f'r_mn: {muscle_names[i]}')
            ax.set_xlabel('time [-]')
            ax.set_ylabel('activation [-]')
            ax.set_ylim((0, 1))
            ax.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(path_fig, 'SC_model_r_mn.png'))
        plt.close()

    if r_Ia is not None:
        r_Ia = np.array(r_Ia)
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        for i, ax in enumerate(axes.flat):
            ax.plot(r_Ia[:, i] if r_Ia.ndim > 1 else [r_Ia[i]])
            ax.set_title(f'r_Ia: {muscle_names[i]}')
            ax.set_xlabel('time [-]')
            ax.set_ylabel('norm. rate [-]')
            ax.set_ylim((0, 1))
            ax.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(path_fig, 'SC_model_r_Ia.png'))
        plt.close()
