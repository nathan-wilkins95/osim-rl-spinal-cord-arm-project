# Derived from keras-rl
import opensim as osim
import numpy as np
import sys

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate
from keras.optimizers import Adam

import os
import numpy as np
import datetime as dt
import pickle

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from osim.env.arm_SC import Arm2DVecEnv

from keras.optimizers import RMSprop

import argparse
import math

from examples.test_SC_agents import test_agent, test_agents
import matplotlib.pyplot as plt

number_of_steps = 200
number_of_iterations = 5
print_best = False
convert_rad_to_deg = True

# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--steps', dest='steps', action='store', default=20000, type=int)
parser.add_argument('--visualize', dest='visualize', action='store_true')
parser.add_argument('--model', dest='model', action='store', default="example.h5f")
args = parser.parse_args()
args.train = False
args.model = "../train_SC21.h5f"

# set to get observation in array
#def _new_step(self, action, project=True, obs_as_dict=False):
#    return super(Arm2DEnv, self).step(action, project=project, obs_as_dict=obs_as_dict)
#Arm2DEnv.step = _new_step
# Load walking environment
#env = Arm2DVecEnv(args.visualize)
env = Arm2DVecEnv(visualize=False)
env.reset()
#env.reset(verbose=False, logfile='arm_log.txt')

nb_actions = env.action_space.shape[0]

# Total number of steps in training
nallsteps = args.steps

# Create networks for DDPG
# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('sigmoid'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = concatenate([action_input, flattened_observation])
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Set up the agent for training
memory = SequentialMemory(limit=200000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=env.noutput)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3,
                  delta_clip=1.)
# agent = ContinuousDQNAgent(nb_actions=env.noutput, V_model=V_model, L_model=L_model, mu_model=mu_model,
#                            memory=memory, nb_steps_warmup=1000, random_process=random_process,
#                            gamma=.99, target_model_update=0.1)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
if args.train:
    agent.fit(env, nb_steps=nallsteps, visualize=False, verbose=1, nb_max_episode_steps=200, log_interval=20000)
    # After training is done, we save the final weights.
    agent.save_weights(args.model, overwrite=True)

else:
    agent.load_weights(args.model)
    # Finally, evaluate our algorithm for 1 episode.
    #results, d_states = test_agent(agent, env, nb_max_episode_steps=200)
    list_results, list_d_states = test_agents(agent, env, number_of_steps, number_of_iterations)

    # Initialize empty dictionary
    d_combined_states = {key: [] for key in list_d_states[0].keys()}
    # Iterate over all d_states variables
    idx_best_run = 0
    reward_best = -1e99
    for k, (results, d_states) in enumerate(zip(list_results, list_d_states)):
        print(f'[{k + 1}/{len(list_results)}]')
        # Add a list for each key to the combined list (and ultimately do this for all d_states in list_d_states, because of the outer loop)
        for key, item in d_states.items():
            # item here equals d_states[key]
            d_combined_states[key] = d_combined_states[key] + item

        reward_mean = np.mean(results['rewards'])
        if reward_mean > reward_best:
            idx_best_run = k
            reward_best = reward_mean

    # Compute statistics for each variable
    for key, item in d_combined_states.items():
        print(f'{key} = {np.mean(item):.4f}±{np.std(item, ddof=1):.4f}')

    # Select best
    d_state_best = list_d_states[idx_best_run]
    results_best = list_results[idx_best_run]

    # Define y-labels for substrings
    y_label_dict = {"pos": "deg", "velocity": "m/s", "vel": "deg/s", "acc": "deg/s^2", "length": "m", "activation": "N"}

    rew = results_best['rewards']
    obs = results_best['obs']
    r_mn = results_best['r_mn']
    r_Ia = results_best['r_Ia']

    if print_best:
        idx_best = np.argmax(rew)
        print(f"Best episode: {idx_best+1} (index {idx_best})")
        for key, item in d_state_best.items():
            unit = "y"
            for key_label, item_label in y_label_dict.items():
                if key_label in key:
                    unit = item_label
                    break
            print(f"{key}: {item[idx_best]:.4f}±{np.std(item, ddof=1):.4f} {unit}")

    file_pickle = '/home/reluctanthero/Code/osim-rl(1)/examples/figures/SC_figures/d_combined_states.pkl'
    with open(file_pickle, 'wb') as fp:
        pickle.dump(d_state_best, fp)

    file_pickle = '/home/reluctanthero/Code/osim-rl(1)/examples/figures/SC_figures/list_results.pkl'
    with open(file_pickle, 'wb') as fp:
        pickle.dump(results_best, fp)


    if convert_rad_to_deg:
        for key, item in d_state_best.items():
            item_label = ""
            for key_label, item_label in y_label_dict.items():
                if key_label in key:
                    break

            #if any([ele in key for ele in ["pos", 'vel', 'acc']]):
            #if ("pos" in key or "vel" in key or "acc" in key):
            if "deg" in item_label:
                print(f"Convert {key} from radians to degrees")
                item_deg = np.rad2deg(item)
                d_state_best[key] = item_deg



    save_plots = True
    path_fig_base = '/home/reluctanthero/Code/osim-rl(1)/examples/figures/SC_figures'

    folder_date = dt.datetime.strftime(dt.datetime.now(), "%Y%m%dT%H%M")
    path_fig = path_fig_base + "/" + folder_date

    # Make directory if path fig does not exist
    if not os.path.isdir(path_fig):
        os.mkdir(path_fig)

    nn = len(rew)
    t = np.linspace(1, nn, nn)
    plt.figure()
    plt.plot(t, rew)
    plt.grid()
    plt.ylim((0, 1))
    plt.xlabel('time [-]')
    plt.ylabel('rew [-]')
    if save_plots:
        file_name_fig = f'SC_model_rew.png'
        file_fig = path_fig + '/' + file_name_fig
        plt.savefig(file_fig)
        plt.close()
    else:
        plt.show()

    obs = np.array(obs)
    targets = obs[:, 0:2]
    act_pos = obs[:, -2:]
    plt.figure()
    c_time = np.linspace(0, 1, targets.shape[0])
    plt.scatter(targets[:, 0], targets[:, 1], marker='x')
    plt.scatter(act_pos[:, 0], act_pos[:, 1], s=10, c=c_time, cmap="jet")
    plt.grid()
    plt.ylim((-1, 1))
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    if save_plots:
        file_name_fig = f'SC_model_obs.png'
        file_fig = path_fig + '/' + file_name_fig
        plt.savefig(file_fig)
        plt.close()
    else:
        plt.show()

    for key, item in d_state_best.items():
        fig, ax = plt.subplots()
        ax.plot(item)
        ax.grid()
        ax.set_title(key)
        ax.set_xlabel('time [-]')
        y_label_text = "y"
        for key_label, item_label in y_label_dict.items():
            if key_label in key:
                y_label_text = item_label
                break

        ax.set_ylabel(y_label_text)
        if save_plots:
            file_name_fig = f'SC_model{key}.png'
            file_fig = path_fig + '/' + file_name_fig
            plt.savefig(file_fig)
            plt.close()
        else:
            plt.show()

