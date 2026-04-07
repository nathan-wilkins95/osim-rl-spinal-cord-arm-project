"""train_arm.py -- Baseline MS arm training/testing script (SB3 DDPG).

Usage
-----
    # Train from scratch (200k steps recommended)
    python train_arm.py --train --steps 200000

    # Test a saved model
    python train_arm.py --test --model models/train_arm_MS.zip

    # Train with live visualisation (slow)
    python train_arm.py --train --steps 200000 --visualize

Requires
--------
    opensim >= 4.4, stable-baselines3 >= 2.3, gymnasium >= 0.29
"""

import numpy as np
import os
import sys
import datetime as dt
import pickle
import argparse
import matplotlib.pyplot as plt

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from osim.env.arm import Arm2DVecEnv
from test_MS_agents import test_agents

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUMBER_OF_STEPS      = 200
NUMBER_OF_ITERATIONS = 5
CONVERT_RAD_TO_DEG   = True

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PATH_FIG_BASE = os.path.join(BASE_DIR, 'figures', 'MS_figures')
PATH_PICKLE   = os.path.join(PATH_FIG_BASE, 'd_combined_states.pkl')
DEFAULT_MODEL = os.path.join(BASE_DIR, 'models', 'train_arm_MS.zip')

parser = argparse.ArgumentParser(description='Train or test MS arm motor controller')
parser.add_argument('--train',     dest='train',     action='store_true',  default=False)
parser.add_argument('--test',      dest='train',     action='store_false')
parser.add_argument('--steps',     dest='steps',     action='store',       default=200000, type=int)
parser.add_argument('--visualize', dest='visualize', action='store_true')
parser.add_argument('--model',     dest='model',     action='store',       default=DEFAULT_MODEL)
args = parser.parse_args()

os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
os.makedirs(PATH_FIG_BASE, exist_ok=True)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
env = Arm2DVecEnv(visualize=args.visualize)
env.reset()

nb_actions = env.action_space.shape[0]

# ---------------------------------------------------------------------------
# DDPG agent (SB3)
# Actor:  3 x 256 (larger than before -- SB3 scales well)
# Critic: 3 x 256
# OU noise: theta=0.15, sigma=0.2 (same as keras-rl baseline)
# ---------------------------------------------------------------------------
action_noise = OrnsteinUhlenbeckActionNoise(
    mean=np.zeros(nb_actions),
    sigma=0.2 * np.ones(nb_actions),
    theta=0.15
)

policy_kwargs = dict(net_arch=[256, 256, 256])

agent = DDPG(
    policy='MlpPolicy',
    env=env,
    learning_rate=1e-3,
    buffer_size=500000,
    learning_starts=1000,
    batch_size=256,
    gamma=0.99,
    tau=1e-3,
    action_noise=action_noise,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=os.path.join(BASE_DIR, 'logs', 'MS'),
)

print(f'Observation space: {env.observation_space.shape}')  # should be (34,)
print(f'Action space:      {env.action_space.shape}')       # should be (6,)

# ---------------------------------------------------------------------------
# Train or Test
# ---------------------------------------------------------------------------
if args.train:
    checkpoint_cb = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(BASE_DIR, 'models', 'checkpoints_MS'),
        name_prefix='train_arm_MS'
    )
    agent.learn(
        total_timesteps=args.steps,
        callback=checkpoint_cb,
        log_interval=100
    )
    agent.save(args.model)
    print(f'Model saved to {args.model}')

else:
    if not os.path.exists(args.model) and not os.path.exists(args.model + '.zip'):
        print(f'[ERROR] Model not found: {args.model}')
        print('  Train first with: python train_arm.py --train --steps 200000')
        sys.exit(1)

    agent = DDPG.load(args.model, env=env)
    print(f'Loaded model from {args.model}')

    list_results, list_d_states = test_agents(agent, env, NUMBER_OF_STEPS, NUMBER_OF_ITERATIONS)

    d_combined_states = {key: [] for key in list_d_states[0].keys()}
    idx_best_run      = 0
    reward_best       = -1e99

    for k, (results, d_states) in enumerate(zip(list_results, list_d_states)):
        print(f'Episode [{k + 1}/{len(list_results)}]')
        for key, item in d_states.items():
            d_combined_states[key] = d_combined_states[key] + item
        reward_mean = np.mean(results['rewards'])
        if reward_mean > reward_best:
            idx_best_run = k
            reward_best  = reward_mean

    for key, item in d_combined_states.items():
        print(f'  {key} = {np.mean(item):.4f} +/- {np.std(item, ddof=1):.4f}')

    cci_vals  = d_combined_states.get('CCI', [])
    rdiv_vals = d_combined_states.get('recruitment_diversity', [])
    if cci_vals:
        print(f'\n--- Validation Metrics ---')
        print(f'  Mean CCI:                   {np.mean(cci_vals):.4f}  (target < 0.1)')
        print(f'  Mean recruitment_diversity: {np.mean(rdiv_vals):.4f}  (target > 0.1)')
        print(f'  Max  CCI:                   {np.max(cci_vals):.4f}')

    d_states_best = list_d_states[idx_best_run]
    results_best  = list_results[idx_best_run]

    with open(PATH_PICKLE, 'wb') as fp:
        pickle.dump(d_combined_states, fp)

    y_label_dict = {
        'pos': 'deg', 'velocity': 'm/s', 'vel': 'deg/s',
        'acc': 'deg/s^2', 'length': 'm', 'activation': '[-]',
        'CCI': '[-]', 'diversity': '[-]',
    }

    if CONVERT_RAD_TO_DEG:
        for key, item in d_combined_states.items():
            for key_label, val_label in y_label_dict.items():
                if key_label in key and 'deg' in val_label:
                    d_combined_states[key] = list(np.rad2deg(item))
                    break

    folder_date = dt.datetime.strftime(dt.datetime.now(), '%Y%m%dT%H%M')
    path_fig    = os.path.join(PATH_FIG_BASE, folder_date)
    os.makedirs(path_fig, exist_ok=True)

    rew = results_best['rewards']
    obs = results_best['obs']

    nn = len(rew)
    t  = np.linspace(1, nn, nn)
    plt.figure()
    plt.plot(t, rew)
    plt.grid()
    plt.ylim((0, 1))
    plt.xlabel('time [-]')
    plt.ylabel('rew [-]')
    plt.savefig(os.path.join(path_fig, 'MS_model_rew.png'))
    plt.close()

    obs     = np.array(obs)
    targets = obs[:, 0:2]
    act_pos = obs[:, -2:]
    c_time  = np.linspace(0, 1, targets.shape[0])
    plt.figure()
    sc = plt.scatter(act_pos[:, 0], act_pos[:, 1], s=10, c=c_time, cmap='jet', label='wrist')
    plt.scatter(targets[:, 0], targets[:, 1], marker='x', label='target')
    plt.colorbar(sc, label='time')
    plt.grid()
    plt.ylim((-1, 1))
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.savefig(os.path.join(path_fig, 'MS_model_obs.png'))
    plt.close()

    if cci_vals:
        plt.figure()
        plt.plot(cci_vals)
        plt.axhline(0.1, color='r', linestyle='--', label='target < 0.1')
        plt.grid()
        plt.ylim((0, 0.6))
        plt.xlabel('time [-]')
        plt.ylabel('CCI [-]')
        plt.title('Co-Contraction Index (MS)')
        plt.legend()
        plt.savefig(os.path.join(path_fig, 'MS_model_CCI.png'))
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
        plt.savefig(os.path.join(path_fig, f'MS_model_{key}.png'))
        plt.close()

    print(f'Plots saved to {path_fig}')
