"""test_MS_agents.py -- Test harness for baseline MS arm agent (SB3 DDPG compatible)."""

from copy import deepcopy
import numpy as np


def test_agent(agent, env, nb_max_episode_steps=200):
    """Run a single test episode using the baseline MS arm environment.

    Parameters
    ----------
    agent : stable_baselines3.DDPG
        A trained SB3 DDPG agent.
    env : Arm2DVecEnv
        The MS arm environment instance.
    nb_max_episode_steps : int, optional
        Maximum steps per episode (default 200).

    Returns
    -------
    results : dict
        Keys: 'rewards', 'obs', 'reward_info', 'episode_reward'.
    d_states : dict
        Per-step state log including CCI and recruitment_diversity.
    """
    results = {
        'rewards':        [],
        'obs':            [],
        'reward_info':    [],
        'episode_reward': 0.0,
    }

    observation, _ = env.reset() if hasattr(env.reset(), '__len__') and len(env.reset()) == 2 \
        else (env.reset(), {})
    observation = np.array(observation, dtype=np.float32)

    episode_reward = 0.0
    episode_step   = 0
    done           = False
    d_states       = {}

    while not done:
        action, _ = agent.predict(observation, deterministic=True)

        step_result = env.step(action)
        if len(step_result) == 5:
            observation, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            observation, reward, done, info = step_result

        observation = np.array(observation, dtype=np.float32)
        reward_info = info if isinstance(info, dict) and 'reward_dist' in info else {}

        d_state = env.get_d_state(action)
        for key, item in d_state.items():
            val = item[0] if (isinstance(item, list) and len(item) == 1) else item
            if key in d_states:
                d_states[key].append(val)
            else:
                d_states[key] = [val]

        episode_reward += reward
        results['rewards'].append(float(reward))
        results['obs'].append(observation.tolist())
        results['reward_info'].append(reward_info)

        episode_step += 1
        if nb_max_episode_steps and episode_step >= nb_max_episode_steps:
            done = True

    results['episode_reward'] = episode_reward
    return results, d_states


def test_agents(agent, env, nb_max_episode_steps=200, iterations=5):
    """Run multiple test episodes and return all results.

    Parameters
    ----------
    agent : stable_baselines3.DDPG
        A trained SB3 DDPG agent.
    env : Arm2DVecEnv
        The MS arm environment instance.
    nb_max_episode_steps : int, optional
        Maximum steps per episode (default 200).
    iterations : int, optional
        Number of test episodes to run (default 5).

    Returns
    -------
    list_results : list of dict
    list_d_states : list of dict
    """
    list_results  = []
    list_d_states = []
    for k in range(iterations):
        print(f'  Test episode {k + 1}/{iterations} ...')
        results, d_states = test_agent(agent, env, nb_max_episode_steps)
        print(f'    episode_reward = {results["episode_reward"]:.4f}')
        list_results.append(results)
        list_d_states.append(d_states)
    return list_results, list_d_states
