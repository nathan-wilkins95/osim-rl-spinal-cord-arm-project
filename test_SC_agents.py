from copy import deepcopy
import numpy as np


def test_agent(agent, env, nb_max_episode_steps=200):
    """Run a single test episode (SC model).

    Returns
    -------
    results : dict
        'rewards'        -- per-step reward list
        'obs'            -- per-step observation list
        'r_Ia'           -- per-step Ia afferent rate arrays
        'r_mn'           -- per-step motor neuron activation arrays
        'reward_info'    -- per-step reward component dicts
        'episode_reward' -- float, total accumulated episode reward
    d_states : dict
        Per-step state log including CCI and recruitment_diversity.
    """
    results = {
        'rewards':        [],
        'obs':            [],
        'r_Ia':           [],
        'r_mn':           [],
        'reward_info':    [],
        'episode_reward': 0.0,
    }

    agent.reset_states()
    observation = deepcopy(env.reset())

    episode_reward = 0.0
    episode_step   = 0
    done           = False
    d_states       = dict()

    while not done:
        action = agent.forward(observation)
        reward = 0.

        observation, r, d, info = env.step(action)

        _r_Ia, _r_mn = env.get_aux_info()
        reward_info  = info if isinstance(info, dict) and 'reward_dist' in info else {}

        d_state = env.get_d_state(action)
        for key, item in d_state.items():
            val = item[0] if (isinstance(item, list) and len(item) == 1) else item
            if key in d_states:
                d_states[key].append(val)
            else:
                d_states[key] = [val]

        observation = deepcopy(observation)

        reward += r
        if d:
            done = True
            break
        if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
            done = True

        agent.backward(reward, terminal=done)
        episode_reward += reward

        results['rewards'].append(reward)
        results['obs'].append(observation)
        results['r_mn'].append(_r_mn)
        results['r_Ia'].append(_r_Ia)
        results['reward_info'].append(reward_info)

        episode_step += 1
        agent.step   += 1

    agent.forward(observation)
    agent.backward(0., terminal=False)

    results['episode_reward'] = episode_reward
    return results, d_states


def test_agents(agent, env, nb_max_episode_steps=200, iterations=5):
    """Run multiple test episodes."""
    list_results  = []
    list_d_states = []
    for k in range(iterations):
        results, d_states = test_agent(agent, env, nb_max_episode_steps)
        list_results.append(results)
        list_d_states.append(d_states)
    return list_results, list_d_states
