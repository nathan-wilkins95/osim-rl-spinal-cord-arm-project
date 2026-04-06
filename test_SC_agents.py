from copy import deepcopy
import numpy as np


def test_agent(agent, env, nb_max_episode_steps=200):
    """Run a single test episode and return per-step results and state logs.

    Returns
    -------
    results : dict
        'rewards'       -- list of per-step rewards
        'obs'           -- list of per-step observations
        'r_Ia'          -- list of per-step normalised Ia afferent rate arrays
        'r_mn'          -- list of per-step motor neuron activation arrays
        'episode_reward'-- float, total accumulated reward for the episode
    d_states : dict
        Per-step log of all state variables returned by env.get_d_state().
    """
    results = {
        'rewards': [],
        'obs': [],
        'r_Ia': [],
        'r_mn': [],
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

        d_state = env.get_d_state(action)
        for key, item in d_state.items():
            # FIX: always wrap the initial value in a list regardless of type.
            # Previously, scalar items (e.g. joint angle floats) were stored
            # directly on first encounter, causing an AttributeError on the
            # second step when .append() was called on a float.
            if isinstance(item, list) and len(item) == 1:
                val = item[0]
            else:
                val = item

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

        episode_step += 1
        agent.step   += 1

    # Terminal step: one more forward-backward with terminal=False
    # (next state is always non-terminal by convention)
    agent.forward(observation)
    agent.backward(0., terminal=False)

    # FIX: store total episode reward so callers can use it directly
    results['episode_reward'] = episode_reward

    return results, d_states


def test_agents(agent, env, nb_max_episode_steps=200, iterations=5):
    """Run multiple test episodes and return all results."""
    list_results  = []
    list_d_states = []
    for k in range(iterations):
        results, d_states = test_agent(agent, env, nb_max_episode_steps)
        list_results.append(results)
        list_d_states.append(d_states)
    return list_results, list_d_states
