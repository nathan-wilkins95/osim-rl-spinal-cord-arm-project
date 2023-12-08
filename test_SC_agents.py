from copy import deepcopy
import numpy as np


def test_agent(agent, env, nb_max_episode_steps=200):
    """documentation"""

    results = {'rewards': [],
               'obs': []}
    # Obtain the initial observation by resetting the environment.
    agent.reset_states()
    observation = deepcopy(env.reset())


    # Run the episode until we're done.
    episode_reward = 0.
    episode_step = 0
    done = False
    d_states = dict()
    while not done:
        #print('>>', episode_step)
        action = agent.forward(observation)
        reward = 0.

        observation, r, d, info = env.step(action)
        d_state = env.get_d_state(action)
        for key, item in d_state.items():
            if key in d_states.keys():
                if isinstance(item, list) and len(item) == 1:
                    d_states[key].append(item[0])
                else:
                    d_states[key].append(item)
            else:
                if isinstance(item, list) and len(item) == 1:
                    d_states[key] = item
                else:
                    d_states[key] = [item]

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
        #results['r_shoulder'].append()



        episode_step += 1
        agent.step += 1

    # We are in a terminal state but the agent hasn't yet seen it. We therefore
    # perform one more forward-backward call and simply ignore the action before
    # resetting the environment. We need to pass in `terminal=False` here since
    # the *next* state, that is the state of the newly reset environment, is
    # always non-terminal by convention.
    agent.forward(observation)
    agent.backward(0., terminal=False)

    return results, d_states


def test_agents(agent, env, nb_max_episode_steps=200, iterations=5):

    list_results = []
    list_d_states = []
    for k in range(iterations):
        results, d_states = test_agent(agent, env, nb_max_episode_steps)

        list_results.append(results)
        list_d_states.append(d_states)

    return list_results, list_d_states