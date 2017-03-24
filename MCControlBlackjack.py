# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 19:45:48 2017

@author: user
"""

import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict

if "../" not in sys.path:
  sys.path.append("../") 
from blackjack import BlackjackEnv
import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()

def epsilon_greedy_policy(Q, epsilon, nA, Ns,N0):

    def policy_fn(observation):
        curr_epsilon=N0/(N0+Ns[observation])
        A = np.ones(nA, dtype=float) * curr_epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - curr_epsilon)
        return A
    return policy_fn
    
def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    Ns=defaultdict(float)
    
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    policy = epsilon_greedy_policy(Q, epsilon, env.action_space.n, Ns, 100)
    
    for i in range(1, num_episodes + 1):
        episode = []
        state = env.reset()
        Ns[state] += 1.0
        for t in range(100):
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            Ns[next_state] += 1.0
            if done:
                break
            state = next_state

        sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
        for state, action in sa_in_episode:
            sa_pair = (state, action)

            first_occurence_idx = next(i for i,x in enumerate(episode)
                                       if x[0] == state and x[1] == action)

            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])

            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
        
    return Q, policy
    
Q, policy = mc_control_epsilon_greedy(env, num_episodes=3000, epsilon=0.1)
    
V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value

plotting.plot_value_function(V, title="Optimal Value Function")


