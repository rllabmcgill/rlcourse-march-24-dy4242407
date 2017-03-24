# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:53:15 2017

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 19:45:48 2017

@author: user
"""

import matplotlib
import numpy as np
import sys

from collections import defaultdict

if "../" not in sys.path:
  sys.path.append("../") 
from easy21game import Easy21
import plotting

matplotlib.style.use('ggplot')

env = Easy21()

def epsilon_greedy_policy(Q, epsilon, nA, Ns,N0):

    def policy_fn(observation):
        curr_epsilon=N0/(N0+Ns[observation])
        A = np.ones(nA, dtype=float) * curr_epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - curr_epsilon)
        return A
    return policy_fn
    
def mc_control(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    
   #Dictionary of returns sum and count
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    Ns=defaultdict(float)
    
    #Dictionary mapping state to action values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    #Create policy based on Q, epsilon, and number of actions
    policy = epsilon_greedy_policy(Q, epsilon, env.action_space.n, Ns, 100)
    
    for i in range(1, num_episodes + 1):
        
        if i % 1000 == 0:
            print "\rEpisode {}/{}.".format(i, num_episodes)
            sys.stdout.flush()
        
        # Generate episode.
        episode = []
        state = env.reset()
        
        for t in range(100):
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            Ns[next_state] += 1.0
            if done:
                break
            state = next_state

        state_action_pair_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
        for state, action in state_action_pair_in_episode:
            state_action_pair = (state, action)
            #First occurence of state action pair in episode
            first_occurence_idx = next(i for i,x in enumerate(episode)
                                       if x[0] == state and x[1] == action)
           #All rewards since first occurence, discounted
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
            
            #Average return calculated
            returns_sum[state_action_pair] += G
            returns_count[state_action_pair] += 1.0
            error=G-Q[state][action]
            step=1.0/returns_count[state_action_pair]
            #Q[state][action] = returns_sum[state_action_pair] / returns_count[state_action_pair]
            Q[state][action]+=step*error
        
        #Q dictionary changed implicitly and policy improved
    
    return Q, policy
    
Q, policy = mc_control(env, num_episodes=100000, epsilon=0.1)

def sample_policy(observation):

    score, dealer_score= observation
    return np.array([1.0, 0.0]) if score >= 18 else np.array([0.0, 1.0])
    

V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value

plotting.plot_value_function(V, title="Optimal Value Function")

rewardSampleArray = [0]*10000

for i in range(0, 10000):
        episode = []
        state = env.reset()
        for t in range(100):
            probs = sample_policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:

                rewardSampleArray[i]=reward
                break
            state = next_state

rewardBestArray = [0]*10000

for i in range(0, 10000):
        
        episode = []
        state = env.reset()
        for t in range(100):
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:

                rewardBestArray[i]=reward
                break
            state = next_state

rewardBestArray=np.array(rewardBestArray)
rewardSampleArray=np.array(rewardSampleArray)

    

