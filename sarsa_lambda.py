#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 22:08:39 2017

@author: yuedong
"""

import os
os.chdir("/Users/yuedong/Downloads/comp767_easy21/")
#%%

from easy21game import Easy21
import numpy as np
from matplotlib import cm
#%%
env = Easy21()

#%%
class Sarsa_Agent:
    def __init__(self, environment, n0, mlambda):
        #n0 is the numver of episodes we play
        self.n0=float(n0)
        self.env = environment
        self.mlambda = mlambda
        
        # N(s) is the number of times that state s has been visited
        # N(s,a) is the number of times that action a has been selected from state s.
        # Initialize Q(s,a) arbitarily and e(s,a)=0 for all s,a
        # here state=(1,1) both dealer and palyer has 1, then it is stored at
        self.N = np.zeros((self.env.dealer_space,
                           self.env.player_space, 
                           self.env.nA))
        
        self.Q = np.zeros((self.env.dealer_space,
                           self.env.player_space, 
                           self.env.nA))
        
        self.E = np.zeros((self.env.dealer_space,
                           self.env.player_space, 
                           self.env.nA))
        # Initialise the value function to zero. 
        self.V = np.zeros((self.env.dealer_space,
                           self.env.player_space))
        
        self.count_wins = 0
        self.iterations = 0
    
    def get_state_number(self, state):
        # a state not busted
        if state[0]>0 and state[0]<22:
            player_idx = state[0]-1
        # busted, go to terminal state which is never get updated
        else:
            player_idx = 21
        dealer_idx=  state[1] - 1  
        return player_idx, dealer_idx

          # get optimal action based on ε-greedy exploration strategy  
    def fixed_epsilon_greedy_action(self, state, epsilon=0.1):
        player_idx, dealer_idx = self.get_state_number(state)
        # action = 0 stick
        # action = 1 hit
        hit = 1
        stick = 0
        # epsilon greedy policy
        if np.random.random() < epsilon:
            r_action = hit if np.random.random()<0.5 else stick
            return r_action
        else:
            action = np.argmax(self.Q[dealer_idx, player_idx, :])
            return action
      
#              # get optimal action based on ε-greedy exploration strategy
#              # the epsilon varies based on the number of visits
#    def var_epsilon_greedy_action(self, state, epsilon=0.1):
#        dealer_idx = state[0]
#        player_idx = state[1]
#        # action = 0 stick
#        # action = 1 hit
#        hit = 1
#        stick = 0
#        # epsilon greedy policy
#        if np.random.random() < epsilon:
#            r_action = hit if np.random.random()<0.5 else stick
#            return r_action
#        else:
#            action = np.argmax(self.Q[dealer_idx, player_idx, :])
#            return action
        
    def get_action(self, state):
        player_idx, dealer_idx = self.get_state_number(state)
        action = np.argmax(self.Q[dealer_idx, player_idx, :])
        return action
    
    
    
#    def validate(self, iterations):        
#        wins = 0; 
#        # Loop episodes
#        for episode in xrange(iterations):
#
#            s = self.env.get_start_state()
#            
#            while not s.term:
#                # execute action
#                a = self.get_action(s)
#                s, r = self.env.step(s, a)
#            wins = wins+1 if r==1 else wins 
#
#        win_percentage = float(wins)/iterations*100
#        return win_percentage
    
    def train(self, iterations):        
        
        # Loop episodes
        for episode in range(iterations):
            self.E = np.zeros((self.env.dealer_space,
                               self.env.player_space, self.env.nA))

            # get initial state for current episode
            s = self.env._reset()
            a = self.fixed_epsilon_greedy_action(s)
            a_next = a
            term = False
            
            # Execute until game ends
            while not term:
                # update visits
                player_idx, dealer_idx = self.get_state_number(s)
                self.N[dealer_idx, player_idx, a] += 1
                
                # execute action
                s_next, r, term = self.env._step(a)[0:3]
                player_idx_next, dealer_idx_next = self.get_state_number(s_next)
                q = self.Q[dealer_idx, player_idx, a]
                                
                if not term:
                    # choose next action with epsilon greedy policy
                    a_next = self.fixed_epsilon_greedy_action(s_next)
                    next_q = self.Q[dealer_idx_next, player_idx_next, a_next]
                    delta = r + next_q - q
                else:
                    delta = r - q
                
#                 alpha = 1.0  / (self.N[s.dealer_idx(), s.player_idx(),  Actions.as_int(a)])
#                 update = alpha * delta
#                 self.Q[s.dealer_idx(), s.player_idx(),  Actions.as_int(a)] += update
                
                self.E[dealer_idx, player_idx, a] += 1
                alpha = 1.0  / (self.N[dealer_idx, player_idx, a])
                update = alpha * delta * self.E
                self.Q += update
                self.E *= self.mlambda

                # reassign s and a
                s = s_next
                a = a_next

            #if episode%10000==0: print "Episode: %d, Reward: %d" %(episode, my_state.rew)
            self.count_wins = self.count_wins+1 if r==1 else self.count_wins

        self.iterations += iterations
#       print float(self.count_wins)/self.iterations*100

        # Derive value function
        for d in range(self.env.dealer_space):
            for p in range(self.env.player_space):
                self.V[d,p] = max(self.Q[d, p, :])
                
    def plot_frame(self, ax):
        def get_stat_val(x, y):
            return self.V[x, y]

        X = np.arange(0, self.env.dealer_space, 1)
        Y = np.arange(0, self.env.player_space, 1)
        X, Y = np.meshgrid(X, Y)
        Z = get_stat_val(X, Y)
        #surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        return surf
    
#%%

N0 = 100
agent = Sarsa_Agent(env, N0, 0.9)

for i in range (100):
    agent.train(1000)

agent.V

##%%
#
#N0 = 100
#lambdas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
#agent_list = []
#sme_list = []
##n_elements = mc_agent.Q.shape[0]*mc_agent.Q.shape[1]*2
#for l in lambdas:
#    agent = Sarsa_Agent(Environment(), N0, l)
#    agent_list.append(l)
#
#    agent.train(1000)
#    #sme = np.sum(np.square(agent.Q-mc_agent.Q))/float(n_elements)
#    #sme_list.append(sme)

#%%

def animate(frame):
    i = agent.iterations
    step_size = i
    step_size = max(1, step_size)
    step_size = min(step_size, 2 ** 16)
    agent.train(step_size)

    ax.clear()
    surf =  agent.plot_frame(ax)
    plt.title('MC score:%s frame:%s step_size:%s ' % (float(agent.count_wins)/agent.iterations*100, frame, step_size) )
    # plt.draw()
    fig.canvas.draw()
    print("done ", frame, step_size, i)
    return surf

#%%
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
%matplotlib inline 

N0 = 100
mlambda = 0.2
agent = Sarsa_Agent(env, N0, mlambda)
fig = plt.figure("N100")
ax = fig.add_subplot(111, projection='3d')

# ani = animation.FuncAnimation(fig, animate, 32, repeat=False)
ani = animation.FuncAnimation(fig, animate, 10, repeat=False)

# note: requires gif writer; swap with plt.show()
ani.save('Sarsa_Agent_py.gif', writer='imagemagick', fps=3)
# plt.show()