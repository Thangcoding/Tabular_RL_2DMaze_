import numpy as np 
import random 
from Env.maze_space import Maze_Space


class Monte_Carlo:
    ''' This code implement online policy (first visit or every visit) monte carlo method 
    if the monte carlo method want to learn off policy, it must to use importance sampling '''
    def __init__(self, gamma = 0.9 , lr = 0.08 , epsilon = 0.2):

        # setup config 
        self.gamma = gamma
        self.lr = lr 
        self.epsilon = epsilon 
    
        # decode action
        self.decode_action = {'LEFT': (0, -1), 'RIGHT': (0, 1), 'UP': (-1,0), 'DOWN': (1, 0)}

        # initial transition
        self.transition = {}
        self.history_episode = []

    def policy(self,state):
        # Epsilon-Greedy policy 
        if random.random() < 1 - self.epsilon:
            action = sorted(self.transition[state].keys(), key=lambda action: self.transition[state][action][0])[-1]
            return action
        else:
            #  exploration valid action 
            valid_action = [action for action in self.transition[state].keys() if self.transition[state][action][2] not in ['WALL','OUT']]
            return random.choice(valid_action)
        
    def run(self,state):
        action = sorted(self.transition[state].keys(), key=lambda action: self.transition[state][action][0])[-1]
        return action
    
    def reset(self):
        self.transition = {}

    def set_config(self, gamma):
        self.gamma = gamma
                            
    def learn(self, state,env, action = None ):
        ''' transition of Agent follow the format : 
                
                            state : {'action': (reward, numbers_visited , get)} 
        ''' 


        if state not in self.transition:
            self.transition[state] = {a : [0,0,None] for a in ['LEFT','RIGHT','UP','DOWN']}
        
        # choose action from policy 
        if not action:
            action = self.policy(state)

        # decode action
        action_decode = self.decode_action[action]

        # repones from environment 
        next_state, reward , done  = env.step(state,action_decode)
        
        # store transition
        self.history_episode.append((state, action, reward, next_state, done)) 

        if done == 'PATH':
            return next_state, reward, done, {}

        #  ---------- terminal state : OUT, WALL, GOAL  |  END EPISDODE AND UPDATE ----------------- 

        # update Monte Carlo 
        G = 0
        for transition in reversed(self.history_episode):
            state, action, reward, next_state , done = transition
            # update state 
            if (next_state not in self.transition) and (done == 'PATH'):
                self.transition[next_state] = {a : [0,0,None] for a in ['LEFT','RIGHT','UP','DOWN']}

            # compute return values and update every visit
            if done != 'PATH':
                G = reward 
            else:
                G = reward + self.gamma*G

            # update  value 
            old_v = self.transition[state][action][0]
            self.transition[state][action][0] += self.lr* (G - old_v)
            self.transition[state][action][1] += 1
            self.transition[state][action][2] = done 
        
        # reset history episode
        self.history_episode = []


        return next_state,reward,done, {} # the last transition
         
    def solve(self, state):
        action = self.run(state)
        action_decode = self.decode_action[action]
        next_state = (state[0] + action_decode[0], state[1] + action_decode[1])

        return next_state 
    
if __name__ =='__main__':
    pass 
