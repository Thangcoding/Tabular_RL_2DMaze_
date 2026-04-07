import numpy as np 
import random 
from Env.maze_space import Maze_Space



class N_Step:
    def __init__(self, gamma = 0.9,lr = 0.1, epsilon = 0.2, num_steps = 3):

        # setup config 
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.lr = lr

        # decode action
        self.decode_action = {'LEFT': (0, -1), 'RIGHT': (0, 1), 'UP': (-1,0), 'DOWN': (1, 0)}

        # initial transition
        self.transition = {}
        self.history_steps = []
        self.steps = 0 

    def policy(self,state):
        # Epsilon-Greedy policy 
        if random.random() < 1 - self.epsilon:
            action = sorted(self.transition[state].keys(), key=lambda action: self.transition[state][action][0])[-1]
            return action
        else:
            # no exploration action valid 
            valid_action = [action for action in self.transition[state].keys() if self.transition[state][action][2] not in ['WALL','OUT']]
            return random.choice(valid_action)
        
    def run(self,state):
        action = sorted(self.transition[state].keys(), key=lambda action: self.transition[state][action][0])[-1]
        return action
    
    def reset(self):
        self.transition = {}

    def set_config(self, gamma):
        self.gamma = gamma
                            
    def learn(self, state,env, action = None):
        ''' transition of Agent follow the format : 
                        state : {'action': (reward, numbers_visited , get)} 
        '''
        # update state 
        if state not in self.transition:
            self.transition.update({state: {'LEFT': [0, 0 , None] , 'RIGHT': [0, 0, None] , 'UP': [0,0, None] , 'DOWN': [0, 0 , None]}})
        
        # choose action follow policy
        if not action:
            action = self.policy(state)
        
        action_decode = self.decode_action[action]
        # repones from environment 
        next_state, reward , done  = env.step(state,action_decode)
        
        # update transition 
        self.history_steps.append((state, action,reward, next_state, done))
        self.steps += 1

        if (done == 'PATH') and (self.steps != self.num_steps):
            return next_state,reward, done, {}

        # update value 
        G = 0
        next_action = None 
        for i, transition in enumerate(reversed(self.history_steps)):
            state, action, reward, next_state, done = transition

            # update state 
            if (next_state not in self.transition) and (done == 'PATH'):
                self.transition[next_state] = {a : [0,0,None] for a in ['LEFT', 'RIGHT', 'UP','DOWN']} 
            
            if i == 0:
                if done != 'PATH': 
                    # terminal state 
                    G = reward 
                else:
                    next_action = self.policy(next_state)
                    G = reward + self.transition[next_state][next_action][0]
            else:
                G = reward + self.gamma*G

            old_v = self.transition[state][action][0]
            self.transition[state][action][0] += self.lr*(G - old_v) 
        
        # reset history and step count
        self.history_steps = []
        self.steps = 0 
            
        return next_state,reward,done, {'next_action': next_action}
    
    def solve(self, state):
        action = self.run(state)
        action_decode = self.decode_action[action]
        next_state = (state[0] + action_decode[0], state[1] + action_decode[1])

        return next_state 
    
if __name__ =='__main__':
    pass 
