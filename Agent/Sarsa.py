import numpy as np 
import random 
from Env.maze_space import Maze_Space

''' SARSA is also a temporal different learning similaring with Q-learning, 
However SARSA is a kind of online policy stead of offline policy in Q-learning. 

Explaining about offline and online learning: 

--------  '''

class SARSA:
    def __init__(self, gamma, lr = 0.05, epsilon = 0.2 ):

        # setup config 
        self.gamma = gamma
        self.lr = 0.1
        self.epsilon = epsilon
        # decode action
        self.decode_action = {'LEFT': (0, -1), 'RIGHT': (0, 1), 'UP': (-1,0), 'DOWN': (1, 0)}

        # initial transition
        self.transition = {}

    def policy(self,state):
        # Epsilon-Greedy policy 
        if random.random() < 1 - self.epsilon:
            action = sorted(self.transition[state].keys(), key=lambda action: self.transition[state][action][0])[-1]
            return action
        else:
            # no exploration action invalid 
            valid_action = [action for action in self.transition[state].keys() if self.transition[state][action][2] not in ['WALL', 'OUT']]
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

        if state not in self.transition:
            self.transition[state] = {a:[0,0,None] for a in ['LEFT','RIGHT','UP','DOWN']}

        # choose action from policy if it has no action from previous step
        if not action:
            action = self.policy(state)

        next_state, reward, done = env.step(state, self.decode_action[action])

        #  terminal state : GOAL, WALL, OUT
        if done != 'PATH':     
            self.transition[state][action][0] += self.lr*(reward - self.transition[state][action][0])
            self.transition[state][action][1] += 1
            self.transition[state][action][2] = done 
            return next_state, reward, done, {}

        # update next_state
        if next_state not in self.transition:
            self.transition[next_state] = {a:[0,0,None] for a in ['LEFT','RIGHT','UP','DOWN']}

        # SARSA-learning update
        next_action = self.policy(next_state)
        self.transition[state][action][0] += self.lr * (
            (reward + self.gamma *self.transition[next_state][next_action][0]) - self.transition[state][action][0]
        )
        self.transition[state][action][1] += 1
        self.transition[state][action][2] = done 

        return next_state, reward, done, {'next_action': next_action}

    def solve(self, state):
        action = self.run(state)
        action_decode = self.decode_action[action]
        next_state = (state[0] + action_decode[0], state[1] + action_decode[1])

        return next_state 
if __name__ =='__main__':
    pass 
