import numpy as np 
import math 
import random 
from Env.maze_space import Maze_Space 
from Random_Maze.maze_generator import Maze_Gen

''' MCTS (Monte Carlo  Tree Search)

In general, MCTS is not an reinforcement learning algorithm.
Exactly, it is a planning algorithm which based on the bandit theory to decision over multi state.
 
'''

class StateNode:
    def __init__(self, state, parent = None , done = None):
        ''' action = { a: [ucb-val, total reward , number visited, next_state : (StateNode)]}
        c : exploration constant '''

        self.state = state 
        self.done = done 
        self.parent = parent # previous action 
        self.depth = 0 if self.parent == None else self.parent.depth + 1  
        self.number_visited = 0 
        self.V_value = 0
        self.actions = {} # list action 

    def expand(self): 
        # expand action 
        self.actions = {a: ActionNode(a) for a in ['LEFT','RIGHT','UP','DOWN']} 
    
    def backup(self,rollout_value = None):
        '''            V(s) = max_{a \in A} Q(s,a)           '''
        if rollout_value is not None:
            self.V_value = rollout_value 
        else:
            max_a = sorted(self.actions.keys(), key = lambda a: self.actions[a].Q_value)[-1]
            self.V_value = self.actions[max_a].Q_value
        
        return self.parent, self.V_value 

class ActionNode: 
    def __init__(self,action,next_state = None, parent = None, c = 0.14, gamma = 0.9): 
        self.action = action 
        self.parent = parent # current state 
        self.c = c 
        self.gamma = gamma 
        self.Q_value = 0 
        self.total_reward = 0 
        self.number_visited = 0 
        self.next_state = next_state 

    def backup(self,V_value):
        ''' ucb (uncertainty confidence bound) policy for multi-arm bandit

                UCB = mu + C * sqrt(log(N)/ N_a) 
                Q(s_{t},a) = UCB + gamma*V(s_{t+1})
                
        ''' 
        
        ucb_val = (self.total_reward/ self.number_visited) + self.c * math.sqrt(math.log(self.parent.number_visited)/self.number_visited)

        self.Q_value = ucb_val + self.gamma * V_value 

        return self.parent 
    
class MCTS_Learning:
    def __init__(self, gamma = 0.9, lr = 0.08, c = 0.14):
        self.gamma = gamma 
        self.lr = lr 
        self.c = c 
        self.root = None 
        self.curr_node = None 

        # rollout setup 
        self.reward_rollout = 0
        self.depth_rollout = None 
        self.state_rollout = None 

        self.path_solution = {} # optimal path 

        self.decode_action = {'LEFT': (0,-1), 'RIGHT': (0,1), 'UP': (-1,0), 'DOWN': (1,0)}

    def selection(self, state_node, env):
        # update number visited 
        state_node.number_visited += 1

        # choose action 
        action  = sorted(state_node.actions.keys(), key = lambda a : state_node.actions[a].Q_value)[-1]
        next_state, reward, done = env.step(state_node.state , self.decode_action[action])
        
        if state_node.actions[action].next_state == None:
            # next node is None 
            next_state_node = StateNode(next_state,state_node.actions[action],done)
            state_node.actions[action].next_state = next_state_node 
            self.curr_node = next_state_node

        else:
            next_state_node = state_node.action[action].next_state
            self.curr_state_node = next_state_node

        # update 
        state_node.actions[action].total_reward += reward
        state_node.actions[action].number_visited += 1
        state_node.actions[action].next_state = next_state_node

        return next_state_node , reward, done 
    
    def simulation(self, state, env):
        ''' random rollout ''' 
        action = random.choice([a for a in ['LEFT','RIGHT','UP','DOWN']])

        next_state, reward, done = env.step(state,self.decode_action[action])

        return next_state, reward, done 

    def backup(self, state_node, rollout_value):
        ''' backup bellman equation '''

        action_node, V_value = state_node.backup(rollout_value)
        state_node = action_node.backup(V_value)

        while state_node.parent != None:
            action_node, V_value = state_node.backup()
            state_node = action_node.backup(V_value)
    
    def reset(self):  
        self.root = None   
    
    def set_config(self,gamma):
        self.gamma = gamma 
    
    def learn(self, state, env, action = None):
        
        if self.root == None:
            self.root = StateNode(state)
            self.root.expand()
            self.root.number_visited = 1
            self.curr_node = self.root 
        
        #-------------- planning -------------------
        
        # selection step 
        if self.curr_node.number_visited != 0: 
            self.curr_node, reward, done = self.selection(self.curr_node, env) 

            return self.curr_node.state, reward, done, {} 
        
        # simulation step 
        if self.depth_rollout == None:
            self.depth_rollout = self.curr_node.depth + 1 
            self.state_rollout = self.curr_node.state 

        if self.curr_node.done == 'PATH':
            next_state, reward, done = self.simulation(self.depth_rollout,env)
            
            # update rollout 
            self.rollout_reward += self.gamma**(self.depth_rollout)*reward 
            self.depth_rollout += 1 
            self.state_rollout = next_state 

            return next_state, reward, done, {} 
        
        # backup step 

        self.backup(self.curr_node, self.reward_rollout) 
                                                                              
        # reset                    
        self.curr_node = self.root 
        self.reward_rollout = 0    
        self.depth_rollout = None  
        self.state_rollout = None  

        return state, reward, done, {}

    def solve(self, state):
        
        if not self.path_solution:
            # find solution
            curr = self.root # start state  
            action_node = None 
            while curr.actions != None:
                
                # select best action
                action_node = sorted(curr.actions.values(), key = lambda a : a.Q_value)[-1]
                next_state_node = action_node.next_state

                self.path_solution[curr.state] = (action_node.action,next_state_node.state)

                curr = next_state_node 

        # inference 
        action, next_state = self.path_solution[state]

        return next_state 

if __name__ == '__main__':

    # setup 
    gen = Maze_Gen(cols = 5, rows= 5, maze_name= 'Maze_Prime')
    gen.gen()

    raw_space, state, goal = gen.maze, gen.start, gen.goal
    done = None 
    env = Maze_Space(raw_space, goal, state)

    agent = MCTS_Learning()

    # learning 

    while done != 'GOAL':
        env.render()
        state, reward, done, infor = agent.learn(state, env)




