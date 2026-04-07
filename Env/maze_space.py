import numpy as np 


class Maze_Space:

    def __init__(self, raw_space, goal, state = None ):
        # 'wall': -2, 'path': -1, 'goal': 1
        self.raw_space = raw_space 
        self.goal = goal 
        self.agent_position = state

        # render visualization 
        self.render_raw = [] 
        for i in range(len(self.raw_space)): 
            line = ['X' if self.raw_space[i][j] == -1 else '-' for j in range(len(self.raw_space[0]))]
            self.render_raw.append(line)

        self.render_raw[self.goal[0]][self.goal[1]] = 'G'

        self.render_raw[self.agent_position[0]][self.agent_position[1]] = 'A'

    def reward_function(self, done):
        if done == 'OUT':
            return -1000
        elif done == 'WALL':
            return -1000
        elif done == 'PATH':
            return -10
        else:
            return 1000
        
    def step(self,state = (int, int), action = (int, int)):
        ''' interating with environment '''
        next_state = (state[0] + action[0], state[1] + action[1])

        # draw render 
        self.render_raw[self.agent_position[0]][self.agent_position[1]] = '-' if self.raw_space[self.agent_position[0]][self.agent_position[1]] == -2 else 'X'
        self.agent_position = next_state
        self.render_raw[self.agent_position[0]][self.agent_position[1]] = 'A'

        # Agent go to outside of maze
        if not (( 0 <= next_state[0] < len(self.raw_space)) and (0 <= next_state[1] < len(self.raw_space[0]))):
            return (None,self.reward_function('OUT'), 'OUT')
        
        # Agent go to wall 
        if self.raw_space[next_state[0]][next_state[1]] == -1:
            return (next_state,self.reward_function('WALL'), 'WALL')

        # Agent follow normal path which not reach on goal 
        if self.raw_space[next_state[0]][next_state[1]] == 0:
            return (next_state ,self.reward_function('PATH'),'PATH')

        # Agent reach to goal
        return (next_state, self.reward_function('GOAL'),'GOAL')

    def reset(self):
        ''' reset old environment  '''
        pass 

    def render(self):
        ''' showing the state of agent in raw space (simple visualize) '''
        print('_____________________2D_MAZE______________________\n')
        print('\n'.join([' '.join( ['              '] + [self.render_raw[i][j] for j in range(len(self.render_raw[0]))]) for i in range(len(self.render_raw))]))
        print('__________________________________________________')

if __name__ == '__main__':
    pass 
