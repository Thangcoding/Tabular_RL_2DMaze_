import random

class Double_Qlearning:
    '''  Double Q-learning using two table to learn and select action 
    helping for stable learning than normal Q-learning    '''
    def __init__(self, gamma, lr = 0.05, epsilon = 0.2):

        # setup config 
        self.gamma = gamma 
        self.lr = lr
        self.epsilon = epsilon
        # decode action
        self.decode_action = {'LEFT': (0, -1), 'RIGHT': (0, 1), 'UP': (-1,0), 'DOWN': (1, 0)}

        # two table Q value 
        self.transition_1 = {}
        self.transition_2 = {}
    
    def policy(self, state):
        if random.random() < 1 - self.epsilon:

            action = sorted(self.transition_1[state].keys(), key = lambda action: (self.transition_1[state][action][0] + self.transition_2[state][action][0]))[-1]
            return action
        else:

            valid_action = [
                action for action in self.transition_1[state].keys()
                if self.transition_1[state][action][2] not in ['WALL','OUT']
                or self.transition_2[state][action][2] not in ['WALL','OUT']
                ]

            return random.choice(valid_action)

    def set_config(self,gamma):
        self.gamma = gamma

    def run(self, state):
        action = sorted(self.transition_1[state].keys(), key = lambda action: (self.transition_1[state][action][0] + self.transition_2[state][action][0]))[-1]
        return action

    def reset(self):
        self.transition = {}

    def learn(self, state, env, action=None):
        
        # Init tables
        if state not in self.transition_1:
            self.transition_1[state] = {a:[0,0,None] for a in ['LEFT','RIGHT','UP','DOWN']}

        if state not in self.transition_2:
            self.transition_2[state] = {a:[0,0,None] for a in ['LEFT','RIGHT','UP','DOWN']}

        # choose action
        if action is None:
            action = self.policy(state)

        next_state, reward, done = env.step(state, self.decode_action[action])

        # Handle terminal state
        if done != 'PATH':
            # update the selected table
            if random.random() < 0.5:
                Q = self.transition_1[state][action][0]
                self.transition_1[state][action][0] += self.lr * (reward - Q)
                self.transition_1[state][action][1] += 1
                self.transition_1[state][action][2] = done
            else:
                Q = self.transition_2[state][action][0]
                self.transition_2[state][action][0] += self.lr * (reward - Q)
                self.transition_2[state][action][1] += 1
                self.transition_2[state][action][2] = done       

            return next_state, reward, done, {}

        # Ensure next_state exists
        if next_state not in self.transition_1:
            self.transition_1[next_state] = {a:[0,0,None] for a in ['LEFT','RIGHT','UP','DOWN']}
        if next_state not in self.transition_2:
            self.transition_2[next_state] = {a:[0,0,None] for a in ['LEFT','RIGHT','UP','DOWN']}


        # Double Q-learning Update
        if random.random() < 0.5:
            # Update Q1
            # target action is from Q1
            a_star = max(self.transition_1[next_state], key=lambda a: self.transition_1[next_state][a][0])
            target = reward + self.gamma * self.transition_2[next_state][a_star][0]

            Q = self.transition_1[state][action][0]
            self.transition_1[state][action][0] += self.lr * (target - Q)
            self.transition_1[state][action][1] += 1
            self.transition_1[state][action][2] = done

        else:
            # Update Q2
            a_star = max(self.transition_2[next_state], key=lambda a: self.transition_2[next_state][a][0])
            target = reward + self.gamma * self.transition_1[next_state][a_star][0]

            Q = self.transition_2[state][action][0]
            self.transition_2[state][action][0] += self.lr * (target - Q)
            self.transition_2[state][action][1] += 1
            self.transition_2[state][action][2] = done

        return next_state, reward, done, {}

    
    def solve(self,state):
        action = self.run(state)
        action_decode = self.decode_action[action]
        next_state = (state[0] + action_decode[0], state[1] + action_decode[1])

        return next_state

if __name__ == '__main__':
    pass  
