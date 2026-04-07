import tkinter as tk                                      
from tkinter import ttk                                   
import time                                             
import threading                                           
import matplotlib.pyplot as plt                                 
from tkinter import messagebox                          
from Random_Maze.maze_generator import Maze_Gen         
from Agent.Qlearning import Q_Learning                    
from Agent.DoubleQlearning import Double_Qlearning   
from Agent.MonteCarlo import Monte_Carlo 
from Agent.Sarsa import SARSA  
from Agent.Nstep import N_Step
from Env.maze_space import Maze_Space
import queue 

''' Main program implement a visualize of the process learning of Agent Q-learning.
    Using tkinter tool GUI to excute it.
    tool tkinter includes:
        Canvas : draw graphic
        
'''
GRID_SIZE = 40
OBJECT_SIZE = 10
SPACE_BUTTON = 30
WIDTH = 100 
HEIGHT = 15 


class MazeGUI:

    def __init__(self, master):
        self.master = master
        self.master.title("Maze Solver")

        # setup maze   
        self.rows = 5
        self.cols = 5
        self.maze_name = 'Maze_Prime'
        self.maze = None 
        self.maze_gen = Maze_Gen(cols= self.cols , rows= self.rows, maze_name= self.maze_name)

        # config for Agent
        self.gamma = 0.9
        self.episodes = 10
        self.location = None
        self.goal = None 
        self.average_reward = {'Q_Learning': [], 
                               'Double_Qlearning': [], 
                               'Sarsa': [], 
                               'Monte_Carlo': [], 
                               'N_Step': []
                               }
        self.agent_name = 'Q_Learning'
        self.set_agent = {'Q_Learning':Q_Learning(self.gamma),
                          'Double_Qlearning': Double_Qlearning(self.gamma), 
                          'Sarsa': SARSA(self.gamma), 
                          'Monte_Carlo': Monte_Carlo(self.gamma), 
                          'N_Step': N_Step(self.gamma)}
        self.agent = self.set_agent[self.agent_name]
        self.env = Maze_Space(self.maze, self.goal)
        self.stop_flag = threading.Event()
        self.speed = 0

        # determine the terminate state of agent in learning 
        self.done_check = None 

        # panel maze 
        self.canvas = tk.Canvas(master, width= self.rows*GRID_SIZE, height= self.cols*GRID_SIZE, bg="white",bd = 5, relief = 'ridge' )
        self.canvas.place(x = 400 , y = 10)
        
        # control botton
        self.create_control_buttons(master)

        # draw maze 
        self.draw_maze()
    
    def create_control_buttons(self, master):
        '''   Control button  '''
    
        # button nums Columns
        self.col_label = tk.Label(master, text="Number Columns")
        self.col_label.place(x=20, y=15, width = WIDTH , height = HEIGHT)
        self.col_input = tk.Spinbox(master, from_=3, to=12, increment=1, state= 'readonly')
        self.col_input.place(x= 120, y=15, width = WIDTH, height = HEIGHT)
        self.col_input.delete(0, tk.END) 
        self.col_input.insert(0, str(self.cols))
    
        # button num Rows
        self.row_label = tk.Label(master, text="Number Rows")
        self.row_label.place(x = 20 , y = 15 + SPACE_BUTTON, width= WIDTH, height= HEIGHT)
        self.row_input = tk.Spinbox(master, from_=3, to=8, increment=1, state= 'readonly')
        self.row_input.place(x = 120, y = 15 + SPACE_BUTTON, width= WIDTH, height = HEIGHT)
        self.row_input.delete(0, tk.END) 
        self.row_input.insert(0, str(self.rows))
    
        # button speed 
        self.speed_scale = tk.Scale(master, from_=0, to=1,resolution= 0.01, orient=tk.HORIZONTAL,length= 100) 
        self.speed_scale.place(x = 110, y =  10 + SPACE_BUTTON*2) 
        self.speed_button = tk.Button(master,text= 'Speed', command= self.update_speed) 
        self.speed_button.place(x = 10 , y = 30 + SPACE_BUTTON*2 , width= WIDTH , height= HEIGHT) 
    
        # button maze agorithm
        self.algorithm = ttk.Combobox(master,width= 27,textvariable= tk.StringVar(), state = 'readonly')
        self.algorithm['values'] = ('Maze_DFS', 'Maze_Prime', 'Maze_Prime_Cycle')
        self.algorithm.place(x = 110 , y = 30 + SPACE_BUTTON*3)
        self.algorithm_button = tk.Button(master, text= 'Maze Name', command= self.update_maze_name)
        self.algorithm_button.place(x =10 , y = 30 + SPACE_BUTTON*3, width= WIDTH, height= HEIGHT)
    
        # button Agent Algorithm 
        self.agent_algorithm = ttk.Combobox(master,width=27,textvariable= tk.StringVar(), state= 'readonly')
        self.agent_algorithm['values'] = [k for k in self.set_agent.keys()]
        self.agent_algorithm.place(x=110,y = 30 + SPACE_BUTTON*4)
        self.agent_algorithm_button = tk.Button(master,text= 'Agent Name', command=self.update_agent)
        self.agent_algorithm_button.place(x = 10, y = 30 + SPACE_BUTTON*4, width= WIDTH, height= HEIGHT)

        # button number process training 
        self.num_process = tk.Entry(master)  
        self.num_process.place(x = 150, y = 170 + SPACE_BUTTON*3, width= WIDTH , height= HEIGHT)
        self.num_button = tk.Button(master, text= 'Num_Process', command= self.set_episodes)
        self.num_button.place(x = 50, y = 170 + SPACE_BUTTON*3 , width= WIDTH, height= HEIGHT)
        
        # button config gamma 
        self.gamma_input = tk.Entry(master) 
        self.gamma_input.place(x = 150 , y = 190 + SPACE_BUTTON*3, width= WIDTH , height= HEIGHT)
        self.gamma_button = tk.Button(master, text="Gamma", command=self.config)
        self.gamma_button.place(x=50, y= 190 + SPACE_BUTTON * 3, width=WIDTH, height=HEIGHT)
        self.gamma_input.setvar(value=0.4)

        # button reset location
        self.set_start_button = tk.Button(master, text="Agent Location", command=self.set_location)
        self.set_start_button.place(x = 50, y = 190 + SPACE_BUTTON*4 , width= WIDTH , height= HEIGHT)
        
        # button reset goal
        self.set_end_button = tk.Button(master, text="Goal", command=self.set_goal)
        self.set_end_button.place(x = 50, y =  190 + SPACE_BUTTON*5, width= WIDTH, height= HEIGHT)
        
        # button reset Agent
        self.reset_button = tk.Button(master, text="Reset Agent", command=self.reset_agent)
        self.reset_button.place(x = 50 , y = 190 + SPACE_BUTTON*6 , width= WIDTH , height= HEIGHT)

        # button train Agent
        self.train_button = tk.Button(master, text="Train Agent", command=self.train_agent)
        self.train_button.place(x = 50 , y = 190 + SPACE_BUTTON*7 , width= WIDTH , height= HEIGHT)

        # button run
        self.run_button = tk.Button(master, text="Run Agent", command=self.run_agent)
        self.run_button.place(x = 50 , y = 190 + SPACE_BUTTON*8 , width= WIDTH, height= HEIGHT)

        # button stop 
        self.stop_button = tk.Button(master,text= 'Stop', command= self.stop)
        self.stop_button.place(x = 50 , y = 190 + SPACE_BUTTON*9 , width= WIDTH , height= HEIGHT)
        # button draw maze
        self.draw_button = tk.Button(master, text="Generate  Maze", command=self.draw_maze)
        self.draw_button.place( x = 50, y = 190 + SPACE_BUTTON*10, width= WIDTH + 20 , height= HEIGHT + 20)

        # button plot average reward 
        self.reward_plot_button = tk.Button(master, text="Average Reward", command=self.draw_reward)
        self.reward_plot_button.place( x = 50, y = 190 + SPACE_BUTTON*11, width= WIDTH + 20 , height= HEIGHT + 20)
    
    def config(self):
        '''  setup config of agent '''
        self.gamma = float(self.gamma_input.get())
        self.agent.set_config(self.gamma)

    def set_episodes(self):
        self.episodes = int(self.num_process.get())
        
    def stop(self):
        ''' stop the process of learning or running  '''  
        self.stop_flag.set()                              

    def update_speed(self):
        ''' Update the speed of learning  '''
        self.speed = 1 - self.speed_scale.get()

    def update_maze_name(self):
        ''' update the maze name algorithm  '''
        self.maze_name = self.algorithm.get()

    def update_agent(self):
        ''' Select algorithm reinforcement learning '''
        self.agent_name = self.agent_algorithm.get()
        self.agent = self.set_agent[self.agent_name]

    def maze_size(self):
        '''  update size of maze gen  '''
        self.cols = int(self.col_input.get())
        self.rows = int(self.row_input.get())

        # setup maze gen again
        self.maze_gen = Maze_Gen(cols= self.cols , rows= self.rows, maze_name = self.maze_name)

    def draw_maze(self):
        self.canvas.delete("all")
        self.gamma = self.gamma_input.get()

        # generate maze 
        self.maze_size()
        self.maze_gen.gen()
        self.maze = self.maze_gen.maze

        # reset the size of pannel 
        self.canvas.config(width= GRID_SIZE*(self.cols*2 - 1), height = GRID_SIZE*(self.rows*2 - 1))
        self.canvas.place(y = 310 - (GRID_SIZE*(self.rows*2 -1))/2 , x = 800 - (GRID_SIZE*(self.cols*2 -1))/2)

        # set location and goal
        self.location = self.maze_gen.start
        self.goal = self.maze_gen.goal

        # setup environment
        self.set_location(self.location)
        self.set_goal(self.goal)

        self.env = Maze_Space(self.maze , self.goal)

        for i in range(self.rows*2 - 1):
            for j in range(self.cols*2 -1):

                if self.maze[i][j] == 0 or self.maze[i][j] == 1:  
                    continue 
                else:
                    self.canvas.create_rectangle(j*GRID_SIZE , i*GRID_SIZE , (j+1)*GRID_SIZE , (i+1)*GRID_SIZE , fill = 'blue')

    def set_location(self, position = None):
        if position != None:
            # set location in random maze
            y , x = position
            self.location_id = self.canvas.create_oval(x*GRID_SIZE + 15 , y*GRID_SIZE + 15,  x*GRID_SIZE + 15 + OBJECT_SIZE, y*GRID_SIZE + 15 + OBJECT_SIZE , fill="red")
        else:
            # set location in deterministic maze
            self.location = self.maze_gen.reset_start(self.goal)
            y , x = self.location
            self.canvas.delete(self.location_id)
            self.location_id = self.canvas.create_oval(x*GRID_SIZE + 15 , y*GRID_SIZE + 15,  x*GRID_SIZE + 15 + OBJECT_SIZE, y*GRID_SIZE + 15 + OBJECT_SIZE , fill="red")

    def set_goal(self, position = None):
        # set goal 
        if position != None:
            y, x = position
            self.goal_id = self.canvas.create_oval(x*GRID_SIZE + 15, y*GRID_SIZE + 15, x*GRID_SIZE + 15 + OBJECT_SIZE, y*GRID_SIZE + 15 + OBJECT_SIZE, fill="green")
        else:
            # set random in deterministic maze
            self.goal = self.maze_gen.reset_goal(self.location)
            y , x = self.goal 
            self.canvas.delete(self.goal_id)
            self.goal_id = self.canvas.create_oval(x*GRID_SIZE + 15, y*GRID_SIZE + 15, x*GRID_SIZE + 15 + OBJECT_SIZE, y*GRID_SIZE + 15 + OBJECT_SIZE, fill="green")
    
    def update_states(self,position):
        y, x = position 
        if self.location_id is not None:
            self.canvas.delete(self.location_id) 
        self.location_id = self.canvas.create_oval(x*GRID_SIZE + 15, y*GRID_SIZE + 15, x*GRID_SIZE + 15 + OBJECT_SIZE, y*GRID_SIZE + 15 + OBJECT_SIZE, fill="red")

    def reset_agent(self):
        ''' Reset anything agent learned '''
        self.agent.reset()
        self.average_reward[self.agent_name] = []
    
    def train_loop(self):
        episode = 0

        self.stop_flag.clear()
        while not self.stop_flag.is_set() and episode < self.episodes:
            state, action = self.location, None 
            x , y = state 
            self.canvas.after(0, self.update_states, (x,y))
            self.canvas.update() 
            avg_reward = 0 
            step = 0

            while not self.stop_flag.is_set():
                next_state, reward, done , infor = self.agent.learn(state, self.env,action = action)
                avg_reward += reward

                if infor:
                    action = infor['next_action']
                 
                if done != 'PATH': 
                    state = self.location 
                    x , y = state 
                    self.canvas.after(0, self.update_states, (x,y))
                    self.canvas.update()   
                    step += 1
                    break                                        
                                                                     
                x, y = next_state                                     
                self.canvas.after(0, self.update_states, (x,y))          
                time.sleep(self.speed)
                self.canvas.update()    

                # continous 
                state = next_state
            
            x , y = self.location
            self.canvas.after(0, self.update_states, (x,y))
            self.canvas.update()  
            self.average_reward[self.agent_name].append(avg_reward/(step*100))  
            episode += 1                                     
        threading.Event().wait(1) 

    def run_loop(self):
        state = self.location
        self.stop_flag.clear()
        self.canvas.after(0 , self.update_states, state)
        while not self.stop_flag.is_set():
            next_state = self.agent.solve(state)
            x , y = next_state
            self.canvas.after(0, self.update_states, (x,y)) 
            time.sleep(self.speed)
            self.canvas.update()  
            if next_state == self.goal:
                break 
            state = next_state
        threading.Event().wait(1) 

    def train_agent(self):
        ''' Training agent with by interact with environment '''

        training_thread = threading.Thread(target=self.train_loop)
        training_thread.daemon = True 
        training_thread.start() 

    def run_agent(self):
        ''' Runing Agent based on the policy learned from algorithm'''
 
        running_thread = threading.Thread(target= self.run_loop)           
        running_thread.daemon = True                             
        running_thread.start() 

    def draw_reward(self):
        plt.figure(figsize= (8,6))
        plt.xlabel("Process")
        plt.ylabel("Reward")
        plt.title("Average Reward")
        plt.plot([i for i in range(1,len(self.average_reward['Q_Learning'])+1)],self.average_reward['Q_Learning'], label = 'Q_Learning', color = 'blue')
        plt.plot([i for i in range(1,len(self.average_reward['Double_Qlearning'])+1)],self.average_reward['Double_Qlearning'], label= 'Double_Qlearning', color = 'red')
        plt.plot([i for i in range(1,len(self.average_reward['Sarsa'])+1)],self.average_reward['Sarsa'], label= 'Sarsa', color = 'yellow')
        plt.plot([i for i in range(1,len(self.average_reward['Monte_Carlo']) + 1)], self.average_reward['Monte_Carlo'], label = 'Monte_Carlo', color = 'green')
        plt.plot([i for i in range(1,len(self.average_reward['N_Step']) + 1)], self.average_reward['N_Step'], label = 'N_Step', color = 'gray')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    # main window
    root = tk.Tk()
    app = MazeGUI(root)
    root.mainloop()


