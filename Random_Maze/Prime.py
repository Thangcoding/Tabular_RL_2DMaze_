import random 


class Maze_Prime:
    def __init__(self, rows ,cols):
        self.rows = rows 
        self.cols = cols 

        self.graph = {}
        self.maze = [[-1 for _ in range(self.cols*2 - 1)] for _ in range(self.rows*2 -1)]
        self.lst_node = [(i , j) for j in range(self.cols) for i in range(self.rows)]
        self.start = None 
        self.goal = None 
        
    def set_up(self):
 
        for i in range(self.rows):
            for j in range(self.cols): 
                node = (i , j)  
                adjacent = [(i + 1, j), (i - 1, j), (i , j + 1), (i , j - 1)] 
                valid_adjacent = [a for a in adjacent if 0 <= a[0] < self.rows and  0 <= a[1] < self.cols]
                self.graph.update({node : {adjacent_node: False for adjacent_node in valid_adjacent}}) 
    
    def is_valid(self, node_1 , node_2): 
        # check created cycle                                                            
        connected_1 = [a for a in self.graph[node_1] if self.graph[node_1][a] == True]   

        if connected_1 == []:
            # not connect before 
            return True 
        stack = connected_1
        start = stack[0]
        visited = [start]         
        while stack != []:  
            start = stack[0]
            if start == node_2: 
                return False  
            adjacents = [a for a in self.graph[start] if self.graph[start][a] == True and a not in visited]
            visited.extend(adjacents) 
            stack.pop(0) 
            stack = adjacents + stack

        return True 

    def conection_node(self):
        number_edge = 0
        while number_edge < self.cols*self.rows-1:
            node = random.choice(self.lst_node)
            lst_adjacent  = [a for a in self.graph[node] if self.graph[node][a] == False]
            if lst_adjacent == []:
                self.lst_node.remove(node)
                continue                            
            adjacent = random.choice(lst_adjacent)  

            if self.is_valid(node , adjacent): 
                # connection edge 
                self.graph[node][adjacent] = True 
                self.graph[adjacent][node] = True 
                number_edge += 1

    def create_cycle(self):

        NUMBER_CYCLE = 14
        count = 0
        while count < NUMBER_CYCLE:
            if self.lst_node == None:
                break 
            node = random.choice(self.lst_node)
            lst_adjacent = [a for a in self.graph[node] if self.graph[node][a] == False]
            if lst_adjacent == []:
                self.lst_node.remove(node)
                continue
            adjacent = random.choice(lst_adjacent)
        
            self.graph[node][adjacent] = True 
            count += 1 
        
    def generator(self, cycle = False):
        self.start , self.goal = random.sample(self.lst_node,2)
        # mapping start and goal on maze 
        self.start = (self.start[0]*2, self.start[1]*2)
        self.goal = (self.goal[0]*2 , self.goal[1]*2)
        
        # set_up edge connection
        self.set_up()
        # generating spanning tree by prime algorithm 
        self.conection_node()
        # create cycle in spanning tree 
        if cycle == True:
            self.create_cycle()

        for i in range(self.rows):
            for j in range(self.cols):
                node = (i , j) 
                # node of graph mapping to maze (i,j) ---> (i*2, j*2)
                # break location
                self.maze[node[0]*2][node[1]*2] = 0 

                for adjacent in self.graph[node]:
                    if self.graph[node][adjacent]:
                        
                        # break wall 
                            # direction path  
                        d_x, d_y = adjacent[0] - node[0],adjacent[1] - node[1]
                        
                        if (node[0]*2 + d_x < self.rows*2 - 1) and (node[1]*2 + d_y < self.cols*2 -1):
                            self.maze[node[0]*2 + d_x][node[1]*2 + d_y] = 0
                        
                        # reset again 
                        self.graph[node][adjacent] = False 
                        self.graph[adjacent][node] = False 
        
        self.maze[self.start[0]][self.start[1]] = 0 
        self.maze[self.goal[0]][self.goal[1]] = 1 

        return self.maze, self.start, self.goal


if __name__ == '__main__':
    gen = Maze_Prime(rows= 5 , cols= 5)
    
    maze, start , goal = gen.generator()

    print(start)
    print(maze)
    print(goal)



