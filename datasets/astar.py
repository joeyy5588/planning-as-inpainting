from queue import PriorityQueue

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.f == other.f
    
    def __lt__(self, other):
        return self.f < other.f
    
    def __le__(self, other):
        return self.f <= other.f
    
    def __gt__(self, other):
        return self.f > other.f
    
    def __ge__(self, other):
        return self.f >= other.f

    def __str__(self):
        return f"Node {self.position, self.parent}"

    


def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""
    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0

    # Initialize both open and closed list
    frontier = PriorityQueue()
    frontier.put((0, start_node))

    distance_to_goal = PriorityQueue()
    current_distance_to_goal = ((start_node.position[0] - end[0]) ** 2) + ((start_node.position[1] - end[1]) ** 2)
    distance_to_goal.put((current_distance_to_goal, start_node))

    closed_list = set()
    cost_so_far = {}
    cost_so_far[start] = 0

    # Loop until you find the end
    while not frontier.empty():

        # Get the current node
        _, current_node = frontier.get()

        # if current_node.position == end:
        #     path = []
        #     current = current_node
        #     while current is not None:
        #         path.append(current.position)
        #         current = current.parent
        #     return path[::-1] # Return reversed path

        closed_list.add(current_node.position)        

        available_next_step = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        for new_position in available_next_step: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # if node_position in closed_list:
            #     continue

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)
            new_node.g = current_node.g + 1
            new_node.h = ((new_node.position[0] - end[0]) ** 2) + ((new_node.position[1] - end[1]) ** 2)
            new_node.f = new_node.g + new_node.h

            if node_position not in cost_so_far or new_node.f < cost_so_far[node_position]:
                cost_so_far[node_position] = new_node.f
                frontier.put((new_node.f, new_node))
                distance_to_goal.put((new_node.h, new_node))

        # print("distance", cost_so_far)
        # print("Closed", closed_list)

    _, closest_node = distance_to_goal.get()
    path = []
    current = closest_node
    while current is not None:
        path.append(current.position)
        current = current.parent

    return path[::-1] # Return reversed path
