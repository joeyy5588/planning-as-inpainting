import numpy as np
from .astar import astar
from transformers import CLIPTextModel, AutoTokenizer
from collections import deque

object_dict = {
  -1: "obstacle",
  1: "agent",
  2: "apple",
  3: "carrot",
  4: "banana",
  5: "lettuce",
  6: "orange",
  7: "tomato",
  8: "grapes",
  9: "broccoli",
  10: "strawberry",
  11: "cucumber"
}

model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

repr_dict = np.zeros((len(object_dict)+2, 512))
for k, v in object_dict.items():
    repr_dict[k] = model(tokenizer(v, return_tensors="pt").input_ids).last_hidden_state.detach().cpu().numpy()[0, 1]


def gaussian_kernel(size, sigma=3.0):
    """Generate a 2D Gaussian kernel."""
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / g.sum()
    
def apply_kernel_at_target(grid, target_index, kernel):
    """Apply a kernel centered at a target index."""
    dx, dy = kernel.shape[0]//2, kernel.shape[1]//2
    padded_grid = np.pad(grid, ((dx, dx), (dy, dy)), mode='constant', constant_values=0)
    
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            padded_grid[target_index[0] + i, target_index[1] + j] += kernel[i, j]
    
    return padded_grid[dx:-dx, dy:-dy]

def get_direction(start, end):
    # Convert array index to coordinates
    y1, x1 = start
    y2, x2 = end
    y1 = -y1
    y2 = -y2
    # calculate the angle
    dx = x2 - x1
    dy = y2 - y1
    rads = np.arctan2(dy,dx)
    rads %= 2*np.pi
    degs = np.rad2deg(rads)
    # print(x1, y1, x2, y2, dx, dy, degs)
    # 8 directions
    if degs >= 337.5 or degs < 22.5:
        return "go east"
    elif degs >= 22.5 and degs < 67.5:
        return "go northeast"
    elif degs >= 67.5 and degs < 112.5:
        return "go north"
    elif degs >= 112.5 and degs < 157.5:
        return "go northwest"
    elif degs >= 157.5 and degs < 202.5:
        return "go west"
    elif degs >= 202.5 and degs < 247.5:
        return "go southwest"
    elif degs >= 247.5 and degs < 292.5:
        return "go south"
    elif degs >= 292.5 and degs < 337.5:
        return "go southeast"

class POMDPGenerator:
    def __init__(self, config):
        self.grid_size = config.grid_size
        self.visible_range = config.visible_range
        self.max_objects = config.max_objects
        self.min_objects = config.min_objects
        self.max_obstacles = config.max_obstacles
        self.min_obstacles = config.min_obstacles
        self.viewable_grid = np.zeros((self.grid_size, self.grid_size))
        self.reference_potential = np.zeros((self.grid_size, self.grid_size))
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.target = np.zeros((self.grid_size, self.grid_size))
        self.path = np.zeros((self.grid_size, self.grid_size))
        self.observations = []
        self.instructions = ""
        self.blur_target = config.blur_target

    def reset(self):
        self.viewable_grid = np.zeros((self.grid_size, self.grid_size))
        self.reference_potential = np.zeros((self.grid_size, self.grid_size))
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.target = np.zeros((self.grid_size, self.grid_size))
        self.path = np.zeros((self.grid_size, self.grid_size))
        self.observations = []
        self.instructions = ""

    def get_viewable_range(self, agent_position, visible_range=None):
        if visible_range is None:
            visible_range = self.visible_range
    
        x_min = max(0, agent_position[0]-visible_range)
        x_max = min(self.grid_size, agent_position[0]+visible_range+1)
        y_min = max(0, agent_position[1]-visible_range)
        y_max = min(self.grid_size, agent_position[1]+visible_range+1)
        return x_min, x_max, y_min, y_max

    def update_viewable_grid(self, agent_position):
        x_min, x_max, y_min, y_max = self.get_viewable_range(agent_position)
        self.viewable_grid[x_min:x_max, y_min:y_max] = 1

    def sample_target_position(self, agent_position):

        invalid_grid = np.ones((self.grid_size, self.grid_size))
        x1, x2, y1, y2 = self.get_viewable_range(agent_position, self.visible_range * 2)
        invalid_grid[x1:x2, y1:y2] = 0
        invalid_grid += self.viewable_grid
        valid_index = np.where(invalid_grid == 0)

        # sample target position from valid index
        target_index = np.random.randint(len(valid_index[0]))
        target_position = (valid_index[0][target_index], valid_index[1][target_index])

        # need to make sure target is not visible at the beginning
        while self.grid[target_position]!= 0 or self.viewable_grid[target_position] == 1 or \
            target_position == agent_position or target_position[0] - agent_position[0] > 12 or target_position[1] - agent_position[1] > 12:
            target_index = np.random.randint(len(valid_index[0]))
            target_position = (valid_index[0][target_index], valid_index[1][target_index])

        return target_position

    def sample_reference_position(self, agent_position):
        valid_index = np.where(self.reference_potential != 0)
        reference_index = np.random.randint(len(valid_index[0]))
        reference_position = (valid_index[0][reference_index], valid_index[1][reference_index])
        while self.grid[reference_position]!= 0 or self.reference_potential[reference_position] == 0 or reference_position == agent_position:
            reference_index = np.random.randint(len(valid_index[0]))
            reference_position = (valid_index[0][reference_index], valid_index[1][reference_index])
        return reference_position

    def sample_position(self):
        return (np.random.randint(self.grid_size), np.random.randint(self.grid_size))

    def blur_target_with_kernel(self):
        kernel = gaussian_kernel(3)
        target_position = np.where(self.target == 1)
        target_position = (target_position[0][0], target_position[1][0])
        self.target = apply_kernel_at_target(self.target, target_position, kernel)

    def get_observation_sequence(self, agent_coord):
        observation_sequence = []
        observed_grid = self.grid.copy()
        viewable_grid = self.viewable_grid.copy()
        for i in range(len(agent_coord)):
            if observed_grid[agent_coord[i]] == 0:
                observed_grid[agent_coord[i]] = 1
            current_grid = observed_grid.copy()
            x_min, x_max, y_min, y_max = self.get_viewable_range(agent_coord[i])
            viewable_grid[x_min:x_max, y_min:y_max] = 1
            current_grid = current_grid * viewable_grid
            observation_sequence.append(current_grid)

        observation_sequence = np.array(observation_sequence)
        return observation_sequence

    def get_mask_sequence(self, agent_coord):
        mask_sequence = []
        viewable_grid = np.zeros((self.grid_size, self.grid_size))
        for i in range(len(agent_coord)):
            x_min, x_max, y_min, y_max = self.get_viewable_range(agent_coord[i])
            viewable_grid[x_min:x_max, y_min:y_max] = 1
            mask_sequence.append(viewable_grid.copy())

        mask_sequence = np.array(mask_sequence)
        return mask_sequence
        
    def generate(self):
        self.reset()
        agent_coord = []
        # Place the agent randomly
        init_position = self.sample_position()
        agent_position = init_position
        # self.grid[agent_position] = 1
        self.update_viewable_grid(agent_position)
        x1, x2, y1, y2 = self.get_viewable_range(agent_position)
        self.reference_potential[x1:x2, y1:y2] = 1

        # Generate objects
        object_count = np.random.randint(self.min_objects, self.max_objects+1)
        object_cls_range = np.arange(2, 12)
        assert len(object_cls_range) >= object_count, "Not enough object classes, range: {}, count: {}".format(object_cls_range, object_count)
        all_objects = np.random.choice(object_cls_range, object_count, replace=False)
        assert len(set(all_objects)) == len(all_objects), "Duplicate object classes: {}".format(all_objects)
        # print(all_objects[0], all_objects[1])

        # First place the target object
        target_object = all_objects[0]
        target_object_position = self.sample_target_position(agent_position)
        self.target[target_object_position] = 1
        x1, x2, y1, y2 = self.get_viewable_range(target_object_position)
        target_range = np.zeros((self.grid_size, self.grid_size))
        target_range[x1:x2, y1:y2] = 1
        # reference point needs to be in the viewable range of the target object and agent
        self.reference_potential = self.reference_potential * target_range

        # Reference object inside viewable grid
        reference_object = all_objects[1]
        reference_object_position = self.sample_reference_position(agent_position)
        sub_path = astar(self.grid, agent_position, reference_object_position)
        self.grid[reference_object_position] = reference_object
        # sub_path include start and end position
        for (x,y) in sub_path:
            self.path[x,y] = 1
            agent_coord.append((x,y))
        # remove the coord of the reference object
        agent_coord.pop()
        self.instructions = "First, go to " + str(object_dict[reference_object]) + ". Then, "

        # Navigate to target object
        sub_path = astar(self.grid, reference_object_position, target_object_position)
        self.grid[target_object_position] = target_object
        # sub_path include start and end position
        for (x,y) in sub_path:
            self.path[x,y] = 1
            agent_coord.append((x,y))
        # remove the coord of the target object
        agent_coord.pop()

        self.instructions += get_direction(reference_object_position, target_object_position) + " to find the " + str(object_dict[target_object]) + "."

        # Place other objects
        for c in range(2, object_count):
            obj_cls = all_objects[c]
            obj_position = self.sample_position()
            while self.grid[obj_position] != 0 or self.path[obj_position] == 1:
                obj_position =self.sample_position()
            self.grid[obj_position] = obj_cls

        # Randomly place obstacles
        n_obstacles = np.random.randint(self.min_obstacles, self.max_obstacles+1)
        for i in range(n_obstacles):
            obstacle_position = self.sample_position()
            while self.grid[obstacle_position] != 0 or self.path[obstacle_position] == 1:
                obstacle_position = self.sample_position()
            self.grid[obstacle_position] = -1

        if self.blur_target:
            self.blur_target_with_kernel()

        # The complete world state 
        full_states = self.grid.copy()
        full_states[init_position] = 1

        return_dict = {
            'observations': self.get_observation_sequence(agent_coord),
            'visible_mask': self.get_mask_sequence(agent_coord),
            'full_states': full_states,
            'path': self.path,
            'target': self.target,
            'instructions': self.instructions,
            'input_ids': target_object,
        }

        # print(self.instructions)
        # print(init_position, reference_object_position, target_object_position, reference_object, target_object)
        # observations = return_dict['observations']
        # print(observations.shape)
        # for i in range(observations.shape[0]):
        #     print(observations[i])

        # print(return_dict['path'])
        # print(np.round(return_dict['target'], 2))
        
        return return_dict

class POMDPAgent:
    def __init__(self, config):
        self.visible_range = config.visible_range
        self.grid_size = config.grid_size

    def initialize(self, states, goal):
        self.states = states
        self.goal = goal
        self.observable_range = np.zeros(self.states.shape)
        self.update_observable_grid(states)
        self.trajectories = (states == 1).astype(np.int32)
        self.current_frame = (states == 1).astype(np.int32)
        self.finished = np.zeros(states.shape[0], dtype=np.int32)

        # print(self.trajectories[0])
        # print(self.current_frame[0])
        # print(self.states[0])

    def get_viewable_range(self, agent_position, visible_range=None):
        if visible_range is None:
            visible_range = self.visible_range
    
        x_min = max(0, agent_position[0]-visible_range)
        x_max = min(self.grid_size, agent_position[0]+visible_range+1)
        y_min = max(0, agent_position[1]-visible_range)
        y_max = min(self.grid_size, agent_position[1]+visible_range+1)
        
        return x_min, x_max, y_min, y_max

    def update_observable_grid(self, frame):
        """
        frame: (b, grid_size, grid_size)
        """
        for i in range(frame.shape[0]):
            agent_position = np.where(frame[i] == 1)
            agent_position = (agent_position[0][0], agent_position[1][0])
            x1, x2, y1, y2 = self.get_viewable_range(agent_position)
            self.observable_range[i, x1:x2, y1:y2] = 1

    def get_observable_states(self):
        """
        Return the observable states of the agent
        """
        next_states = self.states * self.observable_range
        next_states = next_states.astype(np.int32)
        return next_states

    def find_next_position(self, frame, traj, current_position):
        """
        Given the predict path, find the next position of the agent
        """
        adjancent_positions = [(0,1), (0,-1), (1,0), (-1,0)]
        for (x,y) in adjancent_positions:
            next_position = (current_position[0]+x, current_position[1]+y)
            if next_position[0] < 0 or next_position[0] >= self.grid_size or next_position[1] < 0 or next_position[1] >= self.grid_size:
                continue
            if frame[next_position] == 1 and traj[next_position] == 0:
                return next_position
        
        return current_position        

    def step(self, frame):
        """
        The trajectory channel from the prediction frame is used to update the trajectories and return next states.
        frame: (b, grid_size, grid_size)
        """
        frame = np.round(frame)
        next_frame = np.zeros(frame.shape)
        for i in range(frame.shape[0]):
            current_position = np.where(self.current_frame[i] == 1)
            current_position = (current_position[0][0], current_position[1][0])
            # print('current position', current_position, self.states[i, current_position[0], current_position[1]])
            # Don't move if the agent is at the goal
            if self.states[i, current_position[0], current_position[1]] == self.goal[i]:
                next_frame[i] = self.current_frame[i]
                continue
            next_position = self.find_next_position(frame[i], self.trajectories[i], current_position)
            next_frame[i, next_position[0], next_position[1]] = 1
            self.trajectories[i, next_position[0], next_position[1]] = 1
            # print('next position', next_position, self.states[i, next_position[0], next_position[1]])
            if self.states[i, next_position[0], next_position[1]] == 0:
                self.states[i, next_position[0], next_position[1]] = 1
        
        print(self.current_frame[0])
        print(self.trajectories[0])
        self.current_frame = next_frame
        self.update_observable_grid(next_frame)
        next_states = self.get_observable_states()
        print(self.states[0])
        print(np.round(frame[0]))
        print(next_frame[0])

        # b x grid_size x grid_size x 1 -> b x grid_size x grid_size x 512
        repr_frame = repr_dict[next_states]
        # print("repr_frame", repr_frame.shape, next_frame.shape, next_states.shape)
        # b x grid_size x grid_size x 512 -> b x 512 x grid_size x grid_size
        repr_frame = repr_frame.transpose(0, 3, 1, 2)

        return repr_frame


    def step_entire(self, frame):
        """
        The trajectory channel from the prediction frame is used to update the trajectories and return next states.
        Instead of update one step a time, update the entire trajectory at once.
        frame: (b, grid_size, grid_size)
        """
        frame = np.round(frame)
        # In case model predict path out of observable range
        frame = frame * self.observable_range

        next_frame = np.zeros(frame.shape)
        for i in range(frame.shape[0]):
            current_position = np.where(self.current_frame[i] == 1)
            current_position = (current_position[0][0], current_position[1][0])

            # Don't move if the agent is at the goal
            if self.states[i, current_position[0], current_position[1]] == self.goal[i]:
                next_frame[i] = self.current_frame[i]
                continue

            # bfs
            moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            queue = deque([current_position])
            visited = self.trajectories[i]
            visited[current_position] = 1
            while queue:
                pos = queue.popleft()
                for move in moves:
                    new_x, new_y = pos[0] + move[0], pos[1] + move[1]
                    if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size and frame[i, new_x, new_y] == 1 and visited[new_x, new_y] == 0:
                        # update trajectories
                        visited[new_x, new_y] = 1
                        queue.append((new_x, new_y))

                        # update viewable range
                        x1, x2, y1, y2 = self.get_viewable_range(pos)
                        self.observable_range[i, x1:x2, y1:y2] = 1

                        # update states and check if the agent reaches the goal
                        if self.states[i, new_x, new_y] == 0:
                            self.states[i, new_x, new_y] = 1
                        elif self.states[i, new_x, new_y] == self.goal[i]:
                            self.finished[i] = 1
                            break

            # the last position is the end point
            next_position = pos
            next_frame[i, next_position[0], next_position[1]] = 1
        
        # print(self.current_frame[0])
        # print(self.trajectories[0])
        self.current_frame = next_frame
        next_states = self.get_observable_states()
        # print(self.states[0])
        # print(np.round(frame[0]))
        # print(next_frame[0])
        # print(next_states[0])
        print(self.finished[0], np.sum(self.finished))

        # b x grid_size x grid_size x 1 -> b x grid_size x grid_size x 512
        repr_frame = repr_dict[next_states]
        # print("repr_frame", repr_frame.shape, next_frame.shape, next_states.shape)
        # b x grid_size x grid_size x 512 -> b x 512 x grid_size x grid_size
        repr_frame = repr_frame.transpose(0, 3, 1, 2)

        return repr_frame, self.finished



    