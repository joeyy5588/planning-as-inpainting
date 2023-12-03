import torch
from tqdm import tqdm
import numpy as np
import random
from collections import namedtuple
from .astar import astar
from utils.helpers import find_end_from_start

ConditionBatch = namedtuple('Batch', 'trajectories conditions instructions input_ids')

color_dict = {
    0: "red",
    1: "green",
    2: "blue",
    3: "yellow",
}

def get_above_instructions(goal, i, caption_index):
    if caption_index == 0:
        return "Place %s atop %s." % (color_dict[goal[i]], color_dict[goal[i-1]])
    elif caption_index == 1:
        return "Position %s above %s." % (color_dict[goal[i]], color_dict[goal[i-1]])
    elif caption_index == 2:
        return "Set %s on top of %s." % (color_dict[goal[i]], color_dict[goal[i-1]])
    elif caption_index == 3:
        return "Align %s directly over %s." % (color_dict[goal[i]], color_dict[goal[i-1]])
    elif caption_index == 4:
        return "Mount %s onto %s." % (color_dict[goal[i]], color_dict[goal[i-1]])
    
def get_below_instructions(goal, i, caption_index):
    if caption_index == 0:
        return "Position %s below %s." % (color_dict[goal[i-1]], color_dict[goal[i]])
    elif caption_index == 1:
        return "Arrange %s as the base for %s." % (color_dict[goal[i-1]], color_dict[goal[i]])
    elif caption_index == 2:
        return "Put %s down before placing %s on it." % (color_dict[goal[i-1]], color_dict[goal[i]])
    elif caption_index == 3:
        return "Situate %s beneath %s." % (color_dict[goal[i-1]], color_dict[goal[i]])
    elif caption_index == 4:
        return "Lay %s down and then stack %s above it." % (color_dict[goal[i-1]], color_dict[goal[i]])

class KukaDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        if "train" in config:
            self.train = config.train
        else:
            self.train = True
        self.max_n_episodes = config.max_n_episodes
        self.type = config.type
        generate_function = generate_stacking if self.type == "stacking" else generate_rearrangement
        fields = []
        for i in tqdm(range(self.max_n_episodes)):
            fields.append(
                generate_function(config)
            )
        self.fields = fields

    def __len__(self):
        return self.max_n_episodes
    
    def get_conditions(self, observations):
        return {0: observations[0]}
    
    def __getitem__(self, idx):
        data_dict = self.fields[idx]
        if self.train:
            if self.type == "stacking":
                t = np.random.randint(0, 3)
            else:
                t = np.random.randint(0, 2)
            instructions = data_dict['instructions'][t]
        else:
            t = 0
            # concat all instructions
            # instructions = " ".join(data_dict['instructions'])
            instructions = data_dict['instructions']

        observations = data_dict['observations'][t]
        conditions = self.get_conditions(observations)
        trajectories = observations.squeeze()
        
        input_ids = data_dict['input_ids']
        batch = ConditionBatch(trajectories, conditions, instructions, input_ids)
        return batch


def generate_stacking(config):
    grid_size = config.grid_size
    diverse_instruction = config.diverse_instruction
    
    goal = np.random.permutation(4)
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    all_grid = []
    all_path = []
    all_instructions = []
    all_obj_pos = []

    for i in range(4):
        obj_pos = (np.random.randint(grid_size), np.random.randint(grid_size))
        while grid[obj_pos] != 0:
            obj_pos = (np.random.randint(grid_size), np.random.randint(grid_size))
        grid[obj_pos] = goal[i] + 1

        all_obj_pos.append(obj_pos)
        
        if i == 0:
            bottom_pos = obj_pos
    
    for i in range(1, 4):
        obj_pos = all_obj_pos[i]
        # plan from cube to bottom cube
        path = np.zeros((grid_size, grid_size), dtype=np.float32)
        sub_path = astar(grid, obj_pos, bottom_pos)
        for pos in sub_path:
            path[pos] = 1
        path[bottom_pos] = 1

        all_grid.append(grid.copy())
        all_path.append(path.copy())

        grid[obj_pos] = 0
        grid[bottom_pos] = goal[i] + 1

        if diverse_instruction:
            if random.random() < 0.5:
                caption_index = random.randint(0, 4)
                instructions = get_above_instructions(goal, i, caption_index)
            else:
                caption_index = random.randint(0, 4)
                instructions = get_below_instructions(goal, i, caption_index)
        else:
            instructions = "Stack %s on top of %s." % (color_dict[goal[i]], color_dict[goal[i-1]])

        all_instructions.append(instructions)

    if diverse_instruction:
        concat_instructions = " ".join(all_instructions)
        all_instructions = [concat_instructions] * 3

    return_dict = {
        "observations": [np.concatenate([np.expand_dims(grid, axis=0), np.expand_dims(path, axis=0)], axis=0) for grid, path in zip(all_grid, all_path)],
        "instructions": all_instructions,
        "input_ids": goal,
    }

    return return_dict


def generate_rearrangement(config):
    grid_size = config.grid_size
    diverse_instruction = config.diverse_instruction
    
    goal = np.random.permutation(4)
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    all_grid = []
    all_path = []
    all_instructions = []
    all_obj_pos = []

    for i in range(4):
        obj_pos = (np.random.randint(grid_size), np.random.randint(grid_size))
        while grid[obj_pos] != 0:
            obj_pos = (np.random.randint(grid_size), np.random.randint(grid_size))
        grid[obj_pos] = goal[i] + 1

        all_obj_pos.append(obj_pos)
        
        if i == 0:
            first_bottom_pos = obj_pos
        elif i == 2:
            second_bottom_pos = obj_pos
    
    for i in range(1, 4):
        if i == 2:
            continue
        obj_pos = all_obj_pos[i]
        all_grid.append(grid.copy())
        # plan from cube to bottom cube
        path = np.zeros((grid_size, grid_size), dtype=np.float32)
        if i == 1:
            sub_path = astar(grid, obj_pos, first_bottom_pos)
            path[first_bottom_pos] = 1
            grid[first_bottom_pos] = goal[i] + 1
        elif i == 3:
            sub_path = astar(grid, obj_pos, second_bottom_pos)
            path[second_bottom_pos] = 1
            grid[second_bottom_pos] = goal[i] + 1

        for pos in sub_path:
            path[pos] = 1

        all_path.append(path.copy())

        grid[obj_pos] = 0

        if diverse_instruction:
            if random.random() < 0.5:
                caption_index = random.randint(0, 4)
                instructions = get_above_instructions(goal, i, caption_index)
            else:
                caption_index = random.randint(0, 4)
                instructions = get_below_instructions(goal, i, caption_index)
        else:
            instructions = "Stack %s on top of %s." % (color_dict[goal[i]], color_dict[goal[i-1]])

        all_instructions.append(instructions)

    if diverse_instruction:
        concat_instructions = " ".join(all_instructions)
        all_instructions = [concat_instructions] * 2

    return_dict = {
        "observations": [np.concatenate([np.expand_dims(grid, axis=0), np.expand_dims(path, axis=0)], axis=0) for grid, path in zip(all_grid, all_path)],
        "instructions": all_instructions,
        "input_ids": goal,
    }

    return return_dict



class KukaStackAgent:
    def __init__(self, config):
        self.config = config
        self.grid_size = config.grid_size

    def initialize(self, grid, goal):
        self.grid = grid
        self.init_grid = grid
        self.goal = goal
        self.current_step = 1
        goal_pos = []
        for b in range(grid.shape[0]):
            pos = np.where(grid[b] == (goal[b][0] + 1))
            pos = (pos[0][0], pos[1][0])
            goal_pos.append(pos)
        self.goal_pos = goal_pos
        
        self.correct = 0
        self.approx_correct = 0
        self.goal_dist = 0
        self.prev_sucess = [np.zeros(grid.shape[0]), np.zeros(grid.shape[0]), np.zeros(grid.shape[0])]
        self.total_stack = 0
        self.past_grid = np.zeros((grid.shape[0], 3, grid.shape[1], grid.shape[2]), dtype=np.float32)
        self.past_path = np.zeros((grid.shape[0], 3, grid.shape[1], grid.shape[2]), dtype=np.float32)        

    def step(self, frame):
        frame = np.round(frame).astype(np.int32)
        self.total_stack += frame.shape[0]
        for b in range(frame.shape[0]):
            goal_pos = self.goal_pos[b]
            # print(self.grid[b])
            # print(self.goal[b], self.goal[b][self.current_step] + 1)
            start_pos = np.where(self.grid[b] == (self.goal[b][self.current_step] + 1))
            try:
                start_pos = (start_pos[0][0], start_pos[1][0])
            except:
                # for i in  range(3):
                #     print(self.past_grid[b][i])
                #     print(self.past_path[b][i])
                # print(self.grid[b])
                # print(self.goal[b], self.current_step, self.goal[b][self.current_step] + 1)
                # print(frame[b])
                start_pos = np.where(self.init_grid[b] == (self.goal[b][self.current_step-1] + 1))
                start_pos = (start_pos[0][0], start_pos[1][0])
            frame[b][start_pos] = 1
            end_pos = find_end_from_start(frame[b], start_pos)
            self.past_grid[b][self.current_step-1] = self.grid[b].copy()
            self.past_path[b][self.current_step-1] = frame[b].copy()
            self.grid[b][start_pos] = 0
            

            if self.current_step == 1:

                if end_pos[0] == goal_pos[0] and end_pos[1] == goal_pos[1]:
                    self.correct += 1
                    self.prev_sucess[self.current_step-1][b] = 1
                    self.grid[b][goal_pos] = self.goal[b][self.current_step] + 1
                elif abs(end_pos[0] - goal_pos[0]) + abs(end_pos[1] - goal_pos[1]) <= 1:
                    self.approx_correct += 1
                    self.goal_dist += (abs(end_pos[0] - goal_pos[0]) + abs(end_pos[1] - goal_pos[1]))
                    self.prev_sucess[self.current_step-1][b] = 1
                    self.grid[b][goal_pos] = self.goal[b][self.current_step] + 1
                else:
                    self.goal_dist += (abs(end_pos[0] - goal_pos[0]) + abs(end_pos[1] - goal_pos[1]))
                    self.prev_sucess[self.current_step-1][b] = 0
                    self.grid[b][end_pos] = self.goal[b][self.current_step] + 1

            else:

                if end_pos[0] == goal_pos[0] and end_pos[1] == goal_pos[1]:
                    self.correct += (1 & (self.prev_sucess[self.current_step-2][b] == 1))
                    self.prev_sucess[self.current_step-1][b] = (1 & (self.prev_sucess[self.current_step-2][b] == 1))
                    self.grid[b][goal_pos] = self.goal[b][self.current_step] + 1
                elif abs(end_pos[0] - goal_pos[0]) + abs(end_pos[1] - goal_pos[1]) <= 1:
                    self.approx_correct += (1 & (self.prev_sucess[self.current_step-2][b] == 1))
                    self.goal_dist += (abs(end_pos[0] - goal_pos[0]) + abs(end_pos[1] - goal_pos[1]))
                    self.prev_sucess[self.current_step-1][b] = (1 & (self.prev_sucess[self.current_step-2][b] == 1))
                    self.grid[b][goal_pos] = self.goal[b][self.current_step] + 1
                else:
                    self.goal_dist += (abs(end_pos[0] - goal_pos[0]) + abs(end_pos[1] - goal_pos[1]))
                    self.prev_sucess[self.current_step-1][b] = 0
                    self.grid[b][end_pos] = self.goal[b][self.current_step] + 1


        self.current_step += 1

        return self.grid

    def get_result(self):
        return self.correct, self.approx_correct, self.goal_dist, self.total_stack



class KukaRearrangeAgent:
    def __init__(self, config):
        self.config = config
        self.grid_size = config.grid_size

    def initialize(self, grid, goal):
        self.grid = grid
        self.init_grid = grid
        self.goal = goal
        self.current_step = 0
        goal_pos = []
        for b in range(grid.shape[0]):
            pos = np.where(grid[b] == (goal[b][0] + 1))
            pos_1 = (pos[0][0], pos[1][0])
            pos = np.where(grid[b] == (goal[b][2] + 1))
            pos_2 = (pos[0][0], pos[1][0])
            goal_pos.append([pos_1, pos_2])
        self.goal_pos = goal_pos
        
        self.correct = 0
        self.approx_correct = 0
        self.goal_dist = 0
        self.total_stack = 0
        self.past_grid = np.zeros((grid.shape[0], 2, grid.shape[1], grid.shape[2]), dtype=np.float32)
        self.past_path = np.zeros((grid.shape[0], 2, grid.shape[1], grid.shape[2]), dtype=np.float32)        

    def step(self, frame):
        frame = np.round(frame).astype(np.int32)
        self.total_stack += frame.shape[0]
        for b in range(frame.shape[0]):
            goal_pos = self.goal_pos[b][self.current_step]
            # print(self.grid[b])
            # print(self.goal[b], self.goal[b][self.current_step] + 1)
            start_pos = np.where(self.grid[b] == (self.goal[b][self.current_step * 2 + 1] + 1))

            try:
                start_pos = (start_pos[0][0], start_pos[1][0])
            except:
                # for i in  range(2):
                #     print(self.past_grid[b][i])
                #     print(self.past_path[b][i])
                # print(self.grid[b])
                # print(self.goal[b], self.current_step, self.goal[b][self.current_step] + 1)
                # print(frame[b])
                # print(self.init_grid)
                continue
                start_pos = np.where(self.init_grid[b] == (self.goal[b][self.current_step * 2 + 1] + 1))
                start_pos = (start_pos[0][0], start_pos[1][0])

            frame[b][start_pos] = 1
            end_pos = find_end_from_start(frame[b], start_pos)
            self.past_grid[b][self.current_step] = self.grid[b].copy()
            self.past_path[b][self.current_step] = frame[b].copy()
            self.grid[b][start_pos] = 0

            if end_pos[0] == goal_pos[0] and end_pos[1] == goal_pos[1]:
                self.correct += 1
                self.grid[b][goal_pos] = self.goal[b][self.current_step * 2 + 1] + 1

            elif abs(end_pos[0] - goal_pos[0]) + abs(end_pos[1] - goal_pos[1]) <= 2:
                self.approx_correct += 1
                self.goal_dist += ((abs(end_pos[0] - goal_pos[0]) + abs(end_pos[1] - goal_pos[1])))
                self.grid[b][goal_pos] = self.goal[b][self.current_step * 2 + 1] + 1

            else:
                self.goal_dist += ((abs(end_pos[0] - goal_pos[0]) + abs(end_pos[1] - goal_pos[1])))
                self.grid[b][end_pos] = self.goal[b][self.current_step * 2 + 1] + 1


        self.current_step += 1

        return self.grid

    def get_result(self):
        return self.correct, self.approx_correct, self.goal_dist, self.total_stack