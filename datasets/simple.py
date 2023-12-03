import numpy as np
from .astar import astar
from .buffer import ReplayBuffer
from .normalization import DatasetNormalizer
import torch
from collections import namedtuple
from tqdm import tqdm

RewardBatch = namedtuple('Batch', 'trajectories conditions returns')
Batch = namedtuple('Batch', 'trajectories conditions')

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        fields = ReplayBuffer(config)
        self.max_n_episodes = config.max_n_episodes
        print("Generating paths...")
        for i in tqdm(range(self.max_n_episodes)):
            fields.add_path(
                generate_new_path(
                    config.grid_size
                )
            )
        fields.finalize()
        print("Done! Maximum path length:", fields.path_lengths.max())
        self.normalizer = DatasetNormalizer(fields, config.normalizer, path_lengths=fields['path_lengths'])
        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.max_path_length = config.max_path_length
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.fields = fields
        # self.normalize()

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, -1, self.max_path_length)


    def __len__(self):
        return self.max_n_episodes
    
    def get_conditions(self, observations):
        return {0: observations[0]}
    
    def __getitem__(self, idx):
        # sample should be B x C x T
        observations = self.fields.observations[idx]
        observations = (observations - 4.5) / 4.5
        actions = self.fields.actions[idx]
        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        trajectories = trajectories.transpose(1, 0)
        
        batch = Batch(trajectories, conditions)
        return batch


def get_action(current_position, next_position):
    # return the one-hot action in clockwise order: [go_up, turn_right, go_down, turn_left]
    diff_x = next_position[0] - current_position[0]
    diff_y = next_position[1] - current_position[1]
    # Turn right
    if diff_x == 1:
        return [0,1,0,0]
    # Turn left
    elif diff_x == -1:
        return [0,0,0,1]
    # Go up
    elif diff_y == 1:
        return [1,0,0,0]
    # Go down
    elif diff_y == -1:
        return [0,0,1,0]


def generate_new_path(grid_size=10):
    object_count = 1
    obstacle_count = 0
    grid = np.zeros((grid_size, grid_size))

    # Place the agent randomly
    init_position = (np.random.randint(grid_size), np.random.randint(grid_size))
    agent_position = init_position
    # grid[agent_position] = 1  # Agent represented by 1

    path = []
    target_object_list = []
    target_object_position = []
    # Place the object randomly
    for _ in range(object_count):
        obj_cls = 2
        obj_position = (np.random.randint(grid_size), np.random.randint(grid_size))
        while obj_position == agent_position:
            obj_position = (np.random.randint(grid_size), np.random.randint(grid_size))
        grid[obj_position] = obj_cls
        target_object_list.append(obj_cls)
        target_object_position.append(obj_position)
        sub_path = astar(grid, agent_position, obj_position)
        agent_position = obj_position
        path.extend(sub_path)

    path.append(agent_position)
    
    # Place obstacles
    for _ in range(obstacle_count):
        obstacle_position = (np.random.randint(grid_size), np.random.randint(grid_size))
        while grid[obstacle_position] != 0:
            obstacle_position = (np.random.randint(grid_size), np.random.randint(grid_size))
        grid[obstacle_position] = -1
    
    # create observations, actions, terminals
    observations = []
    actions = []
    terminals = []
    for i in range(len(path)):
        temp_grid = grid.copy()
        temp_grid[path[i]] = 1
        observations.append([path[i][0], path[i][1], target_object_position[0][0], target_object_position[0][1]])
        if i == len(path) - 1:
            terminals.append(1.)
            actions.append([0,0,0,0])
        else:
            terminals.append(0.)
            actions.append(get_action(path[i], path[i+1]))

    return_dict = {
        "observations": np.array(observations),
        "actions": np.array(actions),
        "terminals": np.array(terminals).reshape(-1,1),
    }

    # print(grid)
    # print(target_object_list)
    # print(path)
    # print(actions)
    # print(observations)

    return return_dict