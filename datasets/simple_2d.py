import numpy as np
from .astar import astar
from .buffer import ReplayBuffer
from .normalization import DatasetNormalizer
import torch
from collections import namedtuple
from tqdm import tqdm

RewardBatch = namedtuple('Batch', 'trajectories conditions returns')
Batch = namedtuple('Batch', 'trajectories conditions')

class Simple2DDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        self.max_n_episodes = config.max_n_episodes
        print("Generating paths...")
        fields = []
        for i in tqdm(range(self.max_n_episodes)):
            fields.append(
                generate_new_path(
                    config.grid_size
                )
            )
        self.observation_dim = fields[0]['observations'].shape[-1]
        self.action_dim = fields[0]['actions'].shape[-1]
        self.max_path_length = config.max_path_length
        self.n_episodes = self.max_n_episodes
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
        # sample should be (batch, channel, height, width)
        data_dict = self.fields[idx]
        observations = data_dict['observations']
        conditions = self.get_conditions(observations)
        trajectories = observations.squeeze()
        
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
    grid = np.zeros((grid_size, grid_size))
    path = np.zeros((grid_size, grid_size))

    # Place the agent randomly
    init_position = (np.random.randint(grid_size), np.random.randint(grid_size))
    agent_position = init_position
    grid[agent_position] = -1  # Agent represented by -1

    # Place the object randomly
    obj_position = (np.random.randint(grid_size), np.random.randint(grid_size))
    while obj_position == agent_position:
        obj_position = (np.random.randint(grid_size), np.random.randint(grid_size))
    grid[obj_position] = 1
    sub_path = astar(grid, agent_position, obj_position)
    sub_path.append(obj_position)

    for (x,y) in sub_path:
        path[x][y] = 1

        
    # create observations, actions, terminals
    observations = np.concatenate([np.expand_dims(grid, axis=0), np.expand_dims(path, axis=0)], axis=0)
    actions = []
    terminals = []
    for i in range(len(sub_path)):
        if i == len(sub_path) - 1:
            terminals.append(1.)
            actions.append([0,0,0,0])
        else:
            terminals.append(0.)
            actions.append(get_action(sub_path[i], sub_path[i+1]))

    return_dict = {
        "observations": np.array(observations, dtype=np.float32),
        "actions": np.array(actions),
        "terminals": np.array(terminals).reshape(-1,1),
    }

    # print(grid)
    # print(target_object_list)
    # print(path)
    # print(actions)
    # print(observations)

    return return_dict