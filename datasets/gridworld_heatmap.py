
import numpy as np
from .astar import astar
from .buffer import ReplayBuffer
from .normalization import DatasetNormalizer
from collections import namedtuple
from tqdm import tqdm
from transformers import CLIPTextModel, AutoTokenizer
import torch

RewardBatch = namedtuple('Batch', 'trajectories conditions returns')
Batch = namedtuple('Batch', 'trajectories conditions')
ConditionBatch = namedtuple('Batch', 'trajectories conditions instructions input_ids')

object_dict = {
  -1: "Obstacle",
  1: "Agent",
  2: "Apple",
  3: "Carrot",
  4: "Banana",
  5: "Lettuce",
  6: "Orange",
  7: "Tomato",
  8: "Grapes",
  9: "Broccoli",
  10: "Strawberry",
  11: "Cucumber"
}

model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

repr_dict = np.zeros((len(object_dict)+2, 512))
for k, v in object_dict.items():
    repr_dict[k] = model(tokenizer(v, return_tensors="pt").input_ids).last_hidden_state.detach().cpu().numpy()[0, 1]

def gaussian_kernel(size, sigma=1.0):
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

class GridHeatMapDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        self.max_n_episodes = config.max_n_episodes
        print("Generating paths...")
        fields = []
        for i in tqdm(range(self.max_n_episodes)):
            fields.append(
                generate_new_path(
                    config.grid_size, config.max_objects, config.min_objects, config.max_obstacles, config.min_obstacles
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
        return {0: observations[:-2]}
    
    def __getitem__(self, idx):
        # sample should be (batch, channel, height, width)
        data_dict = self.fields[idx]
        observations = data_dict['observations']
        conditions = self.get_conditions(observations)
        trajectories = observations.squeeze()
        instructions = data_dict['instructions']
        input_ids = data_dict['input_ids']
        
        batch = ConditionBatch(trajectories, conditions, instructions, input_ids)
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


def generate_new_path(grid_size=10, max_objects=5, min_object=3, max_obstacles=20, min_obstacles=10):
    grid = np.zeros((grid_size, grid_size))
    target = np.zeros((grid_size, grid_size))
    path = np.zeros((grid_size, grid_size))

    # Place the agent randomly
    init_position = (np.random.randint(grid_size), np.random.randint(grid_size))
    agent_position = init_position
    grid[agent_position] = 1  # Agent represented by -1

    object_count = np.random.randint(min_object, max_objects+1)
    obstacle_count = np.random.randint(min_obstacles, max_obstacles+1)
    object_cls_range = np.arange(2, 12)
    assert len(object_cls_range) >= object_count, "Not enough object classes, range: {}, count: {}".format(object_cls_range, object_count)
    all_objects = np.random.choice(object_cls_range, object_count, replace=False)
    assert len(set(all_objects)) == len(all_objects), "Duplicate object classes: {}".format(all_objects)

    # First object is the target
    target_object = all_objects[0]
    target_object_position = (np.random.randint(grid_size), np.random.randint(grid_size))
    while target_object_position == agent_position:
        target_object_position = (np.random.randint(grid_size), np.random.randint(grid_size))
    sub_path = astar(grid, agent_position, target_object_position)
    grid[target_object_position] = target_object
    target[target_object_position] = 1
    agent_position = target_object_position
    for (x,y) in sub_path:
        path[x,y] = 1
    instructions = "Goto " + str(object_dict[target_object])

    # Randomly place other objects
    for c in range(1, object_count):
        obj_cls = all_objects[c]
        obj_position = (np.random.randint(grid_size), np.random.randint(grid_size))
        while grid[obj_position] != 0 or path[obj_position] == 1:
            obj_position = (np.random.randint(grid_size), np.random.randint(grid_size))
        grid[obj_position] = obj_cls

    # Place obstacles
    for _ in range(obstacle_count):
        obstacle_position = (np.random.randint(grid_size), np.random.randint(grid_size))
        while grid[obstacle_position] != 0 or path[obstacle_position] == 1:
            obstacle_position = (np.random.randint(grid_size), np.random.randint(grid_size))
        grid[obstacle_position] = -1

    grid = grid.astype(np.int32)
    repr_grid = repr_dict[grid]
    repr_grid = repr_grid.transpose(2,0,1)
    # target = apply_kernel_at_target(repr_grid, target_object)
        
    # create observations, actions, terminals
    observations = np.concatenate([repr_grid, np.expand_dims(target, axis=0), np.expand_dims(path, axis=0)], axis=0)
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
        "instructions": instructions,
        "input_ids": target_object,
    }

    # print(grid)
    # print(target_object_list)
    # print(path)
    # print(actions)
    # print(observations)

    return return_dict

