
import numpy as np
from .astar import astar
from .buffer import ReplayBuffer
from .normalization import DatasetNormalizer
from .pomdp import POMDPGenerator
from collections import namedtuple
from tqdm import tqdm
from transformers import CLIPTextModel, AutoTokenizer
import torch

ConditionBatch = namedtuple('Batch', 'trajectories conditions instructions input_ids init_states, visible_mask')

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


class GridSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        if "train" in config:
            self.train = config.train
        else:
            self.train = True
        self.max_n_episodes = config.max_n_episodes
        print("Generating paths...")
        generator = POMDPGenerator(config)
        fields = []
        for i in tqdm(range(self.max_n_episodes)):
            fields.append(
                generator.generate()
            )
        self.observation_dim = fields[0]['observations'].shape[-1]
        self.max_path_length = config.max_path_length
        self.n_episodes = self.max_n_episodes
        self.fields = fields

    def __len__(self):
        return self.max_n_episodes
    
    def get_conditions(self, observations):
        return {0: observations[:-2]}
    
    def __getitem__(self, idx):
        # sample should be (batch, channel, height, width)
        data_dict = self.fields[idx]        
        # T x H x W
        observations = data_dict['observations']
        # randomly sampled one frame
        if self.train:
            t = np.random.randint(0, observations.shape[0])
        else:
            t = 0

        index_grid = observations[t:t+1]
        init_states = data_dict['full_states']
        visible_mask = data_dict['visible_mask'][t]

        # transform into feature map
        index_grid = index_grid.astype(np.int32)
        repr_grid = repr_dict[index_grid].squeeze()
        repr_grid = repr_grid.transpose(2,0,1)
        
        target = data_dict['target']
        path = data_dict['path']
        instructions = data_dict['instructions']
        input_ids = data_dict['input_ids']
        
        observations = np.concatenate([repr_grid, np.expand_dims(target, axis=0), np.expand_dims(path, axis=0)], axis=0)
        observations = np.array(observations, dtype=np.float32)
        conditions = self.get_conditions(observations)
        trajectories = observations.squeeze()

        batch = ConditionBatch(trajectories, conditions, instructions, input_ids, init_states, visible_mask)
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