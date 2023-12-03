
import numpy as np
import os
from .astar import astar
from .buffer import ReplayBuffer
from .normalization import DatasetNormalizer
from .pomdp import POMDPGenerator
from collections import namedtuple
from tqdm import tqdm
from transformers import CLIPTextModel, AutoTokenizer
import torch
import pickle 
import json

ConditionBatch = namedtuple('Batch', 'trajectories conditions instructions valid_mask')

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

def blur_target_with_kernel(target):
    kernel = gaussian_kernel(3)
    target_position = np.where(target == 1)
    target_position = (target_position[0][0], target_position[1][0])
    target = apply_kernel_at_target(target, target_position, kernel)
    return target

def expand_target(target):
    expand_size = 3
    target_position = np.where(target == 1)
    target_position = (target_position[0][0], target_position[1][0])
    x1 = target_position[0] - expand_size if target_position[0] - expand_size >= 0 else 0
    x2 = target_position[0] + expand_size if target_position[0] + expand_size <= 48 else 47
    y1 = target_position[1] - expand_size if target_position[1] - expand_size >= 0 else 0
    y2 = target_position[1] + expand_size if target_position[1] + expand_size <= 48 else 47
    target[x1:x2, y1:y2] = 1
    return target

def get_object_in_instruction(traj_data):
    obj_in_inst = []
    pddl = traj_data["pddl_params"]
    for k, object in pddl.items():
        if k != "object_sliced":
            if object == "ButterKnife":
                object = "Knife"
            elif object == "DeskLamp":
                object = "FloorLamp"
            if object != "":
                obj_in_inst.append(object)

    ll_actions = traj_data["plan"]["low_actions"]
    for act in ll_actions:
        if "objectId" in act["api_action"]:
            object = act["api_action"]["objectId"].split("|")[0]
            if object == "ButterKnife":
                object = "Knife"
            elif object == "DeskLamp":
                object = "FloorLamp"
            obj_in_inst.append(object)
    
    obj_in_inst = list(set(obj_in_inst))

    return obj_in_inst

class AlfredAgent:
    def __init__(self,traj_data):
        self.init_pos = (24,24)
        self.grid_position = [(24,24)]
        self.direction = traj_data["scene"]["init_action"]["rotation"]
        self.action = []
        self.process_annotation(traj_data)

    def process_annotation(self, traj_data):
        # y = 0 -> +z -> (x+1, y)
        # y = 90 -> +x -> (x, y+1)
        # y = 180 -> -z -> (x-1, y)
        # y = 270 -> -x -> (x, y-1)
        all_action = [x for x in traj_data["plan"]["low_actions"] if "objectId" not in x["api_action"]]
        for act in all_action:
            self.action.append(act["api_action"]["action"])
            if act["api_action"]["action"] == "MoveAhead":
                if self.direction == 0:
                    self.grid_position.append((self.grid_position[-1][0]+1, self.grid_position[-1][1]))
                elif self.direction == 90:
                    self.grid_position.append((self.grid_position[-1][0], self.grid_position[-1][1]+1))
                elif self.direction == 180:
                    self.grid_position.append((self.grid_position[-1][0]-1, self.grid_position[-1][1]))
                elif self.direction == 270:
                    self.grid_position.append((self.grid_position[-1][0], self.grid_position[-1][1]-1))
            elif act["api_action"]["action"] == "RotateRight":
                self.direction = (self.direction + 90) % 360
                self.grid_position.append(self.grid_position[-1])
            elif act["api_action"]["action"] == "RotateLeft":
                self.direction = (self.direction - 90) % 360
                self.grid_position.append(self.grid_position[-1])
            else:
                self.grid_position.append(self.grid_position[-1])

        self.grid_position = self.grid_position[1:]

        for i, (x, y) in enumerate(self.grid_position):
            if x >= 48:
                x = 47
            if y >= 48:
                y = 47

            self.grid_position[i] = (x, y)
            



class AlfredDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        if "train" in config:
            self.train = config.train
        else:
            self.train = True
        print("Generating paths...")
        self.cat2idx = json.load(open(os.path.join(config.data_root, 'cat2idx.json')))
        self.idx2cat = {v:k for k,v in self.cat2idx.items()}
        fields = self.process_data()
        print("finish generating")
        print("max x boundary: ", self.x_boundary_max)
        print("max y boundary: ", self.y_boundary_max)
        print("max path x boundary: ", self.path_x_boundary_max)
        print("max path y boundary: ", self.path_y_boundary_max)
        self.observation_dim = fields[0]['observations'].shape[-1]
        # self.max_path_length = config.max_path_length
        # self.n_episodes = self.max_n_episodes
        self.fields = fields

    def __len__(self):
        return len(self.fields)
    
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
        
        index_grid = observations[t]
        target = data_dict['target_heatmap']
        path = data_dict['target_path']
        instructions = data_dict['instructions']
        observations = np.concatenate([index_grid, np.expand_dims(target, axis=0), np.expand_dims(path, axis=0)], axis=0)
        observations = np.array(observations, dtype=np.float32)
        conditions = self.get_conditions(observations)
        trajectories = observations.squeeze()

        batch = ConditionBatch(trajectories, conditions, instructions)
        return batch
    
    def process_data(self):
        if self.train:
            split = 'train'
        else:
            split = 'valid_seen'

        self.x_boundary_max = 0
        self.y_boundary_max = 0
        self.path_x_boundary_max = 0
        self.path_y_boundary_max = 0

        root_dir = os.path.join(self.config.data_root, split)

        fields = []
        for sub_dir in tqdm(os.listdir(root_dir)):
            field = self.process_single_task(os.path.join(root_dir, sub_dir))
            fields.extend(field)

        return fields

    def process_single_task(self, task_dir):
        ### Full map consists of multiple channels containing the following:
        ### 1. Obstacle Map
        ### 2. Exploread Area
        ### 3. Current Agent Location
        ### 4. Past Agent Locations
        ### 5,6,7,.. : Semantic Categories

        trial_fn = task_dir.split('/')[-1]
        all_sem_map = np.load(os.path.join(task_dir, 'all_sem_map.npy'))
        metadata = pickle.load(open(os.path.join(task_dir, 'metadata.pkl'), 'rb'))
        all_high_idx = metadata['all_high_idx']
        goal_idx2cat = metadata['goal_idx2cat']
        goal_cat2idx = {v:k for k,v in goal_idx2cat.items()}
        cat_list = list(goal_idx2cat.values())
        traj_data = json.load(open(os.path.join(task_dir, 'traj_data.json'), 'r'))
        instruction_list = traj_data["turk_annotations"]["anns"]
        obj_in_inst = get_object_in_instruction(traj_data)
        alfred_agent = AlfredAgent(traj_data)
        agent_pos = alfred_agent.grid_position

        # start segmenting the observations
        all_traj = []
        # traj_dict = {'observations','path','target','instructions',}
        cur_traj = []
        cur_target_path = np.zeros((48, 48))
        cur_target_heatmap = np.zeros((48, 48))
        cur_path = np.zeros((48, 48))

        for i, high_idx in enumerate(all_high_idx):
            if i + 1 == len(all_high_idx) or high_idx != all_high_idx[i+1]:
                sem_frame = get_current_frame(all_sem_map[i], obj_in_inst, goal_idx2cat, goal_cat2idx, self.cat2idx)
                new_goal_cat2idx = {k:v for k,v in self.cat2idx.items() if k in cat_list}
                cur_path[agent_pos[i]] = 1
                cur_frame = np.concatenate([sem_frame, np.expand_dims(cur_path, axis=0)], axis=0)
                cur_traj.append(cur_frame)

                cur_target_path = cur_path.copy()
                cur_target_heatmap[agent_pos[i]] = 1
                cur_target_heatmap = blur_target_with_kernel(cur_target_heatmap)
                all_traj.append({
                    'observations': np.stack(cur_traj, axis=0),
                    'target_path': cur_target_path,
                    'target_heatmap': cur_target_heatmap,
                    'high_idx': high_idx,
                    'cat2idx': new_goal_cat2idx,
                })
                self.check_max_boundary(sem_frame, cur_path)

                cur_traj = [cur_traj[-1]]
                cur_target_path = np.zeros((48, 48))
                cur_target_heatmap = np.zeros((48, 48))
                cur_path = np.zeros((48, 48))
                cur_path[agent_pos[i]] = 1

            else:
                sem_frame = get_current_frame(all_sem_map[i], obj_in_inst, goal_idx2cat, goal_cat2idx, self.cat2idx)
                cur_path[agent_pos[i]] = 1
                cur_frame = np.concatenate([sem_frame, np.expand_dims(cur_path, axis=0)], axis=0)
                cur_traj.append(cur_frame)

        final_traj = []
        for i in range(len(all_traj)):
            current_traj = all_traj[i]
            for j in range(len(instruction_list)):
                current_instructions = instruction_list[j]["high_descs"]
                current_traj["instructions"] = current_instructions[current_traj["high_idx"]]

                final_traj.append(current_traj.copy())

        return final_traj
    
    def check_max_boundary(self, sem_map, path):
        landmark_map = sem_map[0]
        target_map = sem_map[1]
        
        pos_x, pos_y = landmark_map.nonzero()
        if (pos_x.size > 0) and (pos_y.size > 0):
            x_bound = np.max(pos_x) - np.min(pos_x)
            y_bound = np.max(pos_y) - np.min(pos_y)

            if x_bound > self.x_boundary_max:
                self.x_boundary_max = x_bound
            if y_bound > self.y_boundary_max:
                self.y_boundary_max = y_bound
        
        pos_x, pos_y = target_map.nonzero()
        if (pos_x.size > 0) and (pos_y.size > 0):
            x_bound = np.max(pos_x) - np.min(pos_x)
            y_bound = np.max(pos_y) - np.min(pos_y)

            if x_bound > self.x_boundary_max:
                self.x_boundary_max = x_bound
            if y_bound > self.y_boundary_max:
                self.y_boundary_max = y_bound

        pos_x, pos_y = path.nonzero()
        if (pos_x.size > 0) and (pos_y.size > 0):
            x_bound = np.max(pos_x) - np.min(pos_x)
            y_bound = np.max(pos_y) - np.min(pos_y)
            if x_bound > self.path_x_boundary_max:
                self.path_x_boundary_max = x_bound
            if y_bound > self.path_y_boundary_max:
                self.path_y_boundary_max = y_bound

class AlfredMergeDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        if "train" in config:
            self.train = config.train
        else:
            self.train = True
        print("Generating paths...")
        self.cat2idx = json.load(open(os.path.join(config.data_root, 'newcat2idx.json')))
        self.idx2cat = {v:k for k,v in self.cat2idx.items()}
        fields = self.process_data()
        print("finish generating")
        print("max x boundary: ", self.x_boundary_max)
        print("max y boundary: ", self.y_boundary_max)
        print("max path x boundary: ", self.path_x_boundary_max)
        print("max path y boundary: ", self.path_y_boundary_max)
        self.observation_dim = fields[0]['observations'].shape[-1]
        # self.max_path_length = config.max_path_length
        # self.n_episodes = self.max_n_episodes
        self.fields = fields

    def __len__(self):
        return len(self.fields)
    
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
        
        index_grid = observations[t]
        target = data_dict['target_heatmap']
        path = data_dict['target_path']
        instructions = data_dict['instructions']
        observations = np.concatenate([np.expand_dims(index_grid, axis=0), np.expand_dims(target, axis=0), np.expand_dims(path, axis=0)], axis=0)
        observations = np.array(observations, dtype=np.float32)
        conditions = self.get_conditions(observations)
        trajectories = observations.squeeze()

        batch = ConditionBatch(trajectories, conditions, instructions)
        return batch
    
    def process_data(self):
        if self.train:
            split = 'train'
        else:
            split = 'valid_seen'

        self.x_boundary_max = 0
        self.y_boundary_max = 0
        self.path_x_boundary_max = 0
        self.path_y_boundary_max = 0

        root_dir = os.path.join(self.config.data_root, split)

        fields = []
        for sub_dir in tqdm(os.listdir(root_dir)):
            field = self.process_single_task(os.path.join(root_dir, sub_dir))
            fields.extend(field)

        return fields

    def process_single_task(self, task_dir):
        ### Full map consists of multiple channels containing the following:
        ### 1. Obstacle Map
        ### 2. Exploread Area
        ### 3. Current Agent Location
        ### 4. Past Agent Locations
        ### 5,6,7,.. : Semantic Categories

        trial_fn = task_dir.split('/')[-1]
        all_sem_map = np.load(os.path.join(task_dir, 'all_sem_map.npy'))
        metadata = pickle.load(open(os.path.join(task_dir, 'metadata.pkl'), 'rb'))
        all_high_idx = metadata['all_high_idx']
        goal_idx2cat = metadata['goal_idx2cat']
        goal_cat2idx = {v:k for k,v in goal_idx2cat.items()}
        cat_list = list(goal_idx2cat.values())
        traj_data = json.load(open(os.path.join(task_dir, 'traj_data.json'), 'r'))
        instruction_list = traj_data["turk_annotations"]["anns"]
        obj_in_inst = get_object_in_instruction(traj_data)
        alfred_agent = AlfredAgent(traj_data)
        agent_pos = alfred_agent.grid_position

        # start segmenting the observations
        all_traj = []
        # traj_dict = {'observations','path','target','instructions',}
        cur_traj = []
        cur_target_path = np.zeros((48, 48))
        cur_target_heatmap = np.zeros((48, 48))
        cur_path = np.zeros((48, 48))

        for i, high_idx in enumerate(all_high_idx):
            if i + 1 == len(all_high_idx) or high_idx != all_high_idx[i+1]:
                sem_frame = get_current_frame(all_sem_map[i], obj_in_inst, goal_idx2cat, goal_cat2idx, self.cat2idx)
                landmark_frame, target_frame = sem_frame[0], sem_frame[1]
                new_goal_cat2idx = {k:v for k,v in self.cat2idx.items() if k in cat_list}
                cur_path[agent_pos[i]] = 1
                cur_frame = np.where(cur_path != 0, cur_path, np.where(target_frame != 0, target_frame, landmark_frame))
                cur_traj.append(cur_frame)

                cur_target_path = cur_path.copy()
                cur_target_heatmap[agent_pos[i]] = 1
                cur_target_heatmap = blur_target_with_kernel(cur_target_heatmap)
                all_traj.append({
                    'observations': np.stack(cur_traj, axis=0),
                    'target_path': cur_target_path,
                    'target_heatmap': cur_target_heatmap,
                    'high_idx': high_idx,
                    'cat2idx': new_goal_cat2idx,
                })
                self.check_max_boundary(sem_frame, cur_path)

                cur_traj = [cur_traj[-1]]
                cur_target_path = np.zeros((48, 48))
                cur_target_heatmap = np.zeros((48, 48))
                cur_path = np.zeros((48, 48))
                cur_path[agent_pos[i]] = 1

            else:
                sem_frame = get_current_frame(all_sem_map[i], obj_in_inst, goal_idx2cat, goal_cat2idx, self.cat2idx)
                landmark_frame, target_frame = sem_frame[0], sem_frame[1]
                cur_path[agent_pos[i]] = 1
                cur_frame = np.where(cur_path != 0, cur_path, np.where(target_frame != 0, target_frame, landmark_frame))
                cur_traj.append(cur_frame)

        final_traj = []
        for i in range(len(all_traj)):
            current_traj = all_traj[i]
            for j in range(len(instruction_list)):
                current_instructions = instruction_list[j]["high_descs"]
                try:
                    current_traj["instructions"] = current_instructions[current_traj["high_idx"]] + " " + instruction_list[j]["high_descs"][current_traj["high_idx"]+1]
                except:
                    current_traj["instructions"] = current_instructions[current_traj["high_idx"]]

                final_traj.append(current_traj.copy())

        return final_traj
    
    def check_max_boundary(self, sem_map, path):
        landmark_map = sem_map[0]
        target_map = sem_map[1]
        
        pos_x, pos_y = landmark_map.nonzero()
        if (pos_x.size > 0) and (pos_y.size > 0):
            x_bound = np.max(pos_x) - np.min(pos_x)
            y_bound = np.max(pos_y) - np.min(pos_y)

            if x_bound > self.x_boundary_max:
                self.x_boundary_max = x_bound
            if y_bound > self.y_boundary_max:
                self.y_boundary_max = y_bound
        
        pos_x, pos_y = target_map.nonzero()
        if (pos_x.size > 0) and (pos_y.size > 0):
            x_bound = np.max(pos_x) - np.min(pos_x)
            y_bound = np.max(pos_y) - np.min(pos_y)

            if x_bound > self.x_boundary_max:
                self.x_boundary_max = x_bound
            if y_bound > self.y_boundary_max:
                self.y_boundary_max = y_bound

        pos_x, pos_y = path.nonzero()
        if (pos_x.size > 0) and (pos_y.size > 0):
            x_bound = np.max(pos_x) - np.min(pos_x)
            y_bound = np.max(pos_y) - np.min(pos_y)
            if x_bound > self.path_x_boundary_max:
                self.path_x_boundary_max = x_bound
            if y_bound > self.path_y_boundary_max:
                self.path_y_boundary_max = y_bound


class AlfredReprDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        if "train" in config:
            self.train = config.train
        else:
            self.train = True
        print("Generating paths...")
        self.cat2idx = json.load(open(os.path.join(config.data_root, 'newcat2idx.json')))
        self.idx2cat = {v:k for k,v in self.cat2idx.items()}
        fields = self.process_data()
        print("finish generating")
        print("max x boundary: ", self.x_boundary_max)
        print("max y boundary: ", self.y_boundary_max)
        print("max path x boundary: ", self.path_x_boundary_max)
        print("max path y boundary: ", self.path_y_boundary_max)
        self.observation_dim = fields[0]['observations'].shape[-1]
        # self.max_path_length = config.max_path_length
        # self.n_episodes = self.max_n_episodes
        self.fields = fields
        self.repr_dict = np.zeros((len(self.idx2cat), 512))
        for k, v in self.idx2cat.items():
            if k == 0:
                continue
            self.repr_dict[k] = model(tokenizer(v, return_tensors="pt").input_ids).last_hidden_state.detach().cpu().numpy()[0, 1]

    def __len__(self):
        # if self.train:
        #     return len(self.fields)
        # return 100
        return len(self.fields)
    
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
        
        index_grid = observations[t]
        index_grid = index_grid.astype(np.int32)
        repr_grid = self.repr_dict[index_grid].squeeze()
        repr_grid = repr_grid.transpose(2,0,1)
        target = data_dict['target_heatmap']
        path = data_dict['target_path']
        instructions = data_dict['instructions']
        observations = np.concatenate([repr_grid, np.expand_dims(target, axis=0), np.expand_dims(path, axis=0)], axis=0)
        observations = np.array(observations, dtype=np.float32)
        conditions = self.get_conditions(observations)
        trajectories = observations.squeeze()
        valid_mask = data_dict['valid_mask']

        batch = ConditionBatch(trajectories, conditions, instructions, valid_mask)
        return batch
    
    def process_data(self):
        if self.train:
            split = 'train'
        else:
            split = 'valid_seen'

        self.x_boundary_max = 0
        self.y_boundary_max = 0
        self.path_x_boundary_max = 0
        self.path_y_boundary_max = 0

        root_dir = os.path.join(self.config.data_root, split)

        fields = []
        for sub_dir in tqdm(os.listdir(root_dir)):
            field = self.process_single_task(os.path.join(root_dir, sub_dir))
            fields.extend(field)

        return fields

    def process_single_task(self, task_dir):
        ### Full map consists of multiple channels containing the following:
        ### 1. Obstacle Map
        ### 2. Exploread Area
        ### 3. Current Agent Location
        ### 4. Past Agent Locations
        ### 5,6,7,.. : Semantic Categories

        trial_fn = task_dir.split('/')[-1]
        all_sem_map = np.load(os.path.join(task_dir, 'all_sem_map.npy'))
        metadata = pickle.load(open(os.path.join(task_dir, 'metadata.pkl'), 'rb'))
        all_high_idx = metadata['all_high_idx']
        goal_idx2cat = metadata['goal_idx2cat']
        goal_cat2idx = {v:k for k,v in goal_idx2cat.items()}
        cat_list = list(goal_idx2cat.values())
        traj_data = json.load(open(os.path.join(task_dir, 'traj_data.json'), 'r'))
        instruction_list = traj_data["turk_annotations"]["anns"]
        obj_in_inst = get_object_in_instruction(traj_data)
        alfred_agent = AlfredAgent(traj_data)
        agent_pos = alfred_agent.grid_position

        # start segmenting the observations
        all_traj = []
        # traj_dict = {'observations','path','target','instructions',}
        cur_traj = []
        cur_target_path = np.zeros((48, 48))
        cur_target_heatmap = np.zeros((48, 48))
        cur_path = np.zeros((48, 48))
        valid_mask = np.zeros((48, 48))

        for i, high_idx in enumerate(all_high_idx):
            if i + 1 == len(all_high_idx) or high_idx != all_high_idx[i+1]:
                sem_frame = get_current_frame(all_sem_map[i], obj_in_inst, goal_idx2cat, goal_cat2idx, self.cat2idx)
                landmark_frame, target_frame = sem_frame[0], sem_frame[1]
                new_goal_cat2idx = {k:v for k,v in self.cat2idx.items() if k in cat_list}
                cur_path[agent_pos[i]] = 1
                
                # merge the landmark, object, and path into one channel
                cur_frame = np.where(cur_path != 0, cur_path, np.where(target_frame != 0, target_frame, landmark_frame))
                cur_traj.append(cur_frame)

                # get the boundary for the task
                l_x_min, l_x_max, l_y_min, l_y_max = self.check_max_boundary(sem_frame, cur_path)
                valid_mask[l_x_min:l_x_max, l_y_min:l_y_max] = 1

                cur_target_path = cur_path.copy()
                cur_target_heatmap[agent_pos[i]] = 1
                cur_target_heatmap = blur_target_with_kernel(cur_target_heatmap)
                all_traj.append({
                    'observations': np.stack(cur_traj, axis=0),
                    'valid_mask': valid_mask,
                    'target_path': cur_target_path,
                    'target_heatmap': cur_target_heatmap,
                    'high_idx': high_idx,
                    'cat2idx': new_goal_cat2idx,
                })
                

                cur_traj = [cur_traj[-1]]
                cur_target_path = np.zeros((48, 48))
                cur_target_heatmap = np.zeros((48, 48))
                cur_path = np.zeros((48, 48))
                visible_mask = np.zeros((48, 48))
                cur_path[agent_pos[i]] = 1

            else:
                sem_frame = get_current_frame(all_sem_map[i], obj_in_inst, goal_idx2cat, goal_cat2idx, self.cat2idx)
                landmark_frame, target_frame = sem_frame[0], sem_frame[1]
                cur_path[agent_pos[i]] = 1
                cur_frame = np.where(cur_path != 0, cur_path, np.where(target_frame != 0, target_frame, landmark_frame))
                cur_traj.append(cur_frame)

        final_traj = []
        for i in range(len(all_traj)):
            current_traj = all_traj[i]
            for j in range(len(instruction_list)):
                current_instructions = instruction_list[j]["high_descs"]
                try:
                    current_traj["instructions"] = current_instructions[current_traj["high_idx"]] + " " + instruction_list[j]["high_descs"][current_traj["high_idx"]+1]
                except:
                    current_traj["instructions"] = current_instructions[current_traj["high_idx"]]

                final_traj.append(current_traj.copy())

        return final_traj
    
    def check_max_boundary(self, sem_map, path):
        landmark_map = sem_map[0]
        target_map = sem_map[1]
        loca_x_min, loca_x_max, loca_y_min, loca_y_max = 47, 0, 47, 0
        
        pos_x, pos_y = landmark_map.nonzero()
        if (pos_x.size > 0) and (pos_y.size > 0):
            x_bound = np.max(pos_x) - np.min(pos_x)
            y_bound = np.max(pos_y) - np.min(pos_y)

            if np.max(pos_x) > loca_x_max:
                loca_x_max = np.max(pos_x)
            if np.min(pos_x) < loca_x_min:
                loca_x_min = np.min(pos_x)
            if np.max(pos_y) > loca_y_max:
                loca_y_max = np.max(pos_y)
            if np.min(pos_y) < loca_y_min:
                loca_y_min = np.min(pos_y)

            if x_bound > self.x_boundary_max:
                self.x_boundary_max = x_bound
            if y_bound > self.y_boundary_max:
                self.y_boundary_max = y_bound
        
        pos_x, pos_y = target_map.nonzero()
        if (pos_x.size > 0) and (pos_y.size > 0):
            x_bound = np.max(pos_x) - np.min(pos_x)
            y_bound = np.max(pos_y) - np.min(pos_y)

            if np.max(pos_x) > loca_x_max:
                loca_x_max = np.max(pos_x)
            if np.min(pos_x) < loca_x_min:
                loca_x_min = np.min(pos_x)
            if np.max(pos_y) > loca_y_max:
                loca_y_max = np.max(pos_y)
            if np.min(pos_y) < loca_y_min:
                loca_y_min = np.min(pos_y)

            if x_bound > self.x_boundary_max:
                self.x_boundary_max = x_bound
            if y_bound > self.y_boundary_max:
                self.y_boundary_max = y_bound

        pos_x, pos_y = path.nonzero()
        if (pos_x.size > 0) and (pos_y.size > 0):
            x_bound = np.max(pos_x) - np.min(pos_x)
            y_bound = np.max(pos_y) - np.min(pos_y)

            if np.max(pos_x) > loca_x_max:
                loca_x_max = np.max(pos_x)
            if np.min(pos_x) < loca_x_min:
                loca_x_min = np.min(pos_x)
            if np.max(pos_y) > loca_y_max:
                loca_y_max = np.max(pos_y)
            if np.min(pos_y) < loca_y_min:
                loca_y_min = np.min(pos_y)

            if x_bound > self.path_x_boundary_max:
                self.path_x_boundary_max = x_bound
            if y_bound > self.path_y_boundary_max:
                self.path_y_boundary_max = y_bound

        return loca_x_min, loca_x_max, loca_y_min, loca_y_max

                

def get_current_frame(sem_map, obj_in_inst, goal_idx2cat, goal_cat2idx, cat2idx):
    # return is 2 x 48 x 48
    landmark_map = sem_map.copy()

    for obj in obj_in_inst:
        landmark_map[4+goal_cat2idx[obj]] = 0

    target_map = sem_map - landmark_map

    landmark_map[-1] = 1e-5
    target_map[-1] = 1e-5    
    # maxpooling 
    landmark_map = np.argmax(landmark_map[4:], axis=0)
    target_map = np.argmax(target_map[4:], axis=0)

    new_landmark_map = np.zeros_like(landmark_map)
    new_target_map = np.zeros_like(target_map)

    for k, v in goal_idx2cat.items():
        new_landmark_map[landmark_map == k] = cat2idx[v]
        new_target_map[target_map == k] = cat2idx[v]

    current_frame = np.stack([new_landmark_map, new_target_map], axis=0)

    return current_frame