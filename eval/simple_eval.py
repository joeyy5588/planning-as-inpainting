import numpy as np
from utils.visualization import visualize_2d_map

class SimpleEvaluator:
    def __init__(self):
        self.total_episode = 0
        self.total_frame = 0
        self.correct_goal = 0
        self.correct_grid = 0
        self.correct_objects = 0
        self.correct_agents = 0
        self.l1_dist = 0
        self.goal_dist = 0

    def reset(self):
        self.total_episode = 0
        self.total_frame = 0
        self.correct_goal = 0
        self.correct_grid = 0
        self.correct_objects = 0
        self.correct_agents = 0
        self.l1_dist = 0
        self.goal_dist = 0

    def update(self, pred, gt, action_dim):
        for b in range(gt.shape[0]):
            self.total_episode += 1
            # skip the first one since it is the initial state
            gt_goal = gt[b, -2:, 0] * 4.5 + 4.5
            gt_goal = np.round(gt_goal)
            min_dist = 100
            min_pred_goal = np.zeros(2)
            for t in range(1, pred.shape[2]):
                # find the closest prediction
                pred_goal = pred[b, action_dim:action_dim + 2, t] * 4.5 + 4.5
                pred_goal = np.round(pred_goal)
                goal_dist = np.sum(np.abs(gt_goal - pred_goal))
                if goal_dist < min_dist:
                    min_dist = goal_dist
                    min_pred_goal = pred_goal
            
            self.goal_dist += min_dist
            self.correct_goal += np.all(gt_goal == min_pred_goal)
            # print(gt_goal, pred_goal)
                
            for t in range(1, gt.shape[2]):
                gt_grid = gt[b, action_dim:, t] * 4.5 + 4.5
                pred_grid = pred[b, action_dim:, t] * 4.5 + 4.5
                gt_grid = np.round(gt_grid)
                pred_grid = np.round(pred_grid)
                # if b == 2:
                #     print(gt_grid, pred_grid)
                # if t == 1:
                #     print(gt_grid, pred_grid)
                gt_agent = gt_grid[:2]
                pred_agent = pred_grid[:2]
                gt_object = gt_grid[2:]
                pred_object = pred_grid[2:]
                if np.all(gt_grid == 0):
                    break
                self.l1_dist += (np.sum(np.abs(gt_grid - pred_grid)) / 4)
                self.total_frame += 1
                self.correct_agents += np.all(gt_agent == pred_agent)
                self.correct_objects += np.all(gt_object == pred_object)

    def summarize(self):
        score_log = {
            "total": (self.correct_agents + self.correct_objects) / (self.total_frame * 2),
            "objects": self.correct_objects / self.total_frame,
            "agents": self.correct_agents / self.total_frame,
            "l1_dist": self.l1_dist / self.total_frame,
            "success_rate": self.correct_goal / self.total_episode,
            "goal_dist": self.goal_dist / self.total_episode,
        }

        # pretty print
        print("="*20)
        print("Evaluation results:")
        for k, v in score_log.items():
            score_log[k] = round(v, 4)
            print(f"{k}: {score_log[k]}")

        print("="*20)
        return score_log

import numpy as np

class Simple2DEvaluator:
    def __init__(self):
        self.total_episode = 0
        self.correct_goal = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.goal_dist = 0

    def reset(self):
        self.total_episode = 0
        self.correct_goal = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.goal_dist = 0

    def update(self, pred, gt):
        pred = np.round(pred)
        for b in range(gt.shape[0]):
            # if b == 0:
            #     print(pred.shape, gt.shape)
            #     print(pred[0][0], gt[0][0])
            #     print(pred[0][1], gt[0][1])
            self.total_episode += 1
            # skip the first one since it is the initial state
            gt_goal_map = gt[b, 0]
            gt_traj_map = gt[b, 1]
            x,y = np.where(gt_goal_map == 1)
            goal_coord = np.array([x[0], y[0]])

            pred_traj_map = pred[b, 1]
            pred_traj = np.where(pred_traj_map == 1)

            min_dist = 100
            min_pred_goal = np.zeros(2)
            for t in range(pred_traj[0].shape[0]):
                # find the closest prediction
                pred_coord = np.array([pred_traj[0][t], pred_traj[1][t]])
                goal_dist = np.sum(np.abs(goal_coord - pred_coord))
                if goal_dist < min_dist:
                    min_dist = goal_dist
                    min_pred_goal = pred_coord
            
            self.goal_dist += min_dist
            self.correct_goal += np.all(goal_coord == min_pred_goal)
            # print(gt_goal, pred_goal)

            tp = np.sum(np.logical_and(pred_traj_map == 1, gt_traj_map == 1))
            fp = np.sum(np.logical_and(pred_traj_map == 1, gt_traj_map == 0))
            fn = np.sum(np.logical_and(pred_traj_map == 0, gt_traj_map == 1))

            self.tp += tp
            self.fp += fp
            self.fn += fn

    def summarize(self):
        score_log = {
            "success_rate": self.correct_goal / self.total_episode,
            "goal_dist": self.goal_dist / self.total_episode,
            "precision": self.tp / (self.tp + self.fp),
            "recall": self.tp / (self.tp + self.fn),
        }

        # pretty print
        print("="*20)
        print("Evaluation results:")
        for k, v in score_log.items():
            score_log[k] = round(v, 4)
            print(f"{k}: {score_log[k]}")

        print("="*20)
        return score_log

