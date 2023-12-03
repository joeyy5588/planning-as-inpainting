import numpy as np
from utils.visualization import visualize_2d_map
from utils.helpers import find_start_end
import matplotlib.pyplot as plt
import os
class GridEvaluator:
    def __init__(self):
        self.total_grid = 0
        self.total_objects = 0
        self.total_obstacles = 0
        self.total_agents = 0
        self.correct_grid = 0
        self.correct_objects = 0
        self.correct_obstacles = 0
        self.correct_agents = 0

    def reset(self):
        self.total_grid = 0
        self.total_objects = 0
        self.total_obstacles = 0
        self.total_agents = 0
        self.correct_grid = 0
        self.correct_objects = 0
        self.correct_obstacles = 0
        self.correct_agents = 0

    def update(self, pred, gt, action_dim):
        pred = np.round(pred)
        for b in range(gt.shape[0]):
            # skip the first one since it is the initial state
            for t in range(1, gt.shape[2]):
                gt_grid = gt[b, action_dim:, t]
                pred_grid = pred[b, action_dim:, t]
                if np.all(gt_grid == 0):
                    break
                self.total_grid += len(gt_grid)
                self.total_obstacles += np.sum(gt_grid == -1)
                self.total_objects += (np.sum(gt_grid > 0) - 1)
                self.total_agents += 1
                self.correct_grid += np.sum(gt_grid == pred_grid)
                self.correct_obstacles += np.sum((gt_grid == -1) & (gt_grid == pred_grid))
                self.correct_objects += np.sum((gt_grid > 0) & (gt_grid == pred_grid))
                self.correct_agents += np.sum((gt_grid == 1) & (gt_grid == pred_grid))

    def summarize(self):
        score_log = {
            "total": self.correct_grid / self.total_grid,
            "obstacles": self.correct_obstacles / self.total_obstacles,
            "objects": self.correct_objects / self.total_objects,
            "agents": self.correct_agents / self.total_agents,
        }

        # pretty print
        print("="*20)
        print("Evaluation results:")
        for k, v in score_log.items():
            score_log[k] = round(v, 4)
            print(f"{k}: {score_log[k]}")

        print("="*20)
        return score_log


class Grid2DEvaluator:
    def __init__(self):
        self.total_episode = 0
        self.total_goal = 0
        self.correct_goal = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.goal_dist = 0

    def reset(self):
        self.total_episode = 0
        self.total_goal = 0
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
            # x,y = np.where(gt_goal_map > 1)
            x, y = np.where(np.logical_and(gt_goal_map>1, gt_traj_map>0))

            self.total_goal += len(x)

            pred_traj_map = pred[b, 1]
            pred_traj = np.where(pred_traj_map > 0)

            if b == 0:
                visualize_2d_map(pred_traj_map, gt_traj_map, gt_goal_map)

            for n in range(len(x)):
                goal_coord = np.array([x[n], y[n]])
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

            tp = np.sum(np.logical_and(gt_traj_map > 0, gt_traj_map == pred_traj_map))
            fn = np.sum(np.logical_and(pred_traj_map == 0, gt_traj_map > 0))
            fp = np.sum(np.logical_and(pred_traj_map > 0, gt_traj_map != pred_traj_map))

            self.tp += tp
            self.fp += fp
            self.fn += fn

    def summarize(self):
        score_log = {
            "success_rate": self.correct_goal / self.total_goal,
            "goal_dist": self.goal_dist / self.total_goal,
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


class Grid2DReprEvaluator(Grid2DEvaluator):
    def __init__(self):
        super().__init__()
        self.output_dir = 'output/visualization/'
        self.approx_correct = 0

    def reset(self):
        self.total_episode = 0
        self.total_goal = 0
        self.correct_goal = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.goal_dist = 0
        self.approx_correct = 0

    def update(self, pred, gt):
        pred = np.round(pred)
        for b in range(gt.shape[0]):
            # if b == 0:
            #     print(pred.shape, gt.shape)
            #     print(pred[0][0], gt[0][0])
            #     print(pred[0][1], gt[0][1])

            self.total_episode += 1
            self.total_goal += 1
            # skip the first one since it is the initial state
            gt_traj_map = gt[b, -1] > 0
            pred_traj_map = pred[b, -1] > 0
            # correct, approx_correct, goal_dist = self.calculate_approx_correct(pred_traj_map, gt_traj_map)
            # self.correct_goal += correct
            # self.approx_correct += approx_correct
            # self.goal_dist += goal_dist

            start, end = find_start_end(gt_traj_map)
            x, y = np.where(pred_traj_map > 0)
            min_dist_start = 100
            min_dist_end = 100
            for i in range(len(x)):
                coord = np.array([x[i], y[i]])
                min_dist_start = np.min([min_dist_start, np.sum(np.abs(start - coord))])
                min_dist_end = np.min([min_dist_end, np.sum(np.abs(end - coord))])

            if min_dist_start == 0 and min_dist_end == 0:
                self.correct_goal += 1
                self.approx_correct += 1
                self.goal_dist += 0
            else:
                min_dist = np.max([min_dist_start, min_dist_end])
                if min_dist == 1:
                    self.correct_goal += 0
                    self.approx_correct += 1
                    self.goal_dist += 1
                else:
                    self.correct_goal += 0
                    self.approx_correct += 0
                    self.goal_dist += min_dist

            if b == 0:
                pred_fn = self.output_dir + 'pred.png'
                gt_fn = self.output_dir + 'gt.png'
                plt.imshow(pred_traj_map)
                plt.colorbar()
                plt.savefig(pred_fn)
                plt.close()

                plt.imshow(gt_traj_map)
                plt.colorbar()
                plt.savefig(gt_fn)
                plt.close()

            tp = np.sum(np.logical_and(gt_traj_map > 0, gt_traj_map == pred_traj_map))
            fn = np.sum(np.logical_and(pred_traj_map == 0, gt_traj_map > 0))
            fp = np.sum(np.logical_and(pred_traj_map > 0, gt_traj_map != pred_traj_map))

            self.tp += tp
            self.fp += fp
            self.fn += fn

    def calculate_approx_correct(self, pred, gt):
        # pred: (H, W)
        # gt: (H, W)
        # find starting point and ending point of gt 
        # function to get neighboring points for a given point (i, j)
        start, end = find_start_end(gt)
        pred_start, pred_end = find_start_end(pred)
        # print(start, end, pred_start, pred_end)
        # Calculate the correctness of the prediction
        # if the start point is actually the start point
        if np.all(start == pred_start):
            # if the end point is actually the end point
            if np.all(end == pred_end):
                return 1, 1, 0
            else:
                if np.sum(np.abs(end - pred_end)) == 1:
                    return 0, 1, 1
                else:
                    return 0, 0, np.sum(np.abs(end - pred_end))

        # if the end point is actually the start point
        elif np.all(end == pred_end):
            if np.all(start == pred_start):
                return 1, 1, 0
            if np.sum(np.abs(start - pred_start)) == 1:
                return 0, 1, 1
            else:
                return 0, 0, np.sum(np.abs(start - pred_start))

        else:

            x, y = np.where(pred > 0)
            min_dist_start = 100
            min_dist_end = 100
            for i in range(len(x)):
                coord = np.array([x[i], y[i]])
                min_dist_start = np.min([min_dist_start, np.sum(np.abs(start - coord))])
                min_dist_end = np.min([min_dist_end, np.sum(np.abs(end - coord))])

            if min_dist_start == 0 and min_dist_end == 0:
                return 1, 1, 0
            else:
                min_dist = np.min([min_dist_start, min_dist_end])
                if min_dist == 1:
                    return 0, 1, 1
                else:
                    return 0, 0, min_dist


    def summarize(self):
        score_log = {
            "success_rate": self.correct_goal / self.total_goal,
            "approx_success_rate": self.approx_correct / self.total_goal,
            "goal_dist": self.goal_dist / self.total_goal,
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



class POMDPReprEvaluator(Grid2DEvaluator):
    def __init__(self):
        super().__init__()
        self.output_dir = 'output/visualization/'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.approx_correct = 0
        self.belief_dist = 0
        self.correct_reference = 0

    def reset(self):
        self.total_episode = 0
        self.total_goal = 0
        self.correct_goal = 0
        self.correct_reference = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.goal_dist = 0
        self.approx_correct = 0
        self.belief_dist = 0

    def update(self, pred, gt, init_states):
        pred = np.round(pred)
        for b in range(gt.shape[0]):
            # if b == 0:
            #     print(pred.shape, gt.shape)
            #     print(pred[0][0], gt[0][0])
            #     print(pred[0][1], gt[0][1])

            self.total_episode += 1
            self.total_goal += 1
            # skip the first one since it is the initial state
            gt_traj_map = gt[b, -1] > 0
            pred_traj_map = pred[b, -1] > 0

            gt_belief_map = gt[b, -2]
            pred_belief_map = pred[b, -2]

            init_state = init_states[b]

            reference_object = np.logical_and(gt_traj_map > 0, gt_belief_map == 0)
            reference_object = np.logical_and(reference_object, init_state > 0)
            reference_correct = np.logical_and(reference_object, pred_traj_map > 0)
            if np.sum(reference_correct) > 0:
                self.correct_reference += 1
                
            
            # print(gt_belief_map.shape, pred_belief_map.shape)
            # print(gt_belief_map, pred_belief_map)
            gt_beleif = np.argmax(gt_belief_map)
            pred_belief = np.argmax(pred_belief_map)
            gt_belif = (gt_beleif // 16, gt_beleif % 16)
            pred_belief = (pred_belief // 16, pred_belief % 16)
            # print(gt_belif, pred_belief)
            self.belief_dist += np.sum(np.abs(np.array(gt_belif) - np.array(pred_belief)))

            start, end = find_start_end(gt_traj_map)
            x, y = np.where(pred_traj_map > 0)
            min_dist_start = 100
            min_dist_end = 100
            for i in range(len(x)):
                coord = np.array([x[i], y[i]])
                min_dist_start = np.min([min_dist_start, np.sum(np.abs(start - coord))])
                min_dist_end = np.min([min_dist_end, np.sum(np.abs(end - coord))])

            if min_dist_start == 0 and min_dist_end == 0:
                self.correct_goal += 1
                self.approx_correct += 1
                self.goal_dist += 0
            else:
                min_dist = np.max([min_dist_start, min_dist_end])
                if min_dist == 1:
                    self.correct_goal += 0
                    self.approx_correct += 1
                    self.goal_dist += 1
                else:
                    self.correct_goal += 0
                    self.approx_correct += 0
                    self.goal_dist += min_dist

            if b == 0:
                pred_fn = self.output_dir + 'pred.png'
                gt_fn = self.output_dir + 'gt.png'
                plt.imshow(pred_traj_map+init_state)
                plt.colorbar()
                plt.savefig(pred_fn)
                plt.close()

                plt.imshow(gt_traj_map+init_state)
                plt.colorbar()
                plt.savefig(gt_fn)
                plt.close()

            tp = np.sum(np.logical_and(gt_traj_map > 0, gt_traj_map == pred_traj_map))
            fn = np.sum(np.logical_and(pred_traj_map == 0, gt_traj_map > 0))
            fp = np.sum(np.logical_and(pred_traj_map > 0, gt_traj_map != pred_traj_map))

            self.tp += tp
            self.fp += fp
            self.fn += fn

    def calculate_approx_correct(self, pred, gt):
        # pred: (H, W)
        # gt: (H, W)
        # find starting point and ending point of gt 
        # function to get neighboring points for a given point (i, j)
        start, end = find_start_end(gt)
        pred_start, pred_end = find_start_end(pred)
        # print(start, end, pred_start, pred_end)
        # Calculate the correctness of the prediction
        # if the start point is actually the start point
        if np.all(start == pred_start):
            # if the end point is actually the end point
            if np.all(end == pred_end):
                return 1, 1, 0
            else:
                if np.sum(np.abs(end - pred_end)) == 1:
                    return 0, 1, 1
                else:
                    return 0, 0, np.sum(np.abs(end - pred_end))

        # if the end point is actually the start point
        elif np.all(end == pred_end):
            if np.all(start == pred_start):
                return 1, 1, 0
            if np.sum(np.abs(start - pred_start)) == 1:
                return 0, 1, 1
            else:
                return 0, 0, np.sum(np.abs(start - pred_start))

        else:

            x, y = np.where(pred > 0)
            min_dist_start = 100
            min_dist_end = 100
            for i in range(len(x)):
                coord = np.array([x[i], y[i]])
                min_dist_start = np.min([min_dist_start, np.sum(np.abs(start - coord))])
                min_dist_end = np.min([min_dist_end, np.sum(np.abs(end - coord))])

            if min_dist_start == 0 and min_dist_end == 0:
                return 1, 1, 0
            else:
                min_dist = np.min([min_dist_start, min_dist_end])
                if min_dist == 1:
                    return 0, 1, 1
                else:
                    return 0, 0, min_dist


    def summarize(self):
        score_log = {
            "success_rate": self.correct_goal / self.total_goal,
            "reference_sr": self.correct_reference / self.total_goal,
            "approx_success_rate": self.approx_correct / self.total_goal,
            "goal_dist": self.goal_dist / self.total_goal,
            "belief_dist": self.belief_dist / self.total_goal,
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