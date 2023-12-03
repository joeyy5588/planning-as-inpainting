import numpy as np
from utils.helpers import find_start_end
import matplotlib.pyplot as plt
import os

class AlfredEvaluator():
    def __init__(self):
        super().__init__()
        self.output_dir = 'output/visualization/'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.total_episode = 0
        self.total_goal = 0
        self.correct_goal = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.goal_dist = 0
        self.belief_dist = 0
        self.approx_correct = 0


    def reset(self):
        self.total_episode = 0
        self.total_goal = 0
        self.correct_goal = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.goal_dist = 0
        self.belief_dist = 0
        self.approx_correct = 0

    def update(self, pred, gt):
        pred = np.round(pred)
        for b in range(gt.shape[0]):

            self.total_episode += 1
            self.total_goal += 1
            gt_traj_map = gt[b, -1] > 0
            pred_traj_map = pred[b, -1] > 0

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

            gt_heatmap = gt[b, -2]
            pred_heatmap = pred[b, -2]
            gt_max_coord = np.unravel_index(np.argmax(gt_heatmap, axis=None), gt_heatmap.shape)
            pred_max_coord = np.unravel_index(np.argmax(pred_heatmap, axis=None), pred_heatmap.shape)
            # l1 dist
            self.belief_dist += np.sum(np.abs(np.array(gt_max_coord) - np.array(pred_max_coord)))

            if b == 0:
                pred_fn = self.output_dir + 'pred_alfred.png'
                gt_fn = self.output_dir + 'gt_alfred.png'
                plt.imshow(pred_traj_map)
                plt.colorbar()
                plt.savefig(pred_fn)
                plt.close()

                plt.imshow(gt_traj_map)
                plt.colorbar()
                plt.savefig(gt_fn)
                plt.close()

                pred_heatmap_fn = self.output_dir + 'pred_heatmap_alfred.png'
                gt_heatmap_fn = self.output_dir + 'gt_heatmap_alfred.png'
                plt.imshow(pred_heatmap)
                plt.colorbar()
                plt.savefig(pred_heatmap_fn)
                plt.close()

                plt.imshow(gt_heatmap)
                plt.colorbar()
                plt.savefig(gt_heatmap_fn)
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