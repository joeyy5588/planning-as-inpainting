import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from utils.helpers import find_start_end
from utils.visualization import visualize_kuka

def visualize_stack(pred, gt):
    lvl_0, lvl_1, lvl_2, lvl_3 = pred[0], pred[1], pred[2], pred[3]
    gt_lvl_0, gt_lvl_1, gt_lvl_2, gt_lvl_3 = gt[0], gt[1], gt[2], gt[3]

    print(lvl_1)
    print(gt_lvl_1)

    # concatenate all levels
    lvl_0_1 = np.concatenate((lvl_0, lvl_1), axis=1)
    lvl_2_3 = np.concatenate((lvl_2, lvl_3), axis=1)
    all_levels = np.concatenate((lvl_0_1, lvl_2_3), axis=0)

    gt_lvl_0_1 = np.concatenate((gt_lvl_0, gt_lvl_1), axis=1)
    gt_lvl_2_3 = np.concatenate((gt_lvl_2, gt_lvl_3), axis=1)
    gt_all_levels = np.concatenate((gt_lvl_0_1, gt_lvl_2_3), axis=0)

    cmap = plt.get_cmap("tab20")  # You can choose different colormaps

    # Create a colormap for object values greater than 2
    object_cmap = plt.cm.get_cmap("tab20", 5)

    # Create a custom colormap for object values
    cmaplist = [cmap(i) for i in range(cmap.N)]
    for i in range(5):
        cmaplist[i] = object_cmap(i)
    custom_cmap = LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    vmin, vmax = min(all_levels.min(), gt_all_levels.min()), max(all_levels.max(), gt_all_levels.max())
    # plot and save
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    im = ax[0].imshow(all_levels, cmap=custom_cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ticks=np.arange(0, 5), ax=ax[0])
    cbar.set_label('Object IDs')
    im = ax[1].imshow(gt_all_levels, cmap=custom_cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ticks=np.arange(0, 5), ax=ax[1])
    cbar.set_label('Object IDs')
    plt.savefig("output/visualization/stack.png")


class KukaEvaluator:
    def __init__(self):
        self.total_stack = 0
        self.correct_stack = 0
        self.approx_correct_stack = 0
        self.stack_dist = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def reset(self):
        self.total_stack = 0
        self.correct_stack = 0
        self.approx_correct_stack = 0
        self.stack_dist = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, pred, gt):
        pred = np.round(pred)
        for b in range(gt.shape[0]):
            # skip the first one since it's condition
            gt_path = gt[b, 1]
            pred_path = pred[b, 1]
            self.total_stack += 1

            start, end = find_start_end(gt_path)
            x, y = np.where(pred_path > 0)
            min_dist_start = 100
            min_dist_end = 100
            for i in range(len(x)):
                coord = np.array([x[i], y[i]])
                min_dist_start = np.min([min_dist_start, np.sum(np.abs(start - coord))])
                min_dist_end = np.min([min_dist_end, np.sum(np.abs(end - coord))])

            if min_dist_start == 0 and min_dist_end == 0:
                self.correct_stack += 1
                self.stack_dist += 0
            else:
                min_dist = np.max([min_dist_start, min_dist_end])
                if min_dist == 1:
                    self.approx_correct_stack += 1
                    self.stack_dist += 1
                else:
                    self.stack_dist += min_dist

            tp = np.sum(np.logical_and(gt_path > 0, gt_path == pred_path))
            fn = np.sum(np.logical_and(pred_path == 0, gt_path > 0))
            fp = np.sum(np.logical_and(pred_path > 0, gt_path != pred_path))

            self.tp += tp
            self.fn += fn
            self.fp += fp

            if b == 0:
                visualize_kuka(pred_path, gt_path, gt[b, 0])


    def update_eval(self, correct_stack, approx_correct_stack, stack_dist, total_stack):
        self.correct_stack += correct_stack
        self.approx_correct_stack += approx_correct_stack
        self.stack_dist += stack_dist
        self.total_stack += total_stack


    def summarize(self):
        score_log = {
            "Task_SR": self.correct_stack / self.total_stack,
            "Approx_SR": (self.correct_stack + self.approx_correct_stack) / self.total_stack,
            "Stack_Dist": self.stack_dist / self.total_stack,
            "Recall": self.tp / (self.tp + self.fn + 1),
            "Precision": self.tp / (self.tp + self.fp + 1),
        }

        # pretty print
        print("="*20)
        print("Evaluation results:")
        for k, v in score_log.items():
            score_log[k] = round(v, 4)
            print(f"{k}: {score_log[k]}")

        print("="*20)
        return score_log

class KukaStackEvaluator2:
    def __init__(self):
        self.total_stack = 0
        self.correct_stack = 0
        self.approx_correct_stack = 0
        self.stack_dist = 0

    def reset(self):
        self.total_stack = 0
        self.correct_stack = 0
        self.approx_correct_stack = 0
        self.stack_dist = 0

    def update(self, pred, gt):
        pred = np.round(pred * 4)
        pred[:, 0] /= 4
        gt = np.round(gt * 4)
        gt[:, 0] /= 4
        for b in range(gt.shape[0]):
            # skip the first one since it's condition
            self.total_stack += 4
            for t in range(1, gt.shape[1]):
                current_pred = pred[b, t]
                current_gt = gt[b, t]
                object_cls = current_gt[np.nonzero(current_gt)][0]
                
                start, end = find_start_end(current_gt)
                x, y = np.where(current_pred > 0)
                min_dist_start = 100
                min_dist_end = 100
                for i in range(len(x)):
                    coord = np.array([x[i], y[i]])
                    min_dist_start = np.min([min_dist_start, np.sum(np.abs(start - coord))])
                    min_dist_end = np.min([min_dist_end, np.sum(np.abs(end - coord))])

                if min_dist_start == 0 and min_dist_end == 0:
                    self.correct_stack += 1
                    self.stack_dist += 0
                else:
                    min_dist = np.max([min_dist_start, min_dist_end])
                    if min_dist == 1:
                        self.approx_correct_stack += 1
                        self.stack_dist += 1
                    else:
                        self.stack_dist += min_dist

            if b == 0:
                visualize_stack(pred[b], gt[b])


    def summarize(self):
        score_log = {
            "Task_SR": self.correct_stack / self.total_stack,
            "Approx_SR": (self.correct_stack + self.approx_correct_stack) / self.total_stack,
            "Stack_Dist": self.stack_dist / self.total_stack
        }

        # pretty print
        print("="*20)
        print("Evaluation results:")
        for k, v in score_log.items():
            score_log[k] = round(v, 4)
            print(f"{k}: {score_log[k]}")

        print("="*20)
        return score_log


class KukaRearrangeEvaluator:
    def __init__(self):
        self.total_stack = 0
        self.correct_stack = 0
        self.approx_correct_stack = 0
        self.stack_dist = 0

    def reset(self):
        self.total_stack = 0
        self.correct_stack = 0
        self.approx_correct_stack = 0
        self.stack_dist = 0

    def update(self, pred, gt):
        pred = np.round(pred)
        for b in range(gt.shape[0]):
            # skip the first one since it's condition
            self.total_stack += 4
            for t in range(1, gt.shape[1]):
                current_pred = pred[b, t-1]
                current_gt = gt[b, t]
                object_cls = gt[np.nonzero(gt)][0]
                gt_coord  = np.argwhere(current_gt == object_cls)[0]
                pred_coord = np.argwhere(current_pred == object_cls)[0]
                if gt_coord[0] == pred_coord[0] and gt_coord[1] == pred_coord[1]:
                    self.correct_stack += 1
                elif abs(gt_coord[0] - pred_coord[0]) + abs(gt_coord[1] - pred_coord[1]) <= 1:
                    self.approx_correct_stack += 1
                self.stack_dist += (abs(gt_coord[0] - pred_coord[0]) + abs(gt_coord[1] - pred_coord[1]))

    def summarize(self):
        score_log = {
            "Task_SR": self.correct_stack / self.total_stack,
            "Approx_SR": (self.correct_stack + self.approx_correct_stack) / self.total_stack,
            "Stack_Dist": self.stack_dist / self.total_stack
        }

        # pretty print
        print("="*20)
        print("Evaluation results:")
        for k, v in score_log.items():
            score_log[k] = round(v, 4)
            print(f"{k}: {score_log[k]}")

        print("="*20)
        return score_log

