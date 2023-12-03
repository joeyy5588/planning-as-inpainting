import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def visualize_2d_map(pred_traj, gt_traj, gt_goal):
    pred_traj = np.abs(pred_traj)
    pred_traj = np.where(gt_goal > 1, gt_goal, pred_traj)
    gt_traj = np.where(gt_goal > 1, gt_goal, gt_traj)
    cmap = plt.get_cmap("tab20")  # You can choose different colormaps

    # Create a colormap for object values greater than 2
    object_cmap = plt.cm.get_cmap("tab20", int(pred_traj.max()))

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create a custom colormap for object values
    cmaplist = [cmap(i) for i in range(cmap.N)]
    for i in range(2, int(pred_traj.max() + 1)):
        cmaplist[i - 1] = object_cmap(i)
    custom_cmap = LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)

    # plot prediciton and save it
    im = ax.imshow(pred_traj, cmap=custom_cmap)
    cbar = plt.colorbar(im, ticks=np.arange(2, pred_traj.max() + 1), ax=ax)
    cbar.set_label('Object IDs')
    plt.savefig('output/visualization/pred_traj.png')

    # plot ground truth and save it
    im = ax.imshow(gt_traj, cmap=custom_cmap)
    plt.savefig('output/visualization/gt_traj.png')

    plt.close()

def visualize_kuka(pred_traj, gt_traj, gt_goal):
    pred_traj = pred_traj > 0
    object_cmap = plt.cm.get_cmap("tab20", int(gt_goal.max()))
    cmaplist = [object_cmap(i) for i in range(int(gt_goal.max() + 1))]
    custom_cmap = LinearSegmentedColormap.from_list('Custom cmap', cmaplist, object_cmap.N)

    # plot prediciton and gt
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    im = ax[0].imshow(pred_traj, cmap=custom_cmap)
    im = ax[1].imshow(gt_traj, cmap=custom_cmap)
    cbar = plt.colorbar(im, ticks=np.arange(2, gt_goal.max() + 1), ax=ax)
    cbar.set_label('Object IDs')
    plt.savefig('output/visualization/kuka.png')

