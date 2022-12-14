import math

import numpy as np
import os

from TartanVO.Datasets.transformation import ses2poses_quat, pos_quats2SE_matrices, motion2pose


# Rescaling as it is done in TartanVO as the final step if sample has "motion" key, i.e label
def rescale(xyz_GT, xyz_pred):
    scale = np.linalg.norm(xyz_GT)
    return xyz_pred / np.linalg.norm(xyz_pred) * scale


# calculates the translations from the poses and the pads and also rescales the predictions according to the
# ground-truth

cur_point = (0, 0)
cur_pred = (0, 0)

points_GT = list()
points_pred = list()
points_planned = list()

SEED = 54
BASE_RENDER_DIR = '/home/vista/ilya_tello_test/OL_trajs_images/'
render_dir = os.path.join(BASE_RENDER_DIR, str(SEED))
viz_dir = os.path.join(render_dir, 'viz')
anim_dir = os.path.join(viz_dir, 'anim')
pose_GT = os.path.join(viz_dir, 'pose_GT.txt')
pose_pred = os.path.join(viz_dir, 'pose_pred.txt')
pose_planned = os.path.join(viz_dir, 'pose_planned.txt')

if not os.path.exists(anim_dir):
    os.makedirs(anim_dir)

with open(pose_GT, 'r') as gt_file, \
        open(pose_pred, 'r') as pred_file, \
        open(pose_planned, 'r') as planned_file, \
        open(viz_dir+'/'+'corrected_pose_GT.txt', 'w') as corrected_gt_file, \
        open(viz_dir+'/'+'corrected_pose_pred.txt', 'w') as corrected_pred_file, \
        open(viz_dir+'/'+'corrected_pose_planned.txt', 'w') as corrected_planned_file:
    GT_lines = [line.rstrip() for line in gt_file]
    pred_lines = [line.rstrip() for line in pred_file]
    prev_GT = GT_lines[0].split()
    prev_x_GT, prev_y_GT, prev_z_GT = float(prev_GT[0]), float(prev_GT[1]), float(prev_GT[2])
    cur_point = (prev_x_GT, prev_y_GT)
    cur_pred = (prev_x_GT, prev_y_GT)
    points_GT.append(cur_point)
    planned_lines = [line.rstrip() for line in planned_file]
    cur_planned = planned_lines[0].split()
    points_planned.append((cur_point[0] + float(cur_planned[0]),
                           cur_point[1] + -float(cur_planned[1])))

    corrected_planned_file.write("%f %f %f %f\n"
                                 % (cur_point[0], cur_point[1],
                                    float(cur_planned[0]),
                                    -float(cur_planned[1])))

    for i in range(1, len(GT_lines)):
        # calculate translation: y positive is from the left of the pad, more intuitive to align to VO where it is to
        # the right

        cur_GT = GT_lines[i].split()
        cur_x_GT, cur_y_GT, cur_z_GT = float(cur_GT[0]), float(cur_GT[1]), float(cur_GT[2])
        # x_trans, y_trans, z_trans = 0, 0, 0

        x_trans, y_trans, z_trans = [cur_x_GT - prev_x_GT,
                                     cur_y_GT - prev_y_GT,
                                     cur_z_GT - prev_z_GT]

        cur_point = [sum(x) for x in zip(cur_point, (x_trans, y_trans))]
        points_GT.append(cur_point)

        cur_planned = planned_lines[i].split()
        points_planned.append((cur_point[0] + float(cur_planned[0]),
                               cur_point[1] - float(cur_planned[1])))

        corrected_planned_file.write("%f %f %f %f\n"
                                     % (cur_point[0], cur_point[1],
                                        float(cur_planned[0]),
                                        -float(cur_planned[1])))


        prev_x_GT, prev_y_GT, prev_z_GT = cur_x_GT, cur_y_GT, cur_z_GT

        corrected_gt_file.write("%f %f %f %s %s %s\n"
                                % (x_trans, y_trans, z_trans,
                                   cur_GT[3], cur_GT[4], cur_GT[5]))

        # rescale and save
        pred = pred_lines[i - 1].split()
        #pred = [float(p) for p in pred_lines[i - 1].split()]
        #pred_format = np.array(pred).reshape((1,-1))
        #motions = motion2pose(pred_format)  # for 2 images, we get after this line 2 motions, one being the identity
        #cur_pred = motions[-1]  # hence we take the latter
        xyz_pred = np.array([float(pred[0]), float(pred[1]), float(pred[2])])
        scaled_x, scaled_y, scaled_z = rescale(np.array([x_trans, y_trans, z_trans]), xyz_pred)
        cur_pred = [sum(x) for x in zip(cur_pred, (scaled_x, scaled_y))]
        points_pred.append(cur_pred)

        corrected_pred_file.write("%f %f %f %s %s %s\n"
                                  % (scaled_x, scaled_y, scaled_z,
                                     cur_GT[3], cur_GT[4], cur_GT[5]))

# TODO: write plotting and saving func

#for i in range(len(points_GT))
# create curves

import matplotlib.pyplot as plt

# Dataset

_, x_start, y_start, _, target_x, target_y, _, x_stop, \
    = np.load("alpha_start_pos_target_pos_x_stop.npy")

chasing_x_points = [x_start, target_x]
chasing_y_points = [y_start, target_y]

chasing_stop_plane = x_stop

x_GT = np.array([pt[0] for pt in points_GT])
y_GT = np.array([pt[1] for pt in points_GT])

x_pred_l = [pt[0] for pt in points_pred]

x_pred = np.array(x_pred_l)

y_pred_l = [pt[1] for pt in points_pred]

y_pred = np.array(y_pred_l)

x_planned= [pt[0] for pt in points_planned[:-1]]
y_planned = [pt[1] for pt in points_planned[:-1]]


# Plotting the Graph
plt.rcParams["figure.figsize"] = [3*6.4,3*6.4]
plt.rcParams['font.size'] = 10

plt.plot(chasing_y_points, chasing_x_points, linestyle="--", marker='p', color='k')
plt.axhline(y=x_stop, color='k', linestyle='--')

plt.title("Groudtruth locations, Visual Odometry estimations and planned"
          " navigation with chasing axis", fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=10)

for i in range(len(x_GT)):
    # Plotting the Graph
    cur_x_GT = x_GT[:i+1]
    cur_y_GT = y_GT[:i+1]
    plt.plot(cur_y_GT, cur_x_GT, marker='o', color='b')
    for i, xy in enumerate(zip(cur_y_GT, cur_x_GT)):
       plt.annotate('%d' % i, xy=xy)
    if i > 0:
        plt.plot(y_pred[i-1], x_pred[i-1], marker='x', color='r')
    if i < len(x_GT)-2:
        plt.scatter(y_planned[i], x_planned[i], marker='v', color='g')
    plt.title("Groudtruth locations, Visual Odometry estimations and planned navigation")
    plt.xlim([-250,150])
    plt.xticks(list(range(-250, 150, 5)))
    plt.ylim([-800, 300])
    plt.yticks(list(range(-800, 300, 25)))
    plt.xlabel("Y(cm)")
    plt.ylabel("X(cm)")
    plt.savefig(anim_dir+'/'+str(i)+'.png')


import matplotlib.pyplot as plt


# for each point colored in different color, create pic

