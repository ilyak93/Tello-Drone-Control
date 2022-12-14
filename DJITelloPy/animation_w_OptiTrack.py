import math

import numpy as np


# Rescaling as it is done in TartanVO as the final step if sample has "motion" key, i.e label
def rescale(xyz_GT, xyz_pred):
    scale = np.linalg.norm(xyz_GT[:2])
    return xyz_pred / np.linalg.norm(xyz_pred[:2]) * scale


# calculates the translations from the poses and the pads and also rescales the predictions according to the
# ground-truth

horizontal_dist_between_adjc_pads = 50 #first exp 35, second 50
vertical_dist_btw_diag_adjc_pads = 50

cur_point = (0, 0)
cur_pred = (0, 0)

points_GT = list()
points_pred = list()
points_planned = list()

with open('data/exp2/pose_GT.txt', 'r') as gt_file, \
        open('data/exp2/pose_pred.txt', 'r') as pred_file, \
        open('data/exp2/corrected_pose_GT.txt', 'w+') as corrected_gt_file, \
        open('data/exp2/corrected_pose_pred.txt', 'w+') as corrected_pred_file, \
        open('data/exp2/pose_planned.txt', 'r') as planned_file, \
        open('data/exp2/corrected_pose_planned.txt', 'w+') as corrected_planned_file:
    GT_lines = [line.rstrip() for line in gt_file]
    pred_lines = [line.rstrip() for line in pred_file]
    prev_GT = GT_lines[0].split()
    prev_x_GT, prev_y_GT, prev_z_GT, prev_pad = float(prev_GT[0]), float(prev_GT[1]), float(prev_GT[2]), int(prev_GT[6])
    cur_point = (prev_x_GT, -prev_y_GT)
    cur_pred = (prev_x_GT, -prev_y_GT)
    points_GT.append(cur_point)
    planned_lines = [line.rstrip() for line in planned_file]
    cur_planned = planned_lines[0].split()
    points_planned.append((cur_point[0] + float(cur_planned[0]),
                           cur_point[1] + -float(cur_planned[1])))

    corrected_planned_file.write("%f %f %f %f\n"
                                 % (cur_point[0], cur_point[1],
                                    float(cur_planned[0]),
                                    -float(cur_planned[1])))

    for i in range(1, len(GT_lines) - 1):
        # calculate translation: y positive is from the left of the pad, more intuitive to align to VO where it is to
        # the right

        cur_GT = GT_lines[i].split()
        cur_x_GT, cur_y_GT, cur_z_GT, cur_pad = float(cur_GT[0]), float(cur_GT[1]), float(cur_GT[2]), int(cur_GT[6])
        assert abs(cur_pad - prev_pad) <= 3
        # x_trans, y_trans, z_trans = 0, 0, 0
        if cur_pad == prev_pad:
            x_trans, y_trans, z_trans = [cur_x_GT - prev_x_GT,
                                         -(cur_y_GT - prev_y_GT),
                                         cur_z_GT - prev_z_GT]
        else:
            x_once = math.copysign(1, cur_pad - prev_pad)
            x_forward = int(abs(cur_pad - prev_pad) == 3)
            x_forward = math.copysign(x_forward, cur_pad - prev_pad)
            dir_right = (cur_pad - prev_pad == -1 or cur_pad - prev_pad == 2)
            horz_dist_sign = -1 if dir_right else 1

            cur_x_GT_relative_to_prev = cur_x_GT + \
                                        x_once * vertical_dist_btw_diag_adjc_pads + \
                                        x_forward * vertical_dist_btw_diag_adjc_pads
            cur_y_GT_relative_to_prev = (1 - x_forward) * horz_dist_sign * \
                                        horizontal_dist_between_adjc_pads + cur_y_GT
            x_trans, y_trans, z_trans = [cur_x_GT_relative_to_prev - prev_x_GT,
                                         -(cur_y_GT_relative_to_prev - prev_y_GT),
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
        prev_pad = cur_pad



        corrected_gt_file.write("%f %f %f %s %s %s\n"
                                % (x_trans, y_trans, z_trans,
                                   cur_GT[3], cur_GT[4], cur_GT[5]))

        # rescale and save
        pred = pred_lines[i - 1].split()
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

x_GT = np.array([pt[0] for pt in points_GT])
y_GT = np.array([pt[1] for pt in points_GT])

x_pred_l = [pt[0] for pt in points_pred]
x_pred_l.insert(0, x_GT[0])
x_pred = np.array(x_pred_l)

y_pred_l = [pt[1] for pt in points_pred]
y_pred_l.insert(0, y_GT[0])
y_pred = np.array(y_pred_l)

x_planned= [pt[0] for pt in points_planned[:-1]]
y_planned = [pt[1] for pt in points_planned[:-1]]


for i in range(len(x_GT)):
    # Plotting the Graph
    cur_x_GT = x_GT[:i+1]
    cur_y_GT = y_GT[:i + 1]
    plt.plot(cur_y_GT, cur_x_GT, marker='o', color='b')
    for i, xy in enumerate(zip(cur_y_GT, cur_x_GT)):
       plt.annotate('%d' % i, xy=xy)
    if i > 0:
        plt.plot(y_pred[i], x_pred[i], marker='x', color='r')
    if i < len(x_GT)-2:
        plt.scatter(y_planned[i], x_planned[i], marker='v', color='g')
    plt.title("Groudtruth locations, Visual Odometry estimations and planned navigation")
    plt.xlim([-75, 75])
    plt.ylim([-50, 300])
    plt.xlabel("Y(cm)")
    plt.ylabel("X(cm)")
    plt.savefig('data/exp1/path/'+str(i)+'.png')

import matplotlib.pyplot as plt


# for each point colored in different color, create pic
