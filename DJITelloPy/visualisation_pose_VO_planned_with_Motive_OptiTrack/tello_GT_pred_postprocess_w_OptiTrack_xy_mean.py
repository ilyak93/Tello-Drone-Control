import math

import numpy as np
import os

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
BASE_RENDER_DIR = 'C:/Users/Ily/Desktop/linkdin/dataset/OL_trajs_images_R25_tbb_center_1/'
render_dir = os.path.join(BASE_RENDER_DIR, str(SEED))
viz_dir = os.path.join(render_dir,'viz')
pose_GT = os.path.join(viz_dir, 'pose_GT.txt')
pose_pred = os.path.join(viz_dir, 'pose_pred.txt')
pose_planned = os.path.join(viz_dir, 'pose_planned.txt')

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
        xyz_pred = np.array([float(pred[0]), float(pred[1]), float(pred[2])])
        scaled_x, scaled_y, scaled_z = rescale(np.array([x_trans, y_trans, z_trans]), xyz_pred)
        cur_pred = [sum(x) for x in zip(cur_pred, (scaled_x, scaled_y))]
        points_pred.append(cur_pred)

        corrected_pred_file.write("%f %f %f %s %s %s\n"
                                  % (scaled_x, scaled_y, scaled_z,
                                     cur_GT[3], cur_GT[4], cur_GT[5]))


# TODO: write plotting and saving func

SEED = 54
BASE_RENDER_DIR = 'C:/Users/Ily/Desktop/linkdin/dataset/OL_trajs_images_R25_tbb_center_2/'
render_dir = os.path.join(BASE_RENDER_DIR, str(SEED))
viz_dir = os.path.join(render_dir,'viz')
pose_GT = os.path.join(viz_dir, 'pose_GT.txt')
pose_pred = os.path.join(viz_dir, 'pose_pred.txt')
pose_planned = os.path.join(viz_dir, 'pose_planned.txt')

points_GT_2 = list()
points_pred_2 = list()
points_planned_2 = list()

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
    points_GT_2.append(cur_point)
    planned_lines = [line.rstrip() for line in planned_file]
    cur_planned = planned_lines[0].split()
    points_planned_2.append((cur_point[0] + float(cur_planned[0]),
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
        points_GT_2.append(cur_point)

        cur_planned = planned_lines[i].split()
        points_planned_2.append((cur_point[0] + float(cur_planned[0]),
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
        xyz_pred = np.array([float(pred[0]), float(pred[1]), float(pred[2])])
        scaled_x, scaled_y, scaled_z = rescale(np.array([x_trans, y_trans, z_trans]), xyz_pred)
        cur_pred = [sum(x) for x in zip(cur_pred, (scaled_x, scaled_y))]
        points_pred_2.append(cur_pred)

        corrected_pred_file.write("%f %f %f %s %s %s\n"
                                  % (scaled_x, scaled_y, scaled_z,
                                     cur_GT[3], cur_GT[4], cur_GT[5]))

#for i in range(len(points_GT))
# create curves

import matplotlib.pyplot as plt

# Dataset

#_, x_start, y_start, _, target_x, target_y, _, x_stop, \
#    = np.load("alpha_start_pos_target_pos_x_stop.npy")
x_start, y_start = -700, -2
target_x, target_y = 600, -2
x_stop = -100

chasing_x_points = [x_start, target_x]
chasing_y_points = [y_start, target_y]

chasing_stop_plane = x_stop

x_GT = [pt[0] for pt in points_GT]
y_GT = [pt[1] for pt in points_GT]

x_pred_l = [pt[0] for pt in points_pred]

x_pred = np.array(x_pred_l)

y_pred_l = [pt[1] for pt in points_pred]

y_pred = np.array(y_pred_l)

x_planned = [pt[0] for pt in points_planned[:-1]]
y_planned = [pt[1] for pt in points_planned[:-1]]\


# -----------------------------------------------------

x_GT_2 = [pt[0] for pt in points_GT_2]
y_GT_2 = [pt[1] for pt in points_GT_2]

x_pred_l_2 = [pt[0] for pt in points_pred_2]

x_pred_2 = np.array(x_pred_l_2)

y_pred_l_2 = [pt[1] for pt in points_pred_2]

y_pred_2 = np.array(y_pred_l_2)

x_planned_2 = [pt[0] for pt in points_planned_2[:-1]]
y_planned_2 = [pt[1] for pt in points_planned_2[:-1]]

# -----------------------------------------------------------

# calculating mean and radius for GT, predicted and planned

x_GT_mean = np.array([(pt1_pt2[0] + pt1_pt2[1]) / 2
                      for pt1_pt2 in list(zip(x_GT, x_GT_2))])
y_GT_mean = np.array([(pt1_pt2[0] + pt1_pt2[1]) / 2
                      for pt1_pt2 in list(zip(y_GT, y_GT_2))])

radius_GT_mean = np.array([max(
    math.sqrt((x_mean-x1)*(x_mean-x1)+(y_mean-y1)*(y_mean-y1)),
    math.sqrt((x_mean-x2)*(x_mean-x2)+(y_mean-y2)*(y_mean-y2)))
    for x_mean, y_mean, x1, y1, x2, y2 in zip(x_GT_mean, y_GT_mean,
                                              x_GT, y_GT, x_GT_2, y_GT_2)])

x_pred_l_mean = [(pt1_pt2[0] + pt1_pt2[1]) / 2
                      for pt1_pt2 in list(zip(x_pred, x_pred_2))]

x_pred_mean = np.array(x_pred_l_mean)

y_pred_l_mean = [(pt1_pt2[0] + pt1_pt2[1]) / 2
                      for pt1_pt2 in list(zip(y_pred, y_pred_2))]

y_pred_mean = np.array(y_pred_l_mean)

radius_pred_mean = np.array([max(
    math.sqrt((x_mean-x1)*(x_mean-x1)+(y_mean-y1)*(y_mean-y1)),
    math.sqrt((x_mean-x2)*(x_mean-x2)+(y_mean-y2)*(y_mean-y2)))
    for x_mean, y_mean, x1, y1, x2, y2 in zip(x_pred_mean, y_pred_mean,
                                              x_pred, y_pred,
                                              x_pred_2, y_pred_2)])

x_planned_mean = [(pt1_pt2[0][0] + pt1_pt2[1][0]) / 2
                      for pt1_pt2 in list(zip(points_planned[:-1],
                                              points_planned_2[:-1]))]

y_planned_mean = [(pt1_pt2[0][1] + pt1_pt2[1][1]) / 2
                      for pt1_pt2 in list(zip(points_planned[:-1],
                                              points_planned_2[:-1]))]

radius_planned_mean = np.array([max(
    math.sqrt((x_mean-x1_y1[0])*(x_mean-x1_y1[0])+(y_mean-x1_y1[1])*(y_mean-x1_y1[1])),
    math.sqrt((x_mean-x2_y2[0])*(x_mean-x2_y2[0])+(y_mean-x2_y2[1])*(y_mean-x2_y2[1])))
    for x_mean, y_mean, x1_y1, x2_y2 in zip(x_planned_mean, y_planned_mean,
                                              points_planned[:-1],
                                              points_planned_2[:-1])])





# Plotting the Graph
plt.rcParams["figure.figsize"] = [0.5*6.4,3*6.4]
plt.rcParams['font.size'] = 7
for i in range(len(y_GT_mean)):
    plt.plot(y_GT_mean[i], x_GT_mean[i], marker='o',
             markersize=radius_GT_mean[i], color='b')
for i, xy in enumerate(zip(y_GT_mean, x_GT_mean)):
   plt.annotate('%d' % i, xy=xy)
for i in range(len(x_pred_mean)):
    plt.plot(y_pred_mean, x_pred_mean, linestyle="--", marker='o',
             markersize=radius_pred_mean[i], color='r')
for i, xy in enumerate(zip(y_pred_mean, x_pred_mean)):
   plt.annotate('%d' % (i+1), xy=xy)
for i, xy in enumerate(zip(y_planned_mean, x_planned_mean)):
    plt.plot(y_planned_mean[i], x_planned_mean[i],
                marker='o', markersize=radius_planned_mean[i], color='g')
for i, xy in enumerate(zip(y_planned_mean, x_planned_mean)):
   plt.annotate('%d' % i, xy=xy)
plt.show()
plt.plot(chasing_y_points, chasing_x_points, linestyle="--", marker='p', color='k')
plt.axhline(y=x_stop, color='k', linestyle='--')

plt.title("Groudtruth locations, Visual Odometry estimations and planned"
          " navigation with chasing axis", fontsize=5)

plt.xlim([-100, 100])
plt.xticks(list(range(-100, 100, 5)))
plt.ylim([-800, 0])
plt.yticks(list(range(-800, 0, 25)))

plt.xlabel("Y(cm)")
plt.ylabel("X(cm)")

plt.tick_params(axis='both', which='major', labelsize=6)

plt.show()

import matplotlib.pyplot as plt


# for each point colored in different color, create pic

