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
BASE_RENDER_DIR = 'C:/Users/Ily/Desktop/linkdin/dataset/OL_trajs_images_R25_btt_center_1/'
render_dir = os.path.join(BASE_RENDER_DIR, str(SEED))
viz_dir = os.path.join(render_dir,'viz')
anim_dir = os.path.join(viz_dir, 'anim_xz')
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
    cur_point = (prev_x_GT, prev_z_GT)
    cur_pred = (prev_x_GT, prev_z_GT)
    points_GT.append(cur_point)
    planned_lines = [line.rstrip() for line in planned_file]
    cur_planned = planned_lines[0].split()
    points_planned.append((cur_point[0] + float(cur_planned[0]),
                           cur_point[1] + float(cur_planned[2])))

    corrected_planned_file.write("%f %f %f %f\n"
                                 % (cur_point[0], cur_point[1],
                                    float(cur_planned[0]),
                                    float(cur_planned[2])))

    for i in range(1, len(GT_lines)):
        # calculate translation: y positive is from the left of the pad, more intuitive to align to VO where it is to
        # the right

        cur_GT = GT_lines[i].split()
        cur_x_GT, cur_y_GT, cur_z_GT = float(cur_GT[0]), float(cur_GT[1]), float(cur_GT[2])
        # x_trans, y_trans, z_trans = 0, 0, 0

        x_trans, y_trans, z_trans = [cur_x_GT - prev_x_GT,
                                     cur_y_GT - prev_y_GT,
                                     cur_z_GT - prev_z_GT]

        cur_point = [sum(x) for x in zip(cur_point, (x_trans, z_trans))]
        points_GT.append(cur_point)

        cur_planned = planned_lines[i].split()
        points_planned.append((cur_point[0] + float(cur_planned[0]),
                               cur_point[1] + float(cur_planned[2])))

        corrected_planned_file.write("%f %f %f %f\n"
                                     % (cur_point[0], cur_point[1],
                                        float(cur_planned[0]),
                                        float(cur_planned[2])))


        prev_x_GT, prev_y_GT, prev_z_GT = cur_x_GT, cur_y_GT, cur_z_GT

        corrected_gt_file.write("%f %f %f %s %s %s\n"
                                % (x_trans, y_trans, z_trans,
                                   cur_GT[3], cur_GT[4], cur_GT[5]))

        # rescale and save
        pred = pred_lines[i - 1].split()
        xzy_pred = np.array([float(pred[0]), float(pred[2]), float(pred[1])])
        scaled_x, scaled_z, scaled_y = rescale(np.array([x_trans, z_trans, y_trans]), xzy_pred)
        cur_pred = [sum(x) for x in zip(cur_pred, (scaled_x, -scaled_z))]
        points_pred.append(cur_pred)

        corrected_pred_file.write("%f %f %f %s %s %s\n"
                                  % (scaled_x, scaled_y, scaled_z,
                                     cur_GT[3], cur_GT[4], cur_GT[5]))

# TODO: write plotting and saving func

#for i in range(len(points_GT))
# create curves

import matplotlib.pyplot as plt

# Dataset

#_, x_start, _, z_start, target_x, _, target_z, x_stop, \
#    = np.load("alpha_start_pos_target_pos_x_stop.npy")
x_start, z_start = -700, 155
target_x, target_z = 0, 260
x_stop = -100

chasing_x_points = [x_start, target_x]
chasing_z_points = [z_start, target_z]

chasing_stop_plane = x_stop

x_GT = np.array([pt[0] for pt in points_GT])
z_GT = np.array([pt[1] for pt in points_GT])

x_pred_l = [pt[0] for pt in points_pred]

x_pred = np.array(x_pred_l)

z_pred_l = [pt[1] for pt in points_pred]

z_pred = np.array(z_pred_l)

x_planned = [pt[0] for pt in points_planned[:-1]]
z_planned = [pt[1] for pt in points_planned[:-1]]

plt.rcParams["figure.figsize"] = [3*6.4, 0.5*6.4]
plt.rcParams['font.size'] = 7

plt.plot(chasing_x_points, chasing_z_points, linestyle="--", marker='p', color='k')
plt.axhline(y=x_stop, color='k', linestyle='--')

plt.tick_params(axis='both', which='major', labelsize=5)

for i in range(len(x_GT)):
    # Plotting the Graph
    cur_x_GT = x_GT[:i + 1]
    cur_z_GT = z_GT[:i + 1]
    plt.plot(cur_x_GT, cur_z_GT, marker='o', markersize=5, color='b')
    for i, xz in enumerate(zip(cur_x_GT, cur_z_GT)):
       plt.annotate('%d' % i, xy=xz,  fontsize=5)
    if i > 0:
        plt.plot(x_pred[i - 1], z_pred[i - 1], marker='x', markersize=5, color='r')
        plt.annotate('%d' % (i), xy=(x_pred[i-1], z_pred[i-1]),  fontsize=5)
    if i < len(x_GT) - 2:
        plt.scatter(x_planned[i], z_planned[i], marker='v', s=10, color='g')
        plt.annotate('%d' % i, xy=(x_planned[i], z_planned[i]), fontsize=5)

    plt.title("Groudtruth locations, Visual Odometry estimations and planned"
              " navigation", fontsize=10)
    plt.xlim([-700, 0])
    plt.xticks(list(range(-700, 0, 25)))
    plt.ylim([100, 300])
    plt.yticks(list(range(100, 300, 5)))
    plt.xlabel("Y(cm)")
    plt.ylabel("X(cm)")
    plt.savefig(anim_dir + '/' + str(i) + '.png')



import matplotlib.pyplot as plt


# for each point colored in different color, create pic
