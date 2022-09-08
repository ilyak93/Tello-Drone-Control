import numpy as np


# Rescaling as it is done in TartanVO as the final step if sample has "motion" key, i.e label
def rescale(xyz_GT, xyz_pred):
    scale = np.linalg.norm(xyz_GT)
    return xyz_pred / np.linalg.norm(xyz_pred) * scale


# calculates the translations from the poses and the pads and also rescales the predictions according to the
# ground-truth

horizontal_dist_between_adjc_pads = 100
vertical_dist_between_diag_adjc_pads = 50

cur_point = (0, 0)
cur_pred = (0, 0)

points_GT = list()
points_GT.append(cur_point)
points_pred = list()
points_pred.append(cur_pred)

with open('data/pose_GT.txt', 'r') as gt_file, open('data/pose_pred.txt', 'r') as pred_file, \
        open('data/corrected_pose_GT.txt', 'w+') as corrected_gt_file, \
        open('data/corrected_pose_pred.txt', 'w+') as corrected_pred_file:
    GT_lines = [line.rstrip() for line in gt_file]
    pred_lines = [line.rstrip() for line in pred_file]
    prev_GT = GT_lines[0].split()
    prev_x_GT, prev_y_GT, prev_z_GT, prev_pad = float(prev_GT[0]), float(prev_GT[1]), float(prev_GT[2]), int(prev_GT[6])
    cur_point = (prev_x_GT, prev_y_GT)
    cur_pred = (prev_x_GT, prev_y_GT)
    for i in range(1, len(GT_lines)-1):
        # calculate translation: y positive is from the left of the pad, more intuitive to align to VO where it is to
        # the right
        cur_GT = GT_lines[i].split()
        cur_x_GT, cur_y_GT, cur_z_GT, cur_pad = float(cur_GT[0]), float(cur_GT[1]), float(cur_GT[2]), int(cur_GT[6])
        assert abs(cur_pad - prev_pad) <= 3
        #x_trans, y_trans, z_trans = 0, 0, 0
        if cur_pad == prev_pad:
            x_trans, y_trans, z_trans = [cur_x_GT - prev_x_GT,
                                         -(cur_y_GT - prev_y_GT),
                                         cur_z_GT - prev_z_GT]
        else:
            pos_dist_sign = int(abs(cur_pad - prev_pad) == 1)
            neg_dist_sign = -int(abs(cur_pad - prev_pad) == 2)
            sign = pos_dist_sign + neg_dist_sign
            twice_vertical = int(abs(cur_pad - prev_pad) == 3)

            cur_x_GT_relative_to_prev = cur_x_GT + vertical_dist_between_diag_adjc_pads + twice_vertical * vertical_dist_between_diag_adjc_pads
            cur_y_GT_relative_to_prev = cur_y_GT + sign * horizontal_dist_between_adjc_pads
            x_trans, y_trans, z_trans = [cur_x_GT_relative_to_prev - prev_x_GT,
                                         -(cur_y_GT_relative_to_prev - prev_y_GT),
                                         cur_z_GT - prev_z_GT]
        cur_point = [sum(x) for x in zip(cur_point, (x_trans, y_trans))]
        points_GT.append(cur_point)

        prev_x_GT, prev_y_GT, prev_z_GT = cur_x_GT, cur_y_GT, cur_z_GT
        prev_pad = cur_pad

        corrected_gt_file.write("%f %f %f %s %s %s\n"
                                % (x_trans, y_trans, z_trans,
                                   cur_GT[3], cur_GT[4], cur_GT[5]))


        # rescale and save
        pred = pred_lines[i-1].split()
        xyz_pred = np.array([float(pred[0]), float(pred[1]), float(pred[2])])
        scaled_x, scaled_y, scaled_z = rescale(np.array([x_trans, y_trans, z_trans]), xyz_pred)
        cur_pred = [sum(x) for x in zip(cur_pred, (scaled_x, scaled_y))]
        points_pred.append(cur_pred)

        corrected_pred_file.write("%f %f %f %s %s %s\n"
                                % (scaled_x, scaled_y, scaled_z,
                                   cur_GT[3], cur_GT[4], cur_GT[5]))

# TODO: write plotting and saving func

# create cruves

import matplotlib.pyplot as plt

# Dataset
x = np.array([pt[0] for pt in points_GT])
y = np.array([pt[1] for pt in points_GT])

# Plotting the Graph
plt.plot(y, x)
plt.title("Curve plotted using the given GT points")
plt.xlim([-100, 100])
plt.ylim([-50, 200])
plt.xlabel("Y")
plt.ylabel("X")
plt.show()

import matplotlib.pyplot as plt

# Dataset
x = np.array([pt[0] for pt in points_pred])
y = np.array([pt[1] for pt in points_pred])

# Plotting the Graph
plt.plot(y, x)
plt.title("Curve plotted using the given predicted points")
plt.xlim([-100, 100])
plt.ylim([-50, 200])
plt.xlabel("Y")
plt.ylabel("X")
plt.show()



# for each point colored in different color, create pic

