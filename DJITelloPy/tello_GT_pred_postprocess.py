import numpy as np


# Rescaling as it is done in TartanVO as the final step if sample has "motion" key, i.e label
def rescale(xyz_GT, xyz_pred):
    scale = np.linalg.norm(xyz_GT)
    return xyz_pred / np.linalg.norm(xyz_pred) * scale


# calculates the translations from the poses and the pads and also rescales the predictions according to the
# ground-truth
with open('data/pose_GT.txt', 'r') as gt_file, open('data/pose_pred.txt', 'r') as pred_file, \
        open('data/corrected_pose_GT.txt', 'w+') as corrected_gt_file, \
        open('data/scaled_pose_pred.txt', 'w+') as corrected_pred_file:
    GT_lines = [line.rstrip() for line in gt_file]
    pred_lines = [line.rstrip() for line in pred_file]
    prev_GT = GT_lines[0].split()
    prev_x_GT, prev_y_GT, prev_z_GT = float(prev_GT[0]), float(prev_GT[1]), float(prev_GT[2])
    for i in range(1, len(GT_lines)):
        # calculate translation: y positive is from the left of the pad, more intuitive to align to VO where it is to
        # the right
        cur_GT = GT_lines[i].split()
        cur_x_GT, cur_y_GT, cur_z_GT = [float(cur_GT[0]), float(cur_GT[1]), float(cur_GT[2])]
        x_trans, y_trans, z_trans = [cur_x_GT - prev_x_GT,
                                     -(cur_y_GT - prev_y_GT),
                                     cur_z_GT - prev_z_GT]
        corrected_gt_file.write("%f %f %f %s %s %s\n"
                                % (x_trans, y_trans, z_trans,
                                   cur_GT[3], cur_GT[4], cur_GT[5]))
        pred = GT_lines[i].split()
        x_pred, y_pred, z_pred = [float(pred[0]), float(pred[1]), float(pred[2])]
        scaled_x, scaled_y, scaled_z = rescale(np.array([x_trans, y_trans, z_trans]), np.array(pred))
        corrected_gt_file.write("%f %f %f %s %s %s\n"
                                % (scaled_x, scaled_y, scaled_z,
                                   cur_GT[3], cur_GT[4], cur_GT[5]))
