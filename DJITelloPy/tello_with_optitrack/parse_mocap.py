import cv2
import os
import numpy as np
import re
import pandas as pd
from Datasets.transformation import ses2poses_quat, SE2pos_quat, pos_quats2SE_matrices
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def motive2NED(ses):
    NED_ses = []
    for se in ses:
        R_se = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        NED_ses.append(R_se @ np.array(se))
    return NED_ses


def invert_SE(se):
    ''' More stable than pure inversion'''
    inv_se = np.eye(4)
    R_T = se[0:3, 0:3].T
    inv_se[0:3, 0:3] = R_T
    inv_se[0:3, -1] = - R_T @ se[0:3, -1]
    return inv_se


def shift_to_origin_SEs(traj_ses):
    '''
    Traj: a list of kitti_pose
    Return: translate and rotate the traj and convert to [t, quat]
    '''
    traj_init_inv_se = invert_SE(traj_ses[0])
    # new_traj = [np.eye(4)]
    # for i in range(len(traj_ses)-1):
    #     new_traj.append(traj_init_inv_se @ traj_ses[i+1])
    new_traj = [traj_init_inv_se @ se for se in traj_ses]
    return np.array(new_traj)


def shift_world_traj_to_VO(traj_ses):
    '''
    Traj: a list of SEs
    '''
    #traj_init_inv_se = invert_SE(traj_ses[0])
    # new_traj = [np.eye(4)]
    # for i in range(len(traj_ses)-1):
    #     new_traj.append(traj_init_inv_se @ traj_ses[i+1])
    #new_traj = [traj_init_inv_se @ se for se in traj_ses]

    new_traj = []
    T_motive_2NED_inv = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    T_motive_2NED = invert_SE(T_motive_2NED_inv)
    T_w_0 = np.array(traj_ses[0])
    T_w_b0 = T_w_0 @ T_motive_2NED
    T_w_b0_inv = invert_SE(T_w_b0)
    for se in traj_ses:
        T_w_bi = np.array(se) @ T_motive_2NED
        T_b0_bi = T_w_b0_inv @ T_w_bi
        new_traj.append(T_b0_bi)
    return np.array(new_traj)


def SEs_to_Kitti(ses):
    ses_kitti = [list(p[0]) + list(p[1]) + list(p[2]) for p in ses]
    return ses_kitti


def plot_traj(SE_poses, title_str=''):
    SE_poses = np.array(SE_poses)
    fig_traj = plt.figure()
    ax = Axes3D(fig_traj)
    ax.scatter(SE_poses[:, 0, -1], SE_poses[:, 1, -1], SE_poses[:, 2, -1], label='traj', s=15)
    ax.set_xlim3d(-1, 3)
    ax.set_ylim3d(-1, 3)
    ax.set_zlim3d(-2, 0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title_str)
    plt.legend()
    plt.show()

########################################################################################

base_dir_path = '/home/vista/renders/DJI_Mavic'
mocap_fname = 'mavic_and_target.csv'
mocap_file = os.path.join(base_dir_path, mocap_fname)
output_motion_fname = os.path.join(base_dir_path, 'processed_' + mocap_fname)

header_rows_num = 6
isTargetData = True  # if target exists, its data is assumed AFTER the drone tracked data


df = pd.read_csv(mocap_file, skiprows=header_rows_num)
df.drop(df.columns[[0, 1]], axis=1, inplace=True)  # remove the first 2 columns: frame idx & time
data = df.to_numpy()
if isTargetData:
    target_pose_VO_fname = os.path.join(base_dir_path, 'target_pos_VO.csv')
    target_pose_world_fname = os.path.join(base_dir_path, 'target_pos_world.csv')
    target_pose = data[0, 7:].reshape((1, 7))
    target_pose = pos_quats2SE_matrices(target_pose, pos_first=False)
    np.savetxt(target_pose_world_fname, np.array(SEs_to_Kitti(target_pose)), delimiter=',')

data = data[:, 0:7]
data = pos_quats2SE_matrices(data, pos_first=False)  # Motive saves orientation first

if isTargetData:
    T_motive_2NED_inv = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    T_motive_2NED = invert_SE(T_motive_2NED_inv)
    T_w_0 = np.array(data[0])
    T_w_b0 = T_w_0 @ T_motive_2NED
    T_w_b0_inv = invert_SE(T_w_b0)
    target_pose_VO = T_w_b0_inv @ target_pose


plot_traj(data, title_str='b4 coord shift')

shifted_data = shift_world_traj_to_VO(data)

plot_traj(shifted_data, title_str='after coord shift')

data_kitti = np.array(SEs_to_Kitti(shifted_data))

np.savetxt(output_motion_fname, data_kitti, delimiter=',')
