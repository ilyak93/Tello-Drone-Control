import cv2
import time
import numpy as np
from djitellopy import Tello
from controller import control_loop
from position import (connectOptitrack, telloState, setwaypoint, waypointUpdate, calc_initial_SE_motive2telloNED_inv,
                      SE_motive2telloNED, tello_go_xyz_speed_from_NED, tello_rotate_clockwise_from_NED,
                      tello_rotate_counter_clockwise_from_NED, patchState, mean_std_hovering)
import os
import csv
from scipy.spatial.transform import Rotation as R
from aruco_detect import ArucoDetector

# Tello's xyz are in ENU system; We want NED. This affect z & yaw.

SEED = 54
BASE_RENDER_DIR = '/home/crl-user/tello_test/OL_trajs_more_images/'
cam_calib_fname =  'tello_960_720_calib_djitellopy.p'
MAX_TIME = 260  # [s]
MAX_INITIAL_PATCH_CHECKS = 5
N_WP = 31  # Number of waypoints. accomodate for the 
body_id_drone1 = 310  # Drone's ID in Motive
body_id_patch = 308  # Patch's ID in Motive
speed = 20  # [cm/s]
dx, dy, dz = (20, 0, 0)  # [cm]. 20 is minimal.
dt_cmd = 3.  # Between commands (avoid SDK's joystick error [s]
dt_b4_grab = 1.  # [s]
dyaw_mu = 0  # [deg]
dyaw_std = 0  # [deg]
N_samples_between = 5
# Waypoints are set relative to starting mocap position. units are [cm]


np.random.seed(SEED)

render_dir = os.path.join(BASE_RENDER_DIR, str(SEED))

if not os.path.exists(render_dir):
    os.makedirs(render_dir)

labels_filename = os.path.join(render_dir, 'pose_file.csv')  # For pose in VO frame
motive_labels_filename = os.path.join(render_dir, 'motive_pose_file.csv')  # For pose in Motive's frame
patch_pose_VO_filename = os.path.join(render_dir, 'patch_pose_VO.csv')

labels_file = open(labels_filename, 'w')
labels_writer = csv.writer(labels_file)
motive_labels_file = open(motive_labels_filename, 'w')
motive_labels_writer = csv.writer(motive_labels_file)
patch_pose_VO_file = open(patch_pose_VO_filename, 'w')
patch_pose_VO_writer = csv.writer(patch_pose_VO_file)


# Initialize the Aruco detector:
ad = ArucoDetector(calib_filename=cam_calib_fname, aruco_type=cv2.aruco.DICT_4X4_50)

# Initialise Tello
tello = Tello()

tello.connect()
print("------------------ Battery % = ", tello.get_battery())

# activate streaming:
tello.streamon()

# initialize the BG frame grabber:
frame_read = tello.get_frame_read()

# Setup connection with optitrack and the waypoints
streamingClient = connectOptitrack(body_id_drone1, body_id_patch)

# Fly to initial height:
tello.takeoff()
time.sleep(1.5 * dt_cmd)  # recover from takeoff is less deterministic
initial_height = 40 + np.abs(np.random.normal(20, 10, 1))
tello.go_xyz_speed(0, 0, round(initial_height[0]), speed)
time.sleep(1.5*dt_cmd)

# t0 = time.time()
# pos_mu, pos_std = mean_std_hovering(streamingClient, N_samples=20)
# print('------ mean sampling time = ', time.time()-t0)
# print('------ pos_mu = {}, pos_std = {}'.format(pos_mu, pos_std))

next_waypoint_idx = 0

# Lets run the control loop
try:
    # Conditionally move until see all 4 markers:
    patch_detected = ad.are_4_markers_detected(frame_read.frame)
    cnt = 0
    while not patch_detected and cnt < MAX_INITIAL_PATCH_CHECKS:
        print('------ Aruco not fully detected! -----------')
        tello_go_xyz_speed_from_NED(tello, int(dx), int(dy), int(dz), speed)
        time.sleep(dt_cmd)
        patch_detected = ad.are_4_markers_detected(frame_read.frame)
        cnt += 1

    if cnt >= MAX_INITIAL_PATCH_CHECKS:
        raise ValueError('Number of trials to detect Aruco exceeded maximal allowed!')

    if N_samples_between > 0:
        dt_cmd /= N_samples_between

    # Save the patch pose relative to tello start in KITTI format and calculate initial location:
    curr_state = telloState(streamingClient)
    patch_state = patchState(streamingClient)
    cv2.imwrite(os.path.join(render_dir, str(next_waypoint_idx).zfill(4) + '.png'), frame_read.frame)

    SE_motive = curr_state[-1]  # in Y UP system
    patch_SE_motive = patch_state[-1]  # in Y UP system
    T_w_b0_inv = calc_initial_SE_motive2telloNED_inv(SE_motive)
    motive_labels_writer.writerow(list(SE_motive[0]) + list(SE_motive[1]) + list(SE_motive[2]))
    SE_tello_NED = np.array(SE_motive2telloNED(SE_motive, T_w_b0_inv))
    labels_writer.writerow(list(SE_tello_NED[0]) + list(SE_tello_NED[1]) + list(SE_tello_NED[2]))
    SE_patch_NED = SE_motive2telloNED(patch_SE_motive, T_w_b0_inv)
    patch_pose_VO_writer.writerow(list(SE_patch_NED[0]) + list(SE_patch_NED[1]) + list(SE_patch_NED[2]))
    patch_pose_VO_file.close()
    next_waypoint_idx += 1

    initial_time = time.time()
    # Outer loop over Waypoints:
    while (time.time() - initial_time) < MAX_TIME and next_waypoint_idx < N_WP:
        start_time = time.time()
        # Initiate the Position CONTROLLER
        curr_state = telloState(streamingClient)  # output is np.array([id_num, rotated_pos, rotated_quat, euler])
        SE_motive = curr_state[-1]  # in Y UP system
        SE_tello_NED = SE_motive2telloNED(SE_motive, T_w_b0_inv)
        euler = R.from_matrix(SE_tello_NED[0:3, 0:3]).as_euler('zyx', degrees=False)
        euler = np.flip(euler)
        print('pos_NED = {} |||| euler_d_NED = {}'.format(SE_tello_NED[0:3, -1].squeeze(), euler / np.pi * 180.))
        # move2WP(tello, curr_state, waypoints[next_waypoint_idx], speed=speed)

        # control_loop(tello, streamingClient, waypoints[next_waypoint_idx])

        dyaw = dyaw_mu
        if dyaw_std > 0 or dyaw_mu > 0:
            dyaw = round(np.random.normal(dyaw_mu, dyaw_std, 1)[0])
            if dyaw > 0:
                # tello.rotate_clockwise(dyaw)
                tello_rotate_clockwise_from_NED(tello, dyaw)
            if dyaw < 0:
                # tello.rotate_counter_clockwise(abs(dyaw))
                tello_rotate_counter_clockwise_from_NED(tello, dyaw)
        time.sleep(1)
        # tello.go_xyz_speed(int(dx), int(dy), int(dz), speed)
        tello_go_xyz_speed_from_NED(tello, int(dx), int(dy), int(dz), speed)

        if N_samples_between > 0:
            for j in range(N_samples_between):
                curr_state = telloState(streamingClient)
                cv2.imwrite(os.path.join(render_dir, str(next_waypoint_idx).zfill(4) + '.png'), frame_read.frame)
                SE_motive = curr_state[-1]
                motive_labels_writer.writerow(list(SE_motive[0]) + list(SE_motive[1]) + list(SE_motive[2]))
                SE_tello_NED = SE_motive2telloNED(SE_motive, T_w_b0_inv)
                labels_writer.writerow(list(SE_tello_NED[0]) + list(SE_tello_NED[1]) + list(SE_tello_NED[2]))
                time.sleep(dt_cmd)
                next_waypoint_idx += 1

        time.sleep(dt_cmd)
        e1 = time.time()

        d_motion = e1 - start_time

        # pos_mu, pos_std = mean_std_hovering(streamingClient, N_samples=240)
        # print('------ pos_mu = {}, pos_std = {}'.format(pos_mu, pos_std))
        curr_state = telloState(streamingClient)
        cv2.imwrite(os.path.join(render_dir, str(next_waypoint_idx).zfill(4) + '.png'), frame_read.frame)
        SE_motive = curr_state[-1]
        motive_labels_writer.writerow(list(SE_motive[0]) + list(SE_motive[1]) + list(SE_motive[2]))
        SE_tello_NED = SE_motive2telloNED(SE_motive, T_w_b0_inv)
        labels_writer.writerow(list(SE_tello_NED[0]) + list(SE_tello_NED[1]) + list(SE_tello_NED[2]))
        d_write = time.time() - e1
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% d_motion, d_write = {}, {}'.format(d_motion, d_write))
        next_waypoint_idx += 1

    print('Total time = {}. Finishing...............'.format(time.time() - initial_time))
    time.sleep(0.5)
    tello.land()
    tello.streamoff()
    print("------------------ Battery % = ", tello.get_battery())
    labels_file.close()
    motive_labels_file.close()

except Exception:
    tello.land()
    print("Main Controller Loop Crashed...")
    tello.streamoff()
    labels_file.close()
    motive_labels_file.close()
    raise
