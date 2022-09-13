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
from threading import Thread


def videoRecorder(n_frames=90, FPS=30):
    # create a VideoWrite object, recording to ./video.avi
    height, width, _ = frame_read.frame.shape
    video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'XVID'), FPS, (width, height))

    idx_frame = 0
    while idx_frame < n_frames:
        video.write(frame_read.frame)
        time.sleep(1 / FPS)
        idx_frame += 1
    video.release()


def dataRecorder(n_frames=90, FPS=30, wait_b4=1.):
    time.sleep(wait_b4)
    height, width, _ = frame_read.frame.shape
    idx_frame = 0
    # Save the patch pose relative to tello start in KITTI format and calculate initial location:
    curr_state = telloState(streamingClient)  # output is np.array([id_num, rotated_pos, rotated_quat, euler])
    patch_state = patchState(streamingClient)
    cv2.imwrite(os.path.join(render_dir, str(idx_frame).zfill(4) + '.png'), frame_read.frame)
    SE_motive = curr_state[-1]  # in Y UP system
    patch_SE_motive = patch_state[-1]  # in Y UP system
    T_w_b0_inv = calc_initial_SE_motive2telloNED_inv(SE_motive)
    motive_labels_writer.writerow(list(SE_motive[0]) + list(SE_motive[1]) + list(SE_motive[2]))
    SE_tello_NED = np.array(SE_motive2telloNED(SE_motive, T_w_b0_inv))
    labels_writer.writerow(list(SE_tello_NED[0]) + list(SE_tello_NED[1]) + list(SE_tello_NED[2]))
    SE_patch_NED = SE_motive2telloNED(patch_SE_motive, T_w_b0_inv)
    patch_pose_VO_writer.writerow(list(SE_patch_NED[0]) + list(SE_patch_NED[1]) + list(SE_patch_NED[2]))
    patch_pose_VO_file.close()
    idx_frame += 1

    while idx_frame < n_frames:
        curr_state = telloState(streamingClient)
        cv2.imwrite(os.path.join(render_dir, str(idx_frame).zfill(4) + '.png'), frame_read.frame)
        SE_motive = curr_state[-1]  # in Y UP system
        motive_labels_writer.writerow(list(SE_motive[0]) + list(SE_motive[1]) + list(SE_motive[2]))
        SE_tello_NED = SE_motive2telloNED(SE_motive, T_w_b0_inv)
        labels_writer.writerow(list(SE_tello_NED[0]) + list(SE_tello_NED[1]) + list(SE_tello_NED[2]))
        time.sleep(1 / FPS)
        idx_frame += 1


# Tello's xyz are in ENU system; We want NED. This affect z & yaw.

SEED = 7
BASE_RENDER_DIR = '/home/crl-user/tello_test/OL_speed_60_160522_big_target/'
cam_calib_fname =  'tello_960_720_calib_djitellopy.p'
MAX_TIME = 60  # [s]
CHECK_INITIAL_PATCH_DETECTION = False
MAX_INITIAL_PATCH_CHECKS = 5
N_WP = 90  # Number of waypoints
GO_HIGHER = True
fps = 30
wait_b4_start_cam = 2.7  # delay between start motion command and start capturing frames
body_id_drone1 = 310  # Drone's ID in Motive
body_id_patch = 308  # Patch's ID in Motive
speed = 50  # [cm/s]
dx, dy, dz = (500, 0, 0)  # [cm]. 20 is minimal.
max_random_height_offset = 170   # [cm]
RAND_HEIGHT = False
dt_cmd = 3.  # Between commands (avoid SDK's joystick error [s]
dyaw_mu = 0  # [deg]
dyaw_std = 0  # [deg]
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

recorder = Thread(target=dataRecorder, args=(N_WP, fps, wait_b4_start_cam))

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

# Setup connection with Optitrack and the waypoints
streamingClient = connectOptitrack(body_id_drone1, body_id_patch)

# Fly to initial height:
tello.takeoff()
time.sleep(1.5 * dt_cmd)  # recover from takeoff is less deterministic
if GO_HIGHER:
    if RAND_HEIGHT:
        initial_height = round(max(20, min(np.array([max_random_height_offset]), 60 + np.abs(np.random.normal(20, 20, 1)))[0])) # minimal z is 20
    else:
        initial_height = max_random_height_offset
    tello.go_xyz_speed(0, 0, initial_height, 50)
    time.sleep(1.5*dt_cmd)

# t0 = time.time()
# pos_mu, pos_std = mean_std_hovering(streamingClient, N_samples=20)
# print('------ mean sampling time = ', time.time()-t0)
# print('------ pos_mu = {}, pos_std = {}'.format(pos_mu, pos_std))

# Let's run the control loop
try:
    # Conditionally move until see all 4 markers:
    patch_detected = ad.are_4_markers_detected(frame_read.frame)
    if not CHECK_INITIAL_PATCH_DETECTION and not patch_detected:
        raise ValueError('Aruco markers not detected! aborting')

    if CHECK_INITIAL_PATCH_DETECTION:
        cnt = 0
        while not patch_detected and cnt < MAX_INITIAL_PATCH_CHECKS:
            print('------ Aruco not fully detected! -----------')
            tello_go_xyz_speed_from_NED(tello, int(dx), int(dy), int(dz), speed)
            time.sleep(dt_cmd)
            patch_detected = ad.are_4_markers_detected(frame_read.frame)
            cnt += 1

        if cnt >= MAX_INITIAL_PATCH_CHECKS:
            raise ValueError('Number of trials to detect Aruco exceeded maximal allowed!')

    initial_time = time.time()

    # dyaw = dyaw_mu
    # if dyaw_std > 0 or dyaw_mu > 0:
    #     dyaw = round(np.random.normal(dyaw_mu, dyaw_std, 1)[0])
    #     if dyaw > 0:
    #         # tello.rotate_clockwise(dyaw)
    #         tello_rotate_clockwise_from_NED(tello, dyaw)
    #     if dyaw < 0:
    #         # tello.rotate_counter_clockwise(abs(dyaw))
    #         tello_rotate_counter_clockwise_from_NED(tello, dyaw)
    # time.sleep(1)

    recorder.start()

    tello_go_xyz_speed_from_NED(tello, int(dx), int(dy), int(dz), speed)

    recorder.join()

    print('Total time = {}. Finishing...............'.format(time.time() - initial_time))
    time.sleep(0.2)
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
