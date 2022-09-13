import cv2
import time
import numpy as np

from djitellopy import Tello
from controller import runWaypoint, control_loop
from position import (connectOptitrack, telloState, setwaypoint, waypointUpdate, calc_initial_SE_motive2telloNED_inv, SE_motive2telloNED)
import os
import csv
from scipy.spatial.transform import Rotation as R

# Refer this link for conversions:
# https://v22.wiki.optitrack.com/index.php?title=Data_Streaming

#### Tello's xyz are in ENU system; We want NED. This affect z & yaw.

SEED = 1
BASE_RENDER_DIR = '/home/crl-user/tello_test/OL_trajs/'
MAX_TIME = 180  # [s]
N_WP = 200  # Number of waypoints
body_id_drone1 = 310  # Drone's ID in Motive

np.random.seed(SEED)

render_dir = os.path.join(BASE_RENDER_DIR, str(SEED))


labels_filename = os.path.join(render_dir, 'pose_file.csv')  # For pose in VO frame
motive_labels_filename = os.path.join(render_dir, 'motive_pose_file.csv')  # For pose in Motive's frame
patch_pose_filename = os.path.join(render_dir, 'patch_motive_pose.csv')
patch_pose_VO_filename = os.path.join(render_dir, 'patch_pose_VO.csv')

#labels_file = open(labels_filename, 'w')
#labels_writer = csv.writer(labels_file)
#motive_labels_file = open(motive_labels_filename, 'w')
#motive_labels_writer = csv.writer(motive_labels_file)
#patch_pose_file = open(patch_pose_filename, 'w')
#patch_pose_writer = csv.writer(patch_pose_file)
#patch_pose_VO_file = open(patch_pose_VO_filename, 'w')
#patch_pose_VO_writer = csv.writer(patch_pose_VO_file)


# Setup connection with optitrack and the waypoints

streamingClient = connectOptitrack(body_id_drone1)	# Get rigid body ID from Motive (4, 3)


# Save the patch pose in KITTI format:
#curr_state = telloState(streamingClient)
#patch_pose_writer.writerow(list(pp[0]) + list(pp[1]) + list(pp[2]))
#patch_pose_file.close()

curr_state = telloState(streamingClient)
SE_motive = curr_state[-1]  # in Y UP system
T_w_b0_inv = calc_initial_SE_motive2telloNED_inv(SE_motive)



try:
    next_waypoint_idx = 1
    initial_time = time.time()
    # Outer loop over Waypoints:
    while (time.time() - initial_time) < MAX_TIME and next_waypoint_idx < N_WP + 1:
        start_time = time.time()
        # Initiate the Position CONTROLLER
        curr_state = telloState(streamingClient)  # output is np.array([id_num, rotated_pos, rotated_quat, euler])
        SE_motive = curr_state[-1]  # in Y UP system
        SE_tello_NED = SE_motive2telloNED(SE_motive, T_w_b0_inv)
        euler = R.from_matrix(SE_tello_NED[0:3, 0:3]).as_euler('zyx', degrees=False)
        euler = np.flip(euler)
        print('pos_NED = {} |||| euler_d_NED = {}'.format(SE_tello_NED[0:3, -1].squeeze(), euler/np.pi*180.))
        e1 = time.time()
        time.sleep(1)
        end_time = time.time()
        dt = end_time - start_time
        d1 = e1 - start_time
        #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% dt, d1 = {}, {}'.format(dt, d1))
	#labels_writer.writerow(data)
        next_waypoint_idx += 1

    print("Total time elapsed. Finishing...............")
    time.sleep(0.5)
    #labels_file.close()
    #motive_labels_file.close()
    #patch_pose_VO_file.close()

except Exception:
    print("Main Controller Loop Crashed...")
    #labels_file.close()
    #motive_labels_file.close()
    #patch_pose_VO_file.close()
    raise

