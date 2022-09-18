import math
import threading
import torch
from PIL import Image
from djitellopy import Tello
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import cv2

from TartanVO.Datasets.utils import Compose, CropCenter, DownscaleFlow, ToTensor, make_intrinsics_layer
from TartanVO.TartanVO import TartanVO
from tello_with_optitrack.aruco_detect import ArucoDetector
from tello_with_optitrack.position import connectOptitrack, telloState, calc_initial_SE_motive2telloNED_inv, \
    SE_motive2telloNED, patchState

SEED = 54
BASE_RENDER_DIR = '/home/vista/ilya_tello_test/OL_trajs_images/'
dt_cmd = 3.
cam_calib_fname = 'tello_960_720_calib_djitellopy.p'

np.random.seed(SEED)
render_dir = os.path.join(BASE_RENDER_DIR, str(SEED))

if not os.path.exists(render_dir):
    os.makedirs(render_dir)

labels_filename = os.path.join(render_dir, 'pose_file.csv')  # For pose in VO frame
patch_pose_VO_filename = os.path.join(render_dir, 'patch_pose_VO.csv')


tello_intrinsics = [
    [785.75708966, 0., 494.5589324],
    [0., 781.95811828, 319.88369613],
    [0., 0., 1.]
]

testvo = TartanVO("tartanvo_1914.pkl")
focalx, focaly, centerx, centery = 785.75708966, 781.95811828, 494.5589324, 319.88369613


class Unsqueeze(object):
    """
    Scale the flow and mask to a fixed size

    """

    def __init__(self, axis=0):
        '''
        size: output frame size, this should be NO LARGER than the input frame size!
        '''
        self.axis = axis

    def __call__(self, sample):
        for key in sample.keys():
            if key != 'motion':
                sample[key] = sample[key].unsqueeze(self.axis).cuda()
            else:
                sample[key] = sample[key].unsqueeze(self.axis)
        return sample


image_width, image_height = 640, 448
transform = Compose([CropCenter((image_height, image_width)), DownscaleFlow(), ToTensor()])
unsqueeze_transform = Unsqueeze()

# how to run the VO on VO:
# res = {'img1': img1, 'img2': img2 }
# h, w, _ = img1.shape
# intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
# res['intrinsic'] = intrinsicLayer
# res = transform(res)
# res['motion'] = groundTruth

# connect, enable missions pads detection and show battery
body_id_drone1 = 328  # Drone's ID in Motive
body_id_patch = 308  # Patch's ID in Motive

# connect to Opti-Track
streamingClient = connectOptitrack(body_id_drone1, body_id_patch)

# Initialize the Aruco detector:
ad = ArucoDetector(calib_filename=cam_calib_fname, aruco_type=cv2.aruco.DICT_4X4_50)

tello = Tello()
tello.connect()
tello.enable_mission_pads()
time.sleep(0.1)
tello.set_mission_pad_detection_direction(2)
time.sleep(0.1)

state = tello.get_current_state()
print("battery is " + str(state["bat"]))

# enable video
tello.streamon()
time.sleep(1)
# take off
tello.takeoff()
tello.go_xyz_speed_mid(x=0, y=0, z=150, speed=100, mid=1)
time.sleep(5)

tello.disable_mission_pads()
time.sleep(0.1)

reader = tello.get_frame_read()

patch_detected = ad.are_4_markers_detected(reader.frame)
print("Patch detected: " + str(patch_detected))

curr_state = telloState(streamingClient)
patch_state = patchState(streamingClient)

SE_motive = curr_state[-1]  # in Y UP system
patch_SE_motive = patch_state[-1]  # in Y UP system
T_w_b0_inv = calc_initial_SE_motive2telloNED_inv(SE_motive)
SE_tello_NED = np.array(SE_motive2telloNED(SE_motive, T_w_b0_inv))
# labels_writer.writerow(list(SE_tello_NED[0]) + list(SE_tello_NED[1]) + list(SE_tello_NED[2]))
SE_patch_NED = SE_motive2telloNED(patch_SE_motive, T_w_b0_inv)
# patch_pose_VO_writer.writerow(list(SE_patch_NED[0]) + list(SE_patch_NED[1]) + list(SE_patch_NED[2]))
#patch_pose_VO_file.close()

data.append([reader.frame, SE_tello_NED, SE_patch_NED])

data = list()

write_idx = 0
planned = list()
# reponse is True as the last command of taking off with alignment using go mid finished
response = True


def writer_thread():
    global data, write_idx, planned
    with open(labels_filename, 'w') as labels_file, \
            open(patch_pose_VO_filename, 'w') as patch_pose_VO_file :
        while len(data) > write_idx:
            img = data[write_idx][0]
            SE_tello_NED = data[write_idx][1]
            SE_patch_NED = data[write_idx][2]
            im = Image.fromarray(img)
            im.save('./data/' + str(write_idx) + '.png')

            labels_writer = csv.writer(labels_file)
            patch_pose_VO_writer = csv.writer(patch_pose_VO_file)

            labels_writer.writerow(list(SE_tello_NED[0]) + list(SE_tello_NED[1]) + list(SE_tello_NED[2]))
            patch_pose_VO_writer.writerow(list(SE_patch_NED[0]) + list(SE_patch_NED[1]) + list(SE_patch_NED[2]))

            #gt_file.write("%f %f %f %f %f %d\n" % (x, y, z, pitch, roll, yaw))
            #if write_idx >= 1:
            #    predicted = data[write_idx][2]
            #    pred_file.write("%f %f %f %f %f %f\n"
            #                    % (predicted[0, 0], predicted[0, 1], predicted[0, 2],
            #                       predicted[0, 3], predicted[0, 4], predicted[0, 5]))
            #planned_file.write("%f %f %f\n" % (planned[write_idx][0],
            #                                   planned[write_idx][1],
            #                                   planned[write_idx][2]))
            #write_idx = write_idx + 1


# last is False as last recording which is the first in this case have not done yet
last = False

response = threading.Event()
ready = threading.Event()


# TODO: make data a readable dict
def recorder_thread(reader):
    global response, data, ready, focalx, focaly, centerx, centery, transform
    while True:
        ready.set()
        response.wait()
        if data[-1][1][6] == -1:
            break
        # state = tello.get_current_state()
        opti_state = telloState(streamingClient)
        SE_motive = opti_state[-1]
        SE_tello_NED = SE_motive2telloNED(SE_motive, T_w_b0_inv)
        #euler = R.from_matrix(SE_tello_NED[0:3, 0:3]).as_euler('zyx', degrees=False)
        #euler = euler / np.pi * 180.
        #(pitch, roll, yaw) = np.flip(euler)

        #x, z, y = opti_state[2][0:3, 3]

        cur_frame = reader.frame
        sample = {'img1': data[-1][0], 'img2': cur_frame}
        h, w, _ = cur_frame.shape
        intrinsicLayer = make_intrinsics_layer(w, h, focalx, focaly, centerx, centery)
        sample['intrinsic'] = intrinsicLayer
        sample = transform(sample)
        sample = unsqueeze_transform(sample)
        VO_motions, VO_flow = testvo.test_batch(sample)
        # data.append([reader.frame, (state['x'], state['y'], state['z'],
        #                            state["pitch"], state["roll"],
        #                            state["yaw"], state['mid']), VO_motions, [x_move, y_move, 0]])
        data.append([reader.frame, SE_tello_NED, SE_patch_NED, VO_motions,
                     [x_move, y_move, 0]])
        ready.set()
        response.clear()


# start recorder and writer threads

recorder = threading.Thread(target=recorder_thread, args=([tello, reader]))
recorder.start()

distance_btw_pads = 50
R = 25
delta_lookahead = 50
# calculate carrot chasing moves and send to execution
# get first frame and its xyz label

while True:
    # this calculatins takes 0.0 seconds
    # start = time.time()
    # print("loc = " + str(executed[-1]))
    (cur_x, cur_y, cur_z) = data[1][0:3, 3]
    x_move, y_move = R, 0
    if cur_y != 0:
        tan_alpha = delta_lookahead / abs(cur_y)
        # (tan_alpha+1)*y**2 = R**2 --> y = math.sqrt(R**2 / (tan_alpha+1))
        y_move_abs = math.sqrt(R ** 2 / (tan_alpha + 1))
        y_move = float(y_move_abs) if cur_y < 0 else float(-y_move_abs)
        x_move = math.sqrt(R ** 2 - y_move ** 2)
        # print("xmove and ymove are: " + str(x_move) + ',' + str(y_move))
        if abs(x_move) < 20.0 and abs(y_move) < 20.0:
            if abs(x_move) > abs(y_move):
                x_move = math.copysign(20.0, x_move)
            else:
                y_move = math.copysign(20.0, y_move)

    # end = time.time()
    # print("time is" + str(end - start))
    planned.append((round(x_move), round(y_move), 0))

    ready.wait()
    if data[-1][1][6] == -1:
        response.set()
        break
    tello.go_xyz_speed(x=round(x_move), y=round(y_move), z=0, speed=100)
    time.sleep(3)
    ready.clear()
    response.set()
    ready.wait()

tello.land()
tello.end()
recorder.join()

writer = threading.Thread(target=writer_thread, args=())
writer.start()
writer.join()

# signals on response and signals on executed :
# carrot chasing should sleep_wait until gets a signal from recorder
# that it recorded the last True executed command
# recorder should sleep_wait while command yet sent to tello drone

