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
from scipy.spatial.transform import Rotation as Rot
from TartanVO.Datasets.utils import Compose, CropCenter, DownscaleFlow, ToTensor, make_intrinsics_layer
from TartanVO.TartanVO import TartanVO
from tello_with_optitrack.aruco_detect import ArucoDetector
from tello_with_optitrack.position import connectOptitrack, telloState, calc_initial_SE_motive2telloNED_inv, \
    SE_motive2telloNED, patchState

m_to_cm = 100
SEED = 54
BASE_RENDER_DIR = '/home/vista/ilya_tello_test/OL_trajs_images/'
dt_cmd = 3.
cam_calib_fname = 'tello_960_720_calib_djitellopy.p'
initial_opti_y = np.load("initial_y_translation_axis.npy") * m_to_cm
initial_rotation_view = np.load("initial_rotation_view.npy")
delta_lookahead = 100
R = 25

np.random.seed(SEED)
render_dir = os.path.join(BASE_RENDER_DIR, str(SEED))

if not os.path.exists(render_dir):
    os.makedirs(render_dir)

labels_filename = os.path.join(render_dir, 'pose_file.csv')  # For pose in VO frame
patch_pose_VO_filename = os.path.join(render_dir, 'patch_pose_VO.csv')

# TODO: make carrot chasing relative to chosen z of the target

# TODO: add carrot chasing with z-axis

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

# first run the script to get initial_rotation_view
# which should be aligned to the target (directly in front of it)
initial_rotation_view = np.load("initial_rotation_view.npy")

# how to run the VO on VO:
# res = {'img1': img1, 'img2': img2 }
# h, w, _ = img1.shape
# intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
# res['intrinsic'] = intrinsicLayer
# res = transform(res)
# res['motion'] = groundTruth

# connect, enable missions pads detection and show battery
body_id_drone1 = 333  # Drone's ID in Motive
body_id_patch = 308  # Patch's ID in Motive

# connect to Opti-Track
streamingClient = connectOptitrack(body_id_drone1, body_id_patch)

if not streamingClient:
    print("Optitrack connection error")
    exit(-1)

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
time.sleep(3)
tello.go_xyz_speed_mid(x=0, y=0, z=180, speed=20, mid=1)
time.sleep(5)

tello.disable_mission_pads()
time.sleep(0.1)

data = list()

reader = tello.get_frame_read()

curr_state = telloState(streamingClient)
SE_motive = curr_state[-1]  # in Y UP system

initial_x, initial_z, initial_y = SE_motive[0:3, 3] * m_to_cm
initial_x_before, initial_y_before = -initial_x, -initial_y

target_translation = 500  # target

# (x, y, z, pitch, roll, yaw) : (cm, cm, cm, deg, deg, deg)
target_pos = np.asarray([initial_x_before + target_translation, initial_opti_y, initial_z, 0, 0, 0])
# TODO replace 0,0,0 with actual angles next
# TODO align to the initial rotation and not to (0,0,0)

SE_tello_NED_to_navigate = SE_motive2telloNED(SE_motive, initial_rotation_view)

euler = Rot.from_matrix(SE_tello_NED_to_navigate[0:3, 0:3]).as_euler('zyx', degrees=False)
euler = euler / np.pi * 180.
(pitch, roll, yaw) = np.flip(euler)
prev_yaw = yaw

print("opti y-axis for carrot chasing is " + str(initial_opti_y))
print("initial x,y,z,pitch,roll,yaw are + " + str([initial_x_before, initial_y_before, initial_z, pitch, roll, yaw]))

if initial_y - target_pos[1] != 0:
    tan_alpha = delta_lookahead / abs(initial_y_before - target_pos[1])

    alpha_rad = math.atan(tan_alpha)
    alpha_deg = 90 - round(alpha_rad * 180. / math.pi)
    alpha_deg = alpha_deg if initial_y_before - target_pos[1] < 0 else -alpha_deg

    cur_rotoation = alpha_deg - int(round(prev_yaw))
    print("cur angle and prev angle are:" + str([alpha_deg, int(round(prev_yaw))]))

    tello.rotate_clockwise(cur_rotoation)
    time.sleep(3)

alfa_deg = alpha_deg  # TODO: correct later
cur_frame = reader.frame
curr_state = telloState(streamingClient)
patch_state = patchState(streamingClient)

SE_motive = curr_state[-1]  # in Y UP system
patch_SE_motive = patch_state[-1]  # in Y UP system

T_w_b0_inv = calc_initial_SE_motive2telloNED_inv(SE_motive)
SE_tello_NED = np.array(SE_motive2telloNED(SE_motive, T_w_b0_inv))
SE_patch_NED = SE_motive2telloNED(patch_SE_motive, T_w_b0_inv)
SE_tello_NED_to_navigate = SE_motive2telloNED(SE_motive, initial_rotation_view)

euler = Rot.from_matrix(SE_tello_NED_to_navigate[0:3, 0:3]).as_euler('zyx', degrees=False)
euler = euler / np.pi * 180.
(roll, pitch, yaw) = np.flip(euler)
initial_x, initial_z, initial_y = SE_motive[0:3, 3] * m_to_cm
initial_x, initial_y = -initial_x, -initial_y

cur_p = initial_x, initial_y, initial_z

print("initial x,y,z,pitch,roll,yaw after rotate are + " + str([initial_x,
                                                                initial_y,
                                                                initial_z,
                                                                pitch,
                                                                roll,
                                                                yaw]))

print("dist from target " + str(math.sqrt(sum((cur_p[:2] - target_pos[:2]) ** 2))))

data.append([cur_frame, SE_tello_NED, SE_patch_NED,
             np.array([initial_x, initial_y, initial_z, pitch, roll, yaw])])

print("target_pos pose " + str(target_pos))

patch_detected = ad.are_4_markers_detected(data[-1][0])
print("Patch detected: " + str(patch_detected))

write_idx = 0
planned = list()

curr_state = telloState(streamingClient)
SE_motive = curr_state[-1]  # in Y UP system
SE_tello_NED_to_navigate = SE_motive2telloNED(SE_motive, initial_rotation_view)
euler = Rot.from_matrix(SE_tello_NED_to_navigate[0:3, 0:3]).as_euler('zyx', degrees=False)
euler = euler / np.pi * 180.
(roll, pitch, yaw) = np.flip(euler)
tello.rotate_counter_clockwise(int(round(yaw)))
time.sleep(3)


# TODO add writing of planned, VO, add statistics
def writer_thread():
    global data, write_idx, planned
    with open(labels_filename, 'w') as labels_file, \
            open(patch_pose_VO_filename, 'w') as patch_pose_VO_file:
        labels_writer = csv.writer(labels_file)
        while len(data) > write_idx:
            img = data[write_idx][0]
            SE_tello_NED = data[write_idx][1]
            img = img[..., ::-1]
            im = Image.fromarray(img)
            im.save(BASE_RENDER_DIR + str(write_idx) + '.png')

            labels_writer.writerow(list(SE_tello_NED[0]) + list(SE_tello_NED[1]) + list(SE_tello_NED[2]))
            write_idx = write_idx + 1
        patch_pose_VO_writer = csv.writer(patch_pose_VO_file)
        SE_patch_NED = data[0][2]
        patch_pose_VO_writer.writerow(list(SE_patch_NED[0]) + list(SE_patch_NED[1]) + list(SE_patch_NED[2]))

        # gt_file.write("%f %f %f %f %f %d\n" % (x, y, z, pitch, roll, yaw))
        # if write_idx >= 1:
        #    predicted = data[write_idx][2]
        #    pred_file.write("%f %f %f %f %f %f\n"
        #                    % (predicted[0, 0], predicted[0, 1], predicted[0, 2],
        #                       predicted[0, 3], predicted[0, 4], predicted[0, 5]))
        # planned_file.write("%f %f %f\n" % (planned[write_idx][0],
        #                                   planned[write_idx][1],
        #                                   planned[write_idx][2]))


# last is False as last recording which is the first in this case have not done yet
last = False

response = threading.Event()
ready = threading.Event()

target_radius = 30


# TODO: make data a readable dict
def recorder_thread(reader):
    global response, data, ready, focalx, focaly, centerx, centery, transform
    while True:
        ready.set()
        response.wait()

        opti_state = telloState(streamingClient)
        SE_motiv = opti_state[-1]
        SE_telo_NED = SE_motive2telloNED(SE_motiv, T_w_b0_inv)

        SE_tello_NED_to_navigat = SE_motive2telloNED(SE_motiv, initial_rotation_view)
        eulr = Rot.from_matrix(SE_tello_NED_to_navigat[0:3, 0:3]).as_euler('zyx', degrees=False)
        eulr = eulr / np.pi * 180.
        (rol, ptch, yw) = np.flip(eulr)

        x, z, y = opti_state[2][0:3, 3] * m_to_cm

        cur_pose = (-x, -y, z)

        print("x,y,z,pitch,roll,yaw after movement are + " + str([cur_pose[0],
                                                                  cur_pose[1],
                                                                  cur_pose[2],
                                                                  ptch, rol,
                                                                  yw]))

        patch_detectd = ad.are_4_markers_detected(data[-1][0])
        print("Patch detected: " + str(patch_detectd))

        cur_fram = reader.frame
        sample = {'img1': data[-1][0], 'img2': cur_fram}
        h, w, _ = cur_fram.shape
        intrinsicLayer = make_intrinsics_layer(w, h, focalx, focaly, centerx, centery)
        sample['intrinsic'] = intrinsicLayer
        sample = transform(sample)
        sample = unsqueeze_transform(sample)
        VO_motions, VO_flow = testvo.test_batch(sample)
        # data.append([reader.frame, (state['x'], state['y'], state['z'],
        #                            state["pitch"], state["roll"],
        #                            state["yaw"], state['mid']), VO_motions, [x_move, y_move, 0]])
        data.append([cur_fram, SE_telo_NED, VO_motions,
                     [90],  # TODO: 90 or [R,0]  should be recalculated and correctly written
                     np.array([cur_pose[0], cur_pose[1], cur_pose[2],
                               ptch, rol, yw])])

        print("current pos is " + str(cur_pose))

        print("dist from target " + str(math.sqrt(sum((cur_pose[:2] - target_pos[:2]) ** 2))))
        if math.sqrt(sum((cur_pose[:2] - target_pos[:2]) ** 2)) <= target_radius:
            ready.set()
            break

        response.clear()
        ready.set()


# start recorder and writer threads

recorder = threading.Thread(target=recorder_thread, args=([reader]))
recorder.start()

# calculate carrot chasing moves and send to execution
# get first frame and its xyz label

first = True

while True:
    # this calculatins takes 0.0 seconds
    # start = time.time()
    # print("loc = " + str(executed[-1]))
    (cur_x, cur_y, cur_z, _, _, prev_yw) = data[-1][-1]
    cur_poz = (cur_x, cur_y, cur_z)

    if math.sqrt(sum((cur_poz[:2] - target_pos[:2]) ** 2)) <= target_radius:
        response.set()
        break

    if not first and cur_y - target_pos[1] != 0:
        tan_alfa = delta_lookahead / abs(cur_y - target_pos[1])

        alfa_rad = math.atan(tan_alfa)
        alfa_deg = 90 - round(alfa_rad * 180. / math.pi)
        alfa_deg = alfa_deg if cur_poz[1] - target_pos[1] < 0 else -alfa_deg

        cur_rotation = alfa_deg - int(round(prev_yw))
        print("cur angle and prev angle are:" + str([alfa_deg, int(round(prev_yw))]))

    if cur_y - target_pos[1] != 0:
        tan_alpha = delta_lookahead / abs(cur_y - target_pos[1])
        # x^2 + y^2 = R^2 ; tan(alpha) = delta_lookahead / y_deviation
        # (tan_alpha+1)*y**2 = R**2 --> y = math.sqrt(R**2 / (tan_alpha+1))
        y_move_abs = math.sqrt(R ** 2 / (tan_alpha + 1))
        y_move = float(y_move_abs) if cur_y - target_pos[1] > 0 else float(-y_move_abs)
        x_move = math.sqrt(R ** 2 - y_move ** 2)
        # print("xmove and ymove are: " + str(x_move) + ',' + str(y_move))
        if abs(x_move) < 20.0 and abs(y_move) < 20.0:
            if abs(x_move) > abs(y_move):
                x_move = math.copysign(20.0, x_move)
            else:
                y_move = math.copysign(20.0, y_move)
    # end = time.time()
    # print("time is" + str(end - start))
    planned.append(round(alfa_deg))  # TODO: x,y planned can be calculated and written for viz

    ready.wait()

    tello.go_xyz_speed(x=int(round(x_move)), y=int(round(y_move)), z=0, speed=50)
    time.sleep(3)

    if first:
        tello.rotate_clockwise(int(round(prev_yw)))
        time.sleep(3)
    else:
        tello.rotate_clockwise(cur_rotation)
        time.sleep(3)

    ready.clear()
    response.set()
    ready.wait()
    first = False
    patch_detected = ad.are_4_markers_detected(data[-1][0])
    print("Patch detected: " + str(patch_detected))

    curr_state = telloState(streamingClient)
    SE_motive = curr_state[-1]  # in Y UP system
    SE_tello_NED_to_navigate = SE_motive2telloNED(SE_motive, initial_rotation_view)
    euler = Rot.from_matrix(SE_tello_NED_to_navigate[0:3, 0:3]).as_euler('zyx', degrees=False)
    euler = euler / np.pi * 180.
    (roll, pitch, yaw) = np.flip(euler)
    tello.rotate_counter_clockwise(int(round(yaw)))
    time.sleep(3)

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

