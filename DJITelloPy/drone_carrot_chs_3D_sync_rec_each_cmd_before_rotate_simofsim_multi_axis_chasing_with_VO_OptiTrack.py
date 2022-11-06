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
from skspatial.objects import Line, Sphere
from scipy.spatial import distance
from sklearn.preprocessing import normalize

m_to_cm = 100
SEED = 54
BASE_RENDER_DIR = '/home/vista/ilya_tello_test/OL_trajs_images/'
dt_cmd = 3.
cam_calib_fname = 'tello_960_720_calib_djitellopy.p'
initial_opti_y = np.load("center_y_axis.npy") * m_to_cm
# initial_rotation_view = np.load("carrot_chasing_rotation_view.npy")
first_alpha_loaded, x, y, z, target_x, target_y, target_z, x_stop, \
    = np.load("alpha_start_pos_target_pos_x_stop.npy")

delta_lookahead = 100
R = 25

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
time.sleep(6)
start_z = 140
start_point3D = (y, x, start_z)
tello.go_xyz_speed_mid(x=0, y=0, z=start_z, speed=50, mid=1)
time.sleep(4)

tello.disable_mission_pads()
time.sleep(0.1)

data = list()

reader = tello.get_frame_read()

curr_state = telloState(streamingClient)
SE_motive = curr_state[-1]  # in Y UP system

initial_x, initial_z, initial_y = SE_motive[0:3, 3] * m_to_cm
initial_x_before, initial_y_before = -initial_x, -initial_y

# (x, y, z, pitch, roll, yaw) : (cm, cm, cm, deg, deg, deg)

target_pos = np.asarray([target_x, target_y, target_z, 0, 0, 0])

first_alpha = first_alpha_loaded
# first_y_chase = compute_y_of_chased_axis(25*first_alpha)
# first_carrot_chase_point = (25 * math.sin(first_alpha), first_y_chase) #sanity check 25 * cos(alpha) == first_y_chase
target_x, target_y, target_z = target_pos[0:3]

line3D = Line.from_points(point_a=start_point3D,
                          point_b=np.array((target_y, target_x, target_z)))

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
time.sleep(2)

# TODO: add visualizations for 2D and 3D
# TODO: write patch detected into name of image
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
            im.save(render_dir + '/' + str(write_idx) + '.png')

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
# TODO: check if last frame recorded is in the same place as previous (and ignore/delete it if it indeed)
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

        #patch_detectd = ad.are_4_markers_detected(data[-1][0])
        #print("Patch detected: " + str(patch_detectd))

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

        print("dist from target " + str(distance.euclidean(cur_poz[:2], target_pos[:2])))
        if distance.euclidean(cur_poz[:2], target_pos[:2]) <= target_radius or \
                cur_poz[0] > x_stop:
            ready.set()
            break

        response.clear()
        ready.set()


# start recorder and writer threads

recorder = threading.Thread(target=recorder_thread, args=([reader]))
recorder.start()

# calculate carrot chasing moves and send to execution
# get first frame and its xyz label

while True:
    # this calculatins takes 0.0 seconds
    # start = time.time()
    # print("loc = " + str(executed[-1]))
    (cur_x, cur_y, cur_z, _, _, prev_yw) = data[-1][-1]
    cur_poz = (cur_x, cur_y, cur_z)

    if distance.euclidean(cur_poz[:2], target_pos[:2]) <= target_radius or \
            cur_poz[0] > x_stop:
        response.set()
        break

    point3D = np.array([cur_y, cur_x, cur_z])
    projected_point3D = np.array(line3D.project_point(point3D))

    tmp_line = Line.from_points(projected_point3D,
                                np.array([target_y, target_x, target_z]))
    dir_norm = math.sqrt(tmp_line.direction[0] ** 2 +
                         tmp_line.direction[1] ** 2 +
                         tmp_line.direction[2] ** 2)
    xyz_lookahead = tmp_line.to_point(delta_lookahead / dir_norm)
    cur_line = Line.from_points(point3D, xyz_lookahead)
    sphere = Sphere(point3D, R)
    point_a, point_b = sphere.intersect_line(cur_line)
    xyz_move = point_b if point_b[1] > point_a[1] else point_a
    x_move, y_move, z_move = (xyz_move[1] - cur_x, xyz_move[0] - cur_y,
                              xyz_move[2] - cur_z)
    print("move before test is " + str((x_move, -y_move, z_move)))
    if abs(x_move) < 20.0 and abs(y_move) < 20.0 and abs(z_move) < 20.0:
        if abs(y_move) > abs(z_move):
            y_move = math.copysign(20.0, y_move)
        else:
            z_move = math.copysign(20.0, z_move)
    # end = time.time()
    # print("time is" + str(end - start))
    # planned.append(round(alfa_deg))  # TODO: x,y planned can be calculated and written for viz
    tan_alpha = round(y_move) / round(x_move)
    alpha_rad = math.atan(tan_alpha)
    alpha_deg = round(alpha_rad * 180. / math.pi)
    #alpha_deg = first_alpha + alpha_deg \
    #    if point3D[0] < projected_point3D[0] else first_alpha - alpha_deg
    cur_rotation = int(round(alpha_deg))

    ready.wait()
    tello.go_xyz_speed(x=int(round(x_move)), y=-int(round(y_move)),
                       z=int(round(z_move)), speed=20)
    time.sleep(3)
    tello.rotate_clockwise(cur_rotation)
    time.sleep(1)

    ready.clear()
    response.set()
    ready.wait()

    patch_detected = ad.are_4_markers_detected(data[-1][0])
    print("Patch detected: " + str(patch_detected))

    curr_state = telloState(streamingClient)
    SE_motive = curr_state[-1]  # in Y UP system
    SE_tello_NED_to_navigate = SE_motive2telloNED(SE_motive, initial_rotation_view)
    euler = Rot.from_matrix(SE_tello_NED_to_navigate[0:3, 0:3]).as_euler('zyx', degrees=False)
    euler = euler / np.pi * 180.
    (roll, pitch, yaw) = np.flip(euler)
    tello.rotate_counter_clockwise(int(round(yaw)))
    time.sleep(1)
    stat = tello.get_current_state()
    print("battery is " + str(stat["bat"]))

tello.land()

state = tello.get_current_state()
print("finish battery is " + str(state["bat"]))

tello.end()
recorder.join()

writer = threading.Thread(target=writer_thread, args=())
writer.start()
writer.join()

# signals on response and signals on executed :
# carrot chasing should sleep_wait until gets a signal from recorder
# that it recorded the last True executed command
# recorder should sleep_wait while command yet sent to tello drone

