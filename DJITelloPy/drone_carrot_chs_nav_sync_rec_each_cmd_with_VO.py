import math
import threading

import torch
from PIL import Image
from djitellopy import Tello
import time
import matplotlib.pyplot as plt

from TartanVO.Datasets.utils import Compose, CropCenter, DownscaleFlow, ToTensor, make_intrinsics_layer
from TartanVO.TartanVO import TartanVO


# function for gettig the average location during 1 sec of measurements
def get_xyz_pad(tello):
    # start = time.time()
    avg, count = ((0, 0, 0), 0)
    # avg, count, cur_xyz  = ((0, 0, 0), 0, [])
    for i in range(1000):
        state = tello.get_current_state()
        if state["mid"] < 0:
            continue
        count = count + 1
        avg = (avg[0] + state["x"], avg[1] + state["y"], avg[2] + state["z"])
        # cur_xyz.append((state["x"], state["y"],  state["z"]))
        # time.sleep(0.1)
        # detect and react to pads until we see pad #1
    if count == 0:
        return ((0, 0, 0), -1)
    avg = (avg[0] / count, avg[1] / count, avg[2] / count)
    # end = time.time()
    # print('elapsed time ' + str(end - start))
    return (avg, state["mid"])


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


VO_approx = list()

# connect, enable missions pads detection and show battery

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
tello.go_xyz_speed_mid(x=0, y=0, z=100, speed=20, mid=1)
time.sleep(5)
data = list()
lock = threading.Lock()
write_idx = 0
planned = list()
# reponse is True as the last command of taking off with alignment using go mid finished
response = True


def writer_thread():
    global data, write_idx, planned
    with open('data/pose_GT.txt', 'w+') as gt_file,\
            open('data/pose_pred.txt', 'w+') as pred_file, \
                open('data/pose_planned.txt', 'w+') as planned_file:
        while len(data) > write_idx:
            img = data[write_idx][0]
            x, y, z, pitch, roll, yaw, pad = data[write_idx][1]
            im = Image.fromarray(img)
            im.save('./data/' + str(write_idx) + '.png')
            gt_file.write("%f %f %f %f %f %f %d\n" % (x, y, z, pitch, roll, yaw, pad))
            if write_idx >= 1:
                predicted = data[write_idx][2]
                pred_file.write("%f %f %f %f %f %f\n"
                                % (predicted[0, 0], predicted[0, 1], predicted[0, 2],
                                   predicted[0, 3], predicted[0, 4], predicted[0, 5]))
            planned_file.write("%f %f %f\n" % (planned[write_idx][0], planned[write_idx][1], planned[write_idx][2]))
            write_idx = write_idx + 1


# last is False as last recording which is the first in this case have not done yet
last = False

response = threading.Event()
ready = threading.Event()


# TODO: make data a readable dict
def recorder_thread(tello, reader):
    global response, data, ready, focalx, focaly, centerx, centery, transform
    while True:
        ready.set()
        response.wait()
        if data[-1][1][6] == -1:
            break
        state = tello.get_current_state()
        cur_frame = reader.frame
        sample = {'img1': data[-1][0], 'img2': cur_frame}
        h, w, _ = cur_frame.shape
        intrinsicLayer = make_intrinsics_layer(w, h, focalx, focaly, centerx, centery)
        sample['intrinsic'] = intrinsicLayer
        sample = transform(sample)
        sample = unsqueeze_transform(sample)
        VO_motions, VO_flow = testvo.test_batch(sample)
        data.append([reader.frame, (state['x'], state['y'], state['z'],
                                    state["pitch"], state["roll"],
                                    state["yaw"], state['mid']), VO_motions, [x_move, y_move, 0]])
        ready.set()
        response.clear()



# start recorder and writer threads
reader = tello.get_frame_read()
recorder = threading.Thread(target=recorder_thread, args=([tello, reader]))
recorder.start()

distance_btw_pads = 50
R = 25
delta_lookahead = 50
# calculate carrot chasing moves and send to execution
# get first frame and its xyz label

state = tello.get_current_state()
data.append([reader.frame, (state['x'], state['y'], state['z'],
                            state["pitch"], state["roll"],
                            state["yaw"], state['mid'])])

while True:
    # this calculatins takes 0.0 seconds
    # start = time.time()
    # print("loc = " + str(executed[-1]))
    (cur_x, cur_y, cur_z, _, _, _, cur_pad) = data[-1][1]
    x_move, y_move = R, 0
    if cur_y != 0:
        cur_y_dist_from_pad = cur_y + distance_btw_pads * int(cur_pad in [2, 5]) - \
                              distance_btw_pads * int(cur_pad in [3, 6])
        tan_alpha = delta_lookahead / abs(cur_y_dist_from_pad)
        # (tan_alpha+1)*y**2 = R**2 --> y = math.sqrt(R**2 / (tan_alpha+1))
        y_move_abs = math.sqrt(R ** 2 / (tan_alpha + 1))
        y_move = float(y_move_abs) if (cur_y < 0 and cur_pad in [1, 4, 7]) or \
                                      (cur_pad in [3, 6]) else float(-y_move_abs)
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

