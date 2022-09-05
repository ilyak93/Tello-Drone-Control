import math
import threading
from PIL import Image
from djitellopy import Tello
import time
import matplotlib.pyplot as plt


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
tello.go_xyz_speed_mid(x=0, y=0, z=80, speed=20, mid=1)
data = list()
write_idx = 0

#reponse is True as the last command of taking off with alignment using go mid finished
response = True

def writer_thread():
    global data, write_idx
    with open('data/pose.txt', 'w+') as f:
        while len(data) > write_idx:
                img = data[write_idx][0]
                x, y, z = data[write_idx][1]
                pad = data[write_idx][2]
                im = Image.fromarray(img)
                im.save('./data/' + str(write_idx) + '.png')
                f.write("%f %f %f pad=%d\n" % (x, y, z, pad))
                write_idx = write_idx + 1


#last is False as last recording which is the first in this case have not done yet
last = False

response = threading.Event()
ready = threading.Event()
def recorder_thread(tello, reader):
    global response, data, ready
    while True:
        ready.set()
        response.wait()
        if data[-1][2] == -1:
            break
        state = tello.get_current_state()
        data.append([reader.frame, (state['x'], state['y'], state['z']), state['mid']])
        response.clear()

# start recorder and writer threads
reader = tello.get_frame_read()
recorder = threading.Thread(target=recorder_thread, args=([tello, reader]))
recorder.start()

distance_btw_pads = 100
R = 25
delta_lookahead = 50
# calculate carrot chasing moves and send to execution
# get first frame and its xyz label

state = tello.get_current_state()
data.append([reader.frame, (state['x'], state['y'], state['z']), state['mid']])

while True:
    # this calculatins takes 0.0 seconds
    #start = time.time()
    #print("loc = " + str(executed[-1]))
    (cur_x, cur_y, cur_z), cur_pad = data[-1][1], data[-1][2]
    x_move, y_move = R, 0
    if cur_y != 0:
        cur_y__dist_from_pad = cur_y + distance_btw_pads * int(cur_pad in [2, 5]) - \
                    distance_btw_pads * int(cur_pad in [4, 7, 8])
        tan_alpha = abs(cur_x + delta_lookahead) / abs(cur_y__dist_from_pad)
        # (tan_alpha+1)*y**2 = R**2 --> y = math.sqrt(R**2 / (tan_alpha+1))
        y_move_abs = math.sqrt(R**2 / (tan_alpha+1))
        y_move = float(y_move_abs) if (cur_y < 0 and cur_pad in [1, 3, 6]) or \
                                  (cur_pad in [4, 7, 8]) else float(-y_move_abs)
        x_move = math.sqrt(R ** 2 - y_move ** 2)
        #print("xmove and ymove are: " + str(x_move) + ',' + str(y_move))
        if abs(x_move) < 20.0 and abs(y_move) < 20.0:
            if abs(x_move) > abs(y_move):
                x_move = math.copysign(20.0, x_move)
            else:
                y_move = math.copysign(20.0, y_move)
    #end = time.time()
    #print("time is" + str(end - start))
    ready.wait()
    if data[-1][2] == -1:
        response.set()
        break
    tello.go_xyz_speed(x=round(x_move), y=round(y_move), z=0, speed=20)
    time.sleep(3)
    ready.clear()
    response.set()




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

