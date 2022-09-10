import math
import threading
from PIL import Image
from djitellopy import Tello
import time
import matplotlib.pyplot as plt


# function for gettig the average location during 1 sec of measurements
def get_xyz(tello):
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
        return (0,0,0)
    avg = (avg[0] / count, avg[1] / count, avg[2] / count)
    # end = time.time()
    # print('elapsed time ' + str(end - start))
    return avg


planned = list()
executed = list()
VO_approx = list()
cur_pad = list()

# connect, enable missions pads detection and show battery

tello = Tello()
tello.connect()
tello.enable_mission_pads()
time.sleep(0.1)
tello.set_mission_pad_detection_direction(2)
time.sleep(0.1)

state = tello.get_current_state()
print("battery is " + str(state["bat"]))

tello_on = True
# take off
tello.takeoff()
tello.go_xyz_speed_mid(x=0, y=0, z=120, speed=20, mid=1)
data = list()
write_idx = 0

#reponse is True as the last command of taking off with alignment using go mid finished
response = True

def writer_thread():
    global data, write_idx
    with open('data/pose.txt', 'w+') as f:
        while True:
            while len(data) > write_idx:
                if len(data) > write_idx:
                    img = data[write_idx][0]
                    x, y, z = data[write_idx][1]
                    im = Image.fromarray(img)
                    im.save('./data/' + str(write_idx) + '.png')
                    f.write("%f %f %f\n" % (x, y, z))
                    write_idx = write_idx + 1
                    time.sleep(0.1)
            while len(data) <= write_idx:
                time.sleep(0.1)
                if tello_on is False:
                    break
            if tello_on is False:
                break

#last is False as last recording which is the first in this case have not done yet
last = False

def recorder_thread(tello, reader):
    global response, data, executed, last, tello_on
    prev_frame = None
    prev_executed = None
    next_rec = 0.25
    while True:
        if response is True:
            cur_frame = reader.frame
            # prev_frame = cur_frame
            xyz_executed = get_xyz(tello)
            # prev_executed = xyz_executed
            executed.append(xyz_executed)
            # xyz_VO = VO(data[-1][0], cur_frame) # 0.25 runtime cost
            # data.append(cur_frame, xyz_executed,  xyz_VO)
            data.append([cur_frame, xyz_executed])
            last = True

        while response is True and tello_on is True:
            time.sleep(0.1)
            continue

        while response is False and tello_on:
            state = tello.get_current_state()
            start_time = state["time"]
            while start_time + next_rec > state["time"]:
                time.sleep(0.05)
                state = tello.get_current_state()
                continue
            cur_frame = reader.frame
            # prev_frame = cur_frame
            xyz_executed = get_xyz(tello)
            executed.append(xyz_executed)
            # xyz_VO = VO(prev_frame, cur_frame)  # 0.25 runtime cost
            # data.append(cur_frame, xyz_executed, xyz_VO)
            data.append([cur_frame, xyz_executed])
            # prev_xyz_executed = xyz_executed
            # prev_frame = cur_frame
        if tello_on is False:
            break


# enable video
tello.streamon()
time.sleep(1)
# get first frame and its xyz label
reader = tello.get_frame_read()
# start recorder and writer threads
recorder = threading.Thread(target=recorder_thread, args=(tello, reader))
writer = threading.Thread(target=writer_thread, args=())
recorder.start()
writer.start()

# prepare for start
# calculate carrot chasing next move
from sympy import Eq, Symbol, solve

distance_btw_pads = 100
R = 25
delta_lookahead = 50

while True:
    while last is False:
        time.sleep(0.1)
        continue
    state = tello.get_current_state()
    cur_pad.append(state['mid'])
    if cur_pad[-1] == -1:
        tello_on = False
        break
    print("loc = " + str(executed[-1]))
    cur_x, cur_y, cur_z = executed[-1]
    x_move, y_move = R, 0
    if cur_y != 0:
        cur_y_pad = cur_y + distance_btw_pads * int(cur_pad[-1] in [2, 5]) - \
                    distance_btw_pads * int(cur_pad[-1] in [4, 7, 8])
        tan_alpha = abs(cur_x + delta_lookahead) / abs(cur_y_pad)
        y = Symbol('y')
        eqn = Eq((tan_alpha * y) ** 2 + y ** 2, R ** 2)
        res = solve(eqn)
        y_move = float(res[1]) if (cur_y < 0 and cur_pad[-1] in [1, 3, 6]) or \
                                  (cur_pad[-1] in [4, 7, 8]) else float(res[0])
        x_move = math.sqrt(R ** 2 - y_move ** 2)
        print("xmove and ymove are: " + str(x_move) + ',' + str(y_move))
        if abs(x_move) < 20.0 and abs(y_move) < 20.0:
            if abs(x_move) > abs(y_move):
                x_move = math.copysign(20.0, x_move)
            else:
                y_move = math.copysign(20.0, y_move)
    last = False
    response = False
    tello.go_xyz_speed(x=round(x_move), y=round(y_move), z=0, speed=20)
    response = True

tello.land()
tello.end()

writer.join()
recorder.join()

# signals on response and signals on executed :
# carrot chasing should sleep_wait until gets a signal from recorder
# that it recorded the last True executed command
# recorder should sleep_wait while command yet sent to tello drone

'''
# carrot chasing explicit cases - last to correct + correct sign of y_move
    if cur_pad[-1] in [1,3,6] :
        cur_x, cur_y, cur_z = executed[-1]
        tan_alpha = abs(cur_x + delta_lookahead) / abs(cur_y)
        y = Symbol('y')
        eqn = Eq((tan_alpha*y)**2 + y**2, 20)
        res = solve(eqn)
        y_move = res[1] if cur_y < 0 else res[0]
        x_move = math.sqrt(20 - y_move ** 2)
    elif cur_pad[-1] in [2,5,8] :
        cur_x, cur_y, cur_z = executed[-1]
        tan_alpha = abs(cur_x + delta_lookahead) / abs(cur_y - 100)
        y = Symbol('y')
        eqn = Eq((tan_alpha * y) ** 2 + y ** 2, 20)
        res = solve(eqn)
        y_move = res[1]
        x_move = math.sqrt(20 - y_move ** 2)
    else:
        cur_x, cur_y, cur_z = executed[-1]
        tan_alpha = abs(cur_x + delta_lookahead) / abs(cur_y + 100)
        x = Symbol('x')
        eqn = Eq(x ** 2 + (tan_alpha * x) ** 2, 20)
        res = solve(eqn)
        x_move = res[1]
        y_move = -math.sqrt(20 - y_move ** 2)

'''
