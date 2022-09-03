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
    avg = (avg[0] / count, avg[1] / count, avg[2] / count)
    # end = time.time()
    # print('elapsed time ' + str(end - start))
    return avg


planned = list()
executed = list()
VO_approx = list()
cur_pad = list()

delta_lookahead = 40

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
tello.stream_on()
# take off
tello.takeoff()
tello.go_xyz_speed_mid(x=0, y=0, z=100, speed=20, mid=1)
response = True
data = list()
write_idx = 0

def writer_thread():
    global data
    with open('./data/pose.txt', 'w+') as f:
        while len(data) > write_idx:
            if len(data) > write_idx :
                img = data[write_idx][0]
                x, y, z = data[write_idx][1]
                im = Image.fromarray(img)
                im.save('./data/' + str(write_idx) + '.png')
                f.write("%f %f %f\n" % (x, y, z))
        while len(data) <= write_idx:
            continue


def recorder_thread(tello, first_frame, prev_loc):
    global response, data
    #last_rec_after_command_done = False
    #prev_frame = first_frame
    prev_xyz_executed = prev_loc
    next_rec = 0.25
    while (True):
        while response == False:
            state = tello.get_current_state()
            while (state["time"] < state["time"] + next_rec):
                time.sleep(0.05)
                state = tello.get_current_state()
                continue
            cur_frame = tello.get_frame_read()
            #prev_frame = cur_frame
            xyz_executed = get_xyz(tello)
            #xyz_VO = VO(prev_frame, cur_frame)  # 0.25 runtime cost
            #data.append(cur_frame, xyz_executed, xyz_VO)
            data.append([cur_frame, xyz_executed])
            prev_xyz_executed
        while response == True:
            continue

        # if response == True and last_rec_after_command_done == False:
        #    cur_frame = tello.frame2
        #    xyz_executed = get_xyz(tello)
        #    xyz_VO = VO(prev_frame, cur_frame) # 0.25 runtime cost
        #    data.append(cur_frame, xyz_executed,  xyz_VO)
        #    last_rec_after_command_done = True


first_frame = tello.get_frame_read()
first_xyz = get_xyz(tello)
recorder = threading.Thread(target=recorder_thread, args=(tello, first_frame,))
response = False
recorder.start()

# prepare for start
# calculate carrot chasing next move
from sympy import Eq, Symbol, solve

while True:
    cur_x, cur_y, cur_z = executed[-1]
    cur_y_pad = cur_y - 100 * int(cur_pad[-1] in [2, 5]) + 100 * int(cur_pad[-1] in [4, 7, 8])
    tan_alpha = abs(cur_x + delta_lookahead) / abs(cur_y_pad)
    y = Symbol('y')
    eqn = Eq((tan_alpha * y) ** 2 + y ** 2, 20)
    res = solve(eqn)
    y_move = res[1] if (cur_y < 0 and cur_pad[-1] in [1, 3, 6]) or (cur_pad[-1] in [2, 5]) else res[0]
    x_move = math.sqrt(20 - y_move ^ 2)
    response = False
    response = tello.go_xyz_speed(x=x_move, y=y_move, z=100)

tello.land()
tello.end()

'''
# carrot chasing explicit cases - last to correct + correct sign of y_move
    if cur_pad[-1] in [1,3,6] :
        cur_x, cur_y, cur_z = executed[-1]
        tan_alpha = abs(cur_x + delta_lookahead) / abs(cur_y)
        y = Symbol('y')
        eqn = Eq((tan_alpha*y)**2 + y**2, 20)
        res = solve(eqn)
        y_move = res[1] if cur_y < 0 else res[0]
        x_move = math.sqrt(20 - y_move ^ 2)
    elif cur_pad[-1] in [2,5,8] :
        cur_x, cur_y, cur_z = executed[-1]
        tan_alpha = abs(cur_x + delta_lookahead) / abs(cur_y - 100)
        y = Symbol('y')
        eqn = Eq((tan_alpha * y) ** 2 + y ** 2, 20)
        res = solve(eqn)
        y_move = res[1]
        x_move = math.sqrt(20 - y_move ^ 2)
    else:
        cur_x, cur_y, cur_z = executed[-1]
        tan_alpha = abs(cur_x + delta_lookahead) / abs(cur_y + 100)
        x = Symbol('x')
        eqn = Eq(x ** 2 + (tan_alpha * x) ** 2, 20)
        res = solve(eqn)
        x_move = res[1]
        y_move = -math.sqrt(20 - y_move ^ 2)

'''