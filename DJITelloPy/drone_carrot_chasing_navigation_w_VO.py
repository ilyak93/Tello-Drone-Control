import math

from djitellopy import Tello
import time
import matplotlib.pyplot as plt

#function for gettig the average location during 1 sec of measurements
def get_xyz(tello):
    start = time.time()
    avg, count  = ((0, 0, 0), 0)
    #avg, count, cur_xyz  = ((0, 0, 0), 0, [])
    for i in range(1000):
        state = tello.get_current_state()
        if state["mid"] < 0:
            continue
        count = count + 1
        avg = (avg[0] + state["x"], avg[1] + state["y"], avg[2] + state["z"])
        #cur_xyz.append((state["x"], state["y"],  state["z"]))
        #time.sleep(0.1)
        # detect and react to pads until we see pad #1
    avg = (avg[0] / count, avg[1] / count, avg[2] / count)
    end = time.time()
    print('elapsed time ' + str(end - start))
    return avg

planned = [(0,0,100)]
executed = list()
VO_approx = list()
cur_pad = list()

delta_lookahead = 40

#connect, enable missions pads detection and show battery

tello = Tello()
tello.connect()
tello.enable_mission_pads()
time.sleep(0.1)
tello.set_mission_pad_detection_direction(2)
time.sleep(0.1)

state = tello.get_current_state()
print("battery is " + str(state["bat"]))

'''
response = False
sent = False
def wait_for_command(command, **args):
    global response
    global sent
    while response == False:
        if sent == False:
            if len(args.items()) > 0:
                response = command(**args)
            response = command()
            sent = True
    sent = False
    response = False
'''
#enable video
tello.takeoff()
#first_frame = tello.get_frame()
#first_xyz = get_xyz(tello)
reponse = True
'''
recorder_thread(first_frame, prev_loc):
    global response 
    last_rec_after_command_done = False
    state = tello.state
    prev_frame = first_frame
    prev_xyz_executed = prev_loc
    next_rec = 0.25
    while (True):
        while(response == False):
            while(state.time < next_rec):
                continue
            cur_frame = tello.frame2
            xyz_executed = get_xyz(tello)
            xyz_VO = VO(prev_frame, cur_frame) # 0.25 runtime cost
            data.append(cur_frame, xyz_executed,  xyz_VO)
            next_rec = next_rec + 0.25
        
        #if response == True and last_rec_after_command_done == False:
        #    cur_frame = tello.frame2
        #    xyz_executed = get_xyz(tello)
        #    xyz_VO = VO(prev_frame, cur_frame) # 0.25 runtime cost
        #    data.append(cur_frame, xyz_executed,  xyz_VO)
        #    last_rec_after_command_done = True
              
        while response == True:
            continue           
'''
# thread(recorder_thread).start()
Response = False
response = tello.go_xyz_speed_mid(x=0, y=0, z=100, speed=20, mid=1)
response = True
#prepare for start
# calculate carrot chasing next move
from sympy import Eq, Symbol, solve
while True:
    cur_x, cur_y, cur_z = executed[-1]
    cur_y_pad = cur_y - 100 * int(cur_pad[-1] in [2,5]) + 100 * int(cur_pad[-1] in [4,7,8])
    tan_alpha = abs(cur_x + delta_lookahead) / abs(cur_y_pad)
    y = Symbol('y')
    eqn = Eq((tan_alpha * y) ** 2 + y ** 2, 20)
    res = solve(eqn)
    y_move = res[1] if (cur_y < 0 and cur_pad[-1] in [1,3,6]) or (cur_pad[-1] in [2,5]) else res[0]
    x_move = math.sqrt(20 - y_move ^ 2)
    response=False
    response = tello.go_xyz_speed(x=x_move, y=y_move, z=cur_z)
    
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