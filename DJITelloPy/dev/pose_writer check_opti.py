import math
import threading
from PIL import Image
from djitellopy import Tello
import time
import matplotlib.pyplot as plt

from tello_with_optitrack.position import telloState, connectOptitrack, SE_motive2telloNED, \
    calc_initial_SE_motive2telloNED_inv, SE_motive2tello_absNED, SE_motive2tello_absNED_v2, \
    calc_initial_SE_motive2telloNED_inv_v2, SE_motive2telloNED_v2, SE_motive2telloNED_v2

import numpy as np
from scipy.spatial.transform import Rotation as R

# function for gettig the average location during 1 sec of measurements
def get_xyz(tello):
    # start = time.time()
    avg, count = ((0, 0, 0), 0)
    # avg, count, cur_xyz  = ((0, 0, 0), 0, [])
    for i in range(1):
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


# connect, enable missions pads detection and show battery
#tello = Tello()
#tello.connect()
#time.sleep(3)

#state = tello.get_current_state()
#print("battery is " + str(state["bat"]))


write_idx = 0
tello_on = True

body_id_drone1 = 328  # Drone's ID in Motive
body_id_patch = 308  # Patch's ID in Motive

streamingClient = connectOptitrack(body_id_drone1, body_id_patch)


#tello.takeoff()

# v2 is like David explained
time.sleep(5)
opti_state = telloState(streamingClient)
SE_motive = opti_state[-1]
T_w_b0_inv = calc_initial_SE_motive2telloNED_inv(SE_motive)
#T_w_b0_inv_v2 = calc_initial_SE_motive2telloNED_inv_v2(SE_motive)

T_w_b0_abs_inv = np.array([
 [9.89469994e-01, -2.88605423e-03,  1.44709368e-01, -3.18212964e+00],
 [-1.44738065e-01, -1.86757113e-02,  9.89293743e-01, -2.63361071e-01],
 [-1.52605008e-04, -9.99821428e-01, -1.88967779e-02,  1.01621539e-01],
 [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]],
    dtype=np.float64)

'''T_w_b0_abs_inv_v2 = np.array([[ 0.99700721,  0.07682759,  0.00861046,  2.11780143],
 [-0.00982162,  0.0154008,   0.99983316,  0.08217029],
 [0.07668216, -0.99692544,  0.01610928,  1.01257479],
 [0.,          0.,          0.,          1.]],
    dtype=np.float64)'''



def writer_thread():
    global tello_on, write_idx, T_w_b0_inv
    with open('./data/pose_check_opti_check1.txt', 'w+') as f:
        while True:
            opti_state = telloState(streamingClient)
            SE_motive = opti_state[-1]
            SE_tello_NED = SE_motive2telloNED(SE_motive, T_w_b0_inv)
            SE_tello_abs_NED = SE_motive2telloNED(SE_motive, T_w_b0_abs_inv)
            euler = R.from_matrix(SE_tello_NED[0:3, 0:3]).as_euler('zyx', degrees=False)
            euler = euler / np.pi * 180.
            (pitch, roll, yaw) = np.flip(euler)
            my_euler = R.from_matrix(SE_tello_abs_NED[0:3, 0:3]).as_euler('zyx', degrees=False)
            my_euler = my_euler / np.pi * 180.
            (my_pitch, my_roll, my_yaw) = np.flip(my_euler)
            #my_euler = opti_state[1]
            #my_euler = my_euler / np.pi * 180.
            #(my_pitch, my_roll, my_yaw) = np.flip(my_euler)
            f.write("%f %f %f %f %f %f\n" % (pitch, roll, yaw, my_pitch, my_roll, my_yaw))
            write_idx = write_idx + 1
            time.sleep(0.030)
            if tello_on is False:
                break


writer = threading.Thread(target=writer_thread, args=())

print("start")

writer.start()

#tello.flip_forward()
#tello.rotate_clockwise(360)
time.sleep(15)

tello_on = False

writer.join()

