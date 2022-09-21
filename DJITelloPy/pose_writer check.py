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
print("Start")
time.sleep(3)
tello = Tello()
tello.connect()
tello.enable_mission_pads()
time.sleep(0.1)
tello.set_mission_pad_detection_direction(2)
time.sleep(0.1)

write_idx = 0
tello_on = True


def writer_thread(tello):
    global tello_on, write_idx
    with open('./data/pose_check.txt', 'w+') as f:
        while True:
            state = tello.get_current_state()
            #x, y, z = get_xyz(tello)
            x, y, z = state["x"], state["y"], state["z"]
            f.write("%f %f %f pad=%d\n" % (x, y, z, state["mid"]))
            write_idx = write_idx + 1
            time.sleep(0.75)
            if tello_on is False:
                break


writer = threading.Thread(target=writer_thread, args=([tello]))

writer.start()

time.sleep(15)

tello_on = False

writer.join()
