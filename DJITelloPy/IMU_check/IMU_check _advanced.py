from djitellopy import Tello
import time
import matplotlib.pyplot as plt
import numpy as np


# create and connect

tello = Tello()
tello.connect()
tello.enable_mission_pads()
time.sleep(0.1)
tello.set_mission_pad_detection_direction(2)
time.sleep(2)
# configure drone
cur_pad = tello.get_mission_pad_id()
state = tello.get_current_state()
print("pad id is " + str(cur_pad) + " " + str(state["mid"]))

if state["mid"] < 0:
    exit(-1)

x_y_z = []

first = True
for k in range(100):
    start = time.time()
    avg = (0, 0, 0)
    cur_xyz = []
    count = 0
    for i in range(100000):
        state = tello.get_current_state()
        if state["mid"] < 0:
            continue
        count = count + 1
        avg = (avg[0] + state["x"], avg[1] + state["y"], avg[2] + state["z"])
        cur_xyz.append((state["x"], state["y"],  state["z"]))
        # detect and react to pads until we see pad #1
    avg = (avg[0] / count, avg[1] / count, avg[2] / count)
    x_y_z.append(avg)

    if k % 10 == 0:
        end = time.time()
        print('time')
        print(end - start)
        print(avg)
        print(state["bat"])
        first = False

        if len(cur_xyz) < 1:
            exit(-1)
        xavg = sum(xyz[0] for xyz in cur_xyz) / len(cur_xyz)
        yavg = sum(xyz[1] for xyz in cur_xyz) / len(cur_xyz)
        zavg = sum(xyz[2] for xyz in cur_xyz) / len(cur_xyz)
        print(len(x_y_z))

        x_axis = list(range(1, len(cur_xyz) + 1))
        y_axis = [xyz[0] for xyz in cur_xyz]
        plt.plot(x_axis, y_axis)
        plt.xlabel('experiment #')
        plt.ylabel('x-dist')
        plt.title("x_dist_from_pad")
        plt.show()

        x_axis = list(range(1, len(cur_xyz) + 1))
        y_axis = [xyz[1] for xyz in cur_xyz]
        plt.plot(x_axis, y_axis)
        plt.xlabel('experiment #')
        plt.ylabel('y-dist')
        plt.title("y_dist_from_pad")
        plt.show()

        x_axis = list(range(1, len(cur_xyz) + 1))
        y_axis = [xyz[2] for xyz in cur_xyz]
        plt.plot(x_axis, y_axis)
        plt.xlabel('experiment #')
        plt.ylabel('z-dist')
        plt.title("z_dist_from_pad")
        plt.show()


time.sleep(1)

tello.disable_mission_pads()
tello.end()

if len(x_y_z) < 1:
    exit(-1)
xavg = sum(xyz[0] for xyz in x_y_z) / len(x_y_z)
yavg = sum(xyz[1] for xyz in x_y_z) / len(x_y_z)
zavg = sum(xyz[2] for xyz in x_y_z) / len(x_y_z)
print(len(x_y_z))

x_axis = list(range(1, len(x_y_z) + 1))
y_axis = [xyz[0] for xyz in x_y_z]
plt.plot(x_axis, y_axis)
plt.xlabel('experiment #')
plt.ylabel('x-dist')
plt.title("x_dist_from_pad")
plt.show()
x_dist = np.array(y_axis)
np.save("x_dist", x_dist)

x_axis = list(range(1, len(x_y_z) + 1))
y_axis = [xyz[1] for xyz in x_y_z]
plt.plot(x_axis, y_axis)
plt.xlabel('experiment #')
plt.ylabel('y-dist')
plt.title("y_dist_from_pad")
plt.show()
y_dist = np.array(y_axis)
np.save("y_dist", y_dist)

x_axis = list(range(1, len(x_y_z) + 1))
y_axis = [xyz[2] for xyz in x_y_z]
plt.plot(x_axis, y_axis)
plt.xlabel('experiment #')
plt.ylabel('z-dist')
plt.title("z_dist_from_pad")
plt.show()
z_dist = np.array(y_axis)
np.save("z_dist", z_dist)
