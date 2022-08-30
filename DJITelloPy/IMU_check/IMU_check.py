from djitellopy import Tello
import time
import matplotlib.pyplot as plt


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
x_y_z_h = []
for i in range(1000):
    time.sleep(0.1)
    state = tello.get_current_state()
    if state["mid"] < 0:
        continue
    # detect and react to pads until we see pad #1
    x_y_z_h.append((state["x"], state["y"], state["z"], state["h"]))
time.sleep(1)
tello.disable_mission_pads()
tello.end()
if len(x_y_z_h) < 1:
    exit(-1)
xavg = sum(xyzh[0] for xyzh in x_y_z_h)/len(x_y_z_h)
yavg = sum(xyzh[1] for xyzh in x_y_z_h)/len(x_y_z_h)
zavg = sum(xyzh[2] for xyzh in x_y_z_h)/len(x_y_z_h)
havg = sum(xyzh[3] for xyzh in x_y_z_h)/len(x_y_z_h)
print(len(x_y_z_h))

x_axis = list(range(1, len(x_y_z_h)+1))
y_axis = [xyzh[0] for xyzh in x_y_z_h]
plt.plot(x_axis, y_axis)
plt.xlabel('experiment #')
plt.ylabel('x-dist')
plt.title("x_dist_from_pad")
plt.show()

x_axis = list(range(1, len(x_y_z_h)+1))
y_axis = [xyzh[1] for xyzh in x_y_z_h]
plt.plot(x_axis, y_axis)
plt.xlabel('experiment #')
plt.ylabel('y-dist')
plt.title("y_dist_from_pad")
plt.show()

x_axis = list(range(1, len(x_y_z_h)+1))
y_axis = [xyzh[2] for xyzh in x_y_z_h]
plt.plot(x_axis, y_axis)
plt.xlabel('experiment #')
plt.ylabel('z-dist')
plt.title("z_dist_from_pad")
plt.show()

x_axis = list(range(1, len(x_y_z_h)+1))
y_axis = [xyzh[3] for xyzh in x_y_z_h]
plt.plot(x_axis, y_axis)
plt.xlabel('experiment #')
plt.ylabel('height')
plt.title("height(exp)")
plt.show()