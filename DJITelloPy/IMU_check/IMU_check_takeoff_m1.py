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


start = time.time()
avg = (0, 0, 0)
cur_xyz = []
count = 0
state = tello.get_current_state()
print("battery is " + str(state["bat"]))

tello.takeoff()
time.sleep(7)
#tello.move_down(50)
#time.sleep(7)

for i in range(100000):
    state = tello.get_current_state()
    if state["mid"] < 0:
        continue
    count = count + 1
    avg = (avg[0] + state["x"], avg[1] + state["y"], avg[2] + state["z"])
    cur_xyz.append((state["x"], state["y"],  state["z"]))
    #time.sleep(0.1)
    # detect and react to pads until we see pad #1
avg = (avg[0] / count, avg[1] / count, avg[2] / count)

end = time.time()
print('elapsed time ' + str(end - start))
print('average position is ' + str(avg))
print("recognized exps = " + str(len(cur_xyz)))
print("battery is " + str(state["bat"]))

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


tello.go_xyz_speed_mid(0, 0, 120, 20, 1)
time.sleep(10)

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
    #time.sleep(0.1)
    # detect and react to pads until we see pad #1
avg = (avg[0] / count, avg[1] / count, avg[2] / count)

print(avg)
print("recognized exps = " + str(len(cur_xyz)))
print(state["bat"])

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


tello.land()
tello.disable_mission_pads()
tello.end()
