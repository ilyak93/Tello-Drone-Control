from djitellopy import Tello
import time, cv2
from threading import Thread

# create and connect

tello = Tello()
tello.connect()

# configure drone

tello.enable_mission_pads()
tello.set_mission_pad_detection_direction(2)  # forward detection only
tello.takeoff()
#tello.move_down(30)
tello.move_up(30)
time.sleep(6)

cur_pad = tello.get_mission_pad_id()
next_pad = cur_pad
# detect and react to pads until we see pad #1
print("pad is " + str(cur_pad))
state = tello.get_current_state()
print("and the distance to it is " + str(state))

tello.move_forward(50)

time.sleep(8)

cur_pad = tello.get_mission_pad_id()
print("pad is " + str(cur_pad))
state = tello.get_current_state()
print("and the distance to it is " + str(state))

# graceful termination

tello.disable_mission_pads()
tello.land()
tello.end()


