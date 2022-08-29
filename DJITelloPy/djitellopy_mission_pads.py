from djitellopy import Tello

# create and connect

tello = Tello()
tello.connect()

# configure drone

tello.enable_mission_pads()
tello.set_mission_pad_detection_direction(2)  # forward detection only

tello.takeoff()
#tello.move_down(60)

cur_pad = tello.get_mission_pad_id()
next_pad = cur_pad
# detect and react to pads until we see pad #1


while next_pad == cur_pad:
    tello.move_forward(20)
    next_pad = tello.get_mission_pad_id()



# graceful termination

tello.disable_mission_pads()
tello.land()
tello.end()