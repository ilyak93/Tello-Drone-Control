from djitellopy import Tello
import time, cv2
from threading import Thread

# create and connect

tello = Tello()
tello.connect()


keepRecording = True
tello.streamon()
frame_read = tello.get_frame_read()
states = list()

def videoRecorder():
    count = 0
    # create a VideoWrite object, recoring to ./video.avi
    height, width, _ = frame_read.frame.shape
    video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    while keepRecording:
        video.write(frame_read.frame)
        states.append(frame_read.state)
        time.sleep(1 / 30)
        count = count + 1
    print(count)
    video.release()

# we need to run the recorder in a seperate thread, otherwise blocking options
#  would prevent frames from getting added to the video
recorder = Thread(target=videoRecorder)
recorder.start()

# configure drone

tello.enable_mission_pads()
tello.set_mission_pad_detection_direction(2)  # forward detection only
tello.takeoff()
#tello.move_down(60)

cur_pad = tello.get_mission_pad_id()
next_pad = cur_pad
# detect and react to pads until we see pad #1


tello.move_forward(20)
cur_pad = tello.get_mission_pad_id()
print("pad is " + str(cur_pad))
cur_dist = tello.get_mission_pad_distance_x()
print("and the distance to it is " + str(cur_dist))

# graceful termination

tello.disable_mission_pads()
tello.land()

keepRecording = False
recorder.join()

tello.end()

print(len(states))
