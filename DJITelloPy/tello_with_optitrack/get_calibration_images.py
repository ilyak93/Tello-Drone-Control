import cv2
import time
import numpy as np

from djitellopy import Tello
from controller import runWaypoint, control_loop
from position import (connectOptitrack, telloState, setwaypoint, waypointUpdate, calc_initial_SE_motive2telloNED_inv, SE_motive2telloNED, tello_go_xyz_speed_from_NED, tello_rotate_clockwise_from_NED, tello_rotate_counter_clockwise_from_NED, patchState, mean_std_hovering)
from time import sleep
import os

BASE_RENDER_DIR = '/home/crl-user/tello_test/calib_images/'
N_images = 20

render_dir = BASE_RENDER_DIR

if not os.path.exists(render_dir):
    os.makedirs(render_dir)

# Initialise Tello
tello = Tello()

tello.connect()
print("------------------ Battery % = ", tello.get_battery())

tello.streamon()


frame_read = tello.get_frame_read()  # initialize the BG object



idx = 0


initial_time = time.time()

while idx < N_images:
	os.system('play -nq -t alsa synth {} sine {}'.format(1.5,1000))
	sleep(3)
	os.system('play -nq -t alsa synth {} sine {}'.format(0.5,1000))
	sleep(3)
	# input('Waiting for Enter')
	cv2.imwrite(os.path.join(render_dir, str(idx).zfill(4)+'.png'), frame_read.frame)
	os.system('play -nq -t alsa synth {} sine {}'.format(0.2,1000))
	sleep(2)
	idx += 1

tello.streamoff()



