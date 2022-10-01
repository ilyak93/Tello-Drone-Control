# connect, enable missions pads detection and show battery
import numpy as np

from tello_with_optitrack.position import connectOptitrack, telloState, calc_initial_SE_motive2telloNED_inv

body_id_drone1 = 334  # Drone's ID in Motive
body_id_patch = 308  # Patch's ID in Motive

# connect to Opti-Track
streamingClient = connectOptitrack(body_id_drone1, body_id_patch)

if not streamingClient:
    print("Optitrack connection error")
    exit(-1)

m_to_cm = 100

opti_state = telloState(streamingClient)
SE_motive = opti_state[-1]
_, _, y = SE_motive[0:3, 3] * m_to_cm

initial_rotation_view = calc_initial_SE_motive2telloNED_inv(SE_motive)

np.save("initial_rotation_view.npy", initial_rotation_view)
np.save("center_y_axis.npy", -y)

