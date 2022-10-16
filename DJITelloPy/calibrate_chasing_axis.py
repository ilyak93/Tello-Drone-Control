# connect, enable missions pads detection and show battery
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from tello_with_optitrack.position import connectOptitrack, telloState, calc_initial_SE_motive2telloNED_inv, \
    SE_motive2telloNED
import math
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
x, z, y = SE_motive[0:3, 3] * m_to_cm
cur_x, cur_y = -x, -y


initial_center_y = np.load("center_y_axis.npy")
delta_x = 500
delta_y = cur_y - initial_center_y
slope = -(delta_x / delta_y) if delta_y < 0 else delta_x / delta_y


alpha = 0

initial_rotation_view = np.load("initial_rotation_view.npy")
SE_tello_NED_to_navigate = SE_motive2telloNED(SE_motive, initial_rotation_view)

euler = Rot.from_matrix(SE_tello_NED_to_navigate[0:3, 0:3]).as_euler('zyx', degrees=False)
euler = euler / np.pi * 180.
(pitch, roll, yaw) = np.flip(euler)

if delta_y != 0:
    tan_alpha = delta_x / abs(delta_y)
    alpha_rad = math.atan(tan_alpha)
    alpha_deg = 90 - round(alpha_rad * 180. / math.pi)
    alpha_deg = alpha_deg if delta_y < 0 else -alpha_deg
    alpha = alpha_deg - yaw

np.save("slope_bias_alpha_start_pos.npy", np.array((slope, delta_y, alpha, cur_x, cur_y, z)))
print("turn ur drone to postion of " + str(alpha) + "degrees right (negative is left correspondingly")


