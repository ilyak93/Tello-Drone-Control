# connect, enable missions pads detection and show battery
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from skspatial.objects import Line, Sphere

from tello_with_optitrack.position import connectOptitrack, telloState, \
    SE_motive2telloNED
import math

body_id_drone1 = 334  # Drone's ID in Motive
body_id_patch = 308  # Patch's ID in Motive

m_to_cm = 100

initial_center_y = np.load("center_y_axis.npy")
# connect to Opti-Track
streamingClient = connectOptitrack(body_id_drone1, body_id_patch)

if not streamingClient:
    print("Optitrack connection error")
    exit(-1)

m_to_cm = 100
opti_state = telloState(streamingClient)
SE_motive = opti_state[-1]
x, _, y = SE_motive[0:3, 3] * m_to_cm
initial_x, initial_y = -x, -y
takeoff_height = 140

target_translation = 700  # target

# (x, y, z, pitch, roll, yaw) : (cm, cm, cm, deg, deg, deg)
target_z_to_choose = 220
target_pos = np.asarray([initial_x + target_translation, initial_center_y,
                         target_z_to_choose])

start_point3D = (initial_y, initial_x, takeoff_height)
target_x, target_y, target_z = target_pos[0:3]

source_to_target_chase_axis = Line.from_points(point_a=start_point3D,
                                               point_b=np.array((target_y,
                                                                 target_x,
                                                                 target_z)))

R = target_translation
sphere = Sphere(start_point3D, R)
point_a, point_b = sphere.intersect_line(source_to_target_chase_axis)

actual_target = point_a if point_a[1] > point_b[1] else point_b

#delta_x = target_translation
delta_x = actual_target[1] - initial_x
#delta_y = initial_y - initial_center_y
delta_y = initial_y - actual_target[0]
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

np.save("alpha_start_pos_target_pos.npy", np.array((alpha, initial_x,
                                                    initial_y, takeoff_height,
                                                    actual_target[1],
                                                    actual_target[0],
                                                    actual_target[2])))
print("turn ur drone to postion of " + str(alpha) + "degrees right (negative is left correspondingly")

