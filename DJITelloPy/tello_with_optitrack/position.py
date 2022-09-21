import time
import numpy as np
import math
from tello_with_optitrack.NatNetClient import NatNetClient
from scipy.spatial.transform import Rotation as R



def connectOptitrack(body_id_drone1, body_id_patch):
	timeout_s = 10
	# This will create a new NatNet client
	streamingClient = NatNetClient(body_id_drone1, body_id_patch)

	# Configure the streaming client to call our rigid body handler on the
	# emulator to send data out.
	streamingClient.newFrameListener = True
	streamingClient.rigidBodyListener = np.zeros((2, 3), dtype=object)  # to occupy ID, pos & quat.

	# Start up the streaming client now that the callbacks are set up.
	# This will run perpetually, and operate on a separate thread.
	streamingClient.run()
	# Slow the OptiTrack threads down?

	# Time to retrieve first state. If no state currently received, the listener
	# just remains the initial value
	start_time = time.time()
	print('Connecting to Opti-Track .....')
	while streamingClient.rigidBodyListener[0, 0] == 0:
		current_time = time.time()
		elapsed_time = current_time - start_time
		if elapsed_time > timeout_s:
			print('Did not receive data from Opti-Track')
			return False

	print('Opti-Track connected')
	return streamingClient


import math

def euler_from_quaternion(x, y, z, w):
	"""
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + y * y)
	roll_x = math.atan2(t0, t1)

	t2 = +2.0 * (w * y - z * x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	pitch_y = math.asin(t2)

	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (y * y + z * z)
	yaw_z = math.atan2(t3, t4)

	return roll_x, pitch_y, yaw_z  # in radians


def quaternion_to_rotation_matrix(q):
	"""Return a 3x3 rotation matrix representing the orientation specified by a quaternion in x,y,z,w format.
    The matrix is a Python list of lists.
    """

	x = q[0]
	y = q[1]
	z = q[2]
	w = q[3]

	return [[w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
			[2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
			[2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z]]

def telloState(streamingClient):

	# Retrieve rigid body data from OptiTrack.
	id_num = streamingClient.rigidBodyListener[0, 0]
	pos = streamingClient.rigidBodyListener[0, 1]
	pos = np.vstack(pos)
	#print('pos = ', pos)
	
	quato = streamingClient.rigidBodyListener[0, 2]
	quat = np.vstack(quato)
	#print('quat', quat)
	# Rotate coordinates to aircraft standard (forward x, right y, down z) from
	# (forward x, right z, up y).
	pos_quat = np.concatenate((pos, quat), axis=0).squeeze()
	#print('%%%%% pos quat = ', pos_quat)
	SE_motive = pos_quat2SE(pos_quat)
	#print('%%% SE = ', SE_motive)


	# 'zyx' works, but need to swap euler[0] and euler[2]:
	euler_motive = R.from_matrix(SE_motive[0:3, 0:3]).as_euler('zyx', degrees=False)
	#new_quat = np.vstack([quato[0], quato[2], -quato[1], quato[3]])
	#euler_matrix = quaternion_to_rotation_matrix([quato[0], -quato[2], quato[1], quato[3]])
	#euler_motive = euler_from_quaternion(quato[0], quato[2], quato[1], -quato[3])

	#euler_motive = R.from_quat(new_quat.squeeze()).as_euler('zyx', degrees=False)
	euler_motive = np.flip(euler_motive)

	#euler_motive = R.from_matrix(euler_matrix).as_euler('zxy', degrees=False)
	#qx, qy, qz, qw = quato[3], -quato[0], -quato[1], quato[3]
	#my_quat = np.vstack([qx, qy, qz, qw])

	#euler_motive = R.from_quat(my_quat.squeeze()).as_euler('xyz', degrees=True)


	#euler = Eul_FromQuat(quat.squeeze())

	rotated_pos = Cx(-np.pi/2) @ pos
	rotated_pos = rotated_pos * 100  # change to [cm]

	rotated_quat = ConvertRHSRotYUp2Zdown(quat)
	#print('rotated_quat', rotated_quat)
	
	#https://stackoverflow.com/questions/18818102/convert-quaternion-representing-rotation-from-one-coordinate-system-to-another

	M = quaternion2matrix(quat)
	M_rotated = Cx(-np.pi/2) @ M
	#euler = rotationMatrix2Euler(M_rotated)

	#print('pos = {} |||| euler_d = {}'.format(pos.squeeze(), euler_motive/np.pi*180.))

	return (id_num, euler_motive, np.array(SE_motive))


def patchState(streamingClient):

	# Retrieve rigid body data from OptiTrack.
	id_num = streamingClient.rigidBodyListener[1, 0]
	pos = streamingClient.rigidBodyListener[1, 1]
	pos = np.vstack(pos)
	#print('Patch_pos = ', pos)
	
	quat = streamingClient.rigidBodyListener[1, 2]
	quat = np.vstack(quat)
	#print('Patch_quat', quat)
	# Rotate coordinates to aircraft standard (forward x, right y, down z) from
	# (forward x, right z, up y).
	pos_quat = np.concatenate((pos, quat), axis=0).squeeze()
	#print('%%%%% pos quat = ', pos_quat)
	SE_motive = pos_quat2SE(pos_quat)
	#print('%%% SE = ', SE_motive)

	# 'zyx' works, but need to swap euler[0] and euler[2]:
	euler_motive = R.from_matrix(SE_motive[0:3, 0:3]).as_euler('zyx', degrees=False)
	#euler = R.from_quat(quat.squeeze()).as_euler('zyx', degrees=False) 
	euler_motive = np.flip(euler_motive)

	#print('rotated_quat', rotated_quat)

	print('Patch_pos = {} |||| Patch_euler_d = {}'.format(pos.squeeze(), euler_motive/np.pi*180.))

	return (id_num, euler_motive, np.array(SE_motive))


def setwaypoint(streamingClient, num_of_waypoints=5):

	start_SE = telloState(streamingClient)[-1] # of tello
	start_position = start_SE[0:3, -1]	

	# State number of waypoints and list them below. Note units are in cm and 
	# order is (x,y,z) in the standard aircraft coordinate system.
	# Remember, forward x, right y, down z
	waypoints = np.zeros([num_of_waypoints, 3])

	# Waypoint 1 is the starting point
	waypoints[0] = start_position

	# Put Waypoint 1 here
	waypoints[1] = waypoints[0] + np.array([100, 0, 0])

	# Put Waypoint 2 here
	waypoints[2] = waypoints[1] + np.array([0, 50, 0])

	# Put Waypoint 3 here
	waypoints[3] = waypoints[2] + np.array([0, -50, 0])

	# Put Waypoint 4 here
	waypoints[4] = waypoints[3] + np.array([-100, 0, 0])

	return waypoints


def waypointUpdate(true_state, waypoint):
	distance_th = 20
	# Attain current position (x,y,z)
	current_position = true_state[1]

	try:
		# Relative vector from current position to current waypoint
		r_wd = waypoint[waypointUpdate.current_waypoint] - current_position
	except (IndexError, AttributeError):
		# First time run or all waypoints reached therefore set to start
		waypointUpdate.current_waypoint = 0
		r_wd = waypoint[waypointUpdate.current_waypoint] - current_position

	# Distance to the next waypoint. Transition to next if within 50 cm.
	distance_to_waypoint = np.linalg.norm(r_wd)
	if distance_to_waypoint < distance_th:
		waypointUpdate.current_waypoint += 1
		# r_wd will be updated in next iteration to avoid IndexError's.

	return r_wd, waypointUpdate.current_waypoint


def quaternion2Euler(quat):

	# Separate variables for the quaternions
	q0 = quat[3]
	q1 = quat[0]
	q2 = quat[1]
	q3 = quat[2]

	# Calculate the Euler Angles
	theta = np.arctan2(q0 * q2 - q1 * q3,
		np.sqrt((q0 ** 2 + q1 ** 2 - 0.5) ** 2 + (q1 * q2 + q0 * q3) ** 2))
	phi = np.arctan2(q2 * q3 + q0 * q1, q0 ** 2 + q3 ** 2 - 0.5)
	psi = np.arctan2(q1 * q2 + q0 * q3, q0 ** 2 + q1 ** 2 - 0.5)

	# Construct the return array
	euler = np.array([phi, theta, psi])
	return euler


def rotationMatrix2Euler(R):

	sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
		
	singular = sy < 1e-6

	if not singular:
		x = math.atan2(R[2,1] , R[2,2])
		y = math.atan2(-R[2,0], sy)
		z = math.atan2(R[1,0], R[0,0])
	else :
		x = math.atan2(-R[1,2], R[1,1])
		y = math.atan2(-R[2,0], sy)
		z = 0

	return np.array([x, y, z])


def quaternion2matrix(quat):
	# Separate variables for the quaternions. Motive's quat order is (x,y,z,w).
	w = quat[3]
	x = quat[0]
	y = quat[1]
	z = quat[2]

	M = 2 * np.array([[(w**2 + x**2 - 0.5), (x*y - w*z), (w*y + x*z)],
			[(w*z + x*y), (w**2 + y**2 - 0.5), (y*z - w*x)],
			[(x*z - w*y), (w*x + y*z), (w**2 + z**2 - 0.5)]])

#	M = np.array([[(w**2 + x**2 - y**2 - z**2), (2*x*y + 2*w*z), (2*x*z - 2*w*y)],
#		      [(-2*w*z + 2*x*y), (w**2 -x**2 + y**2 - z**2), (2*y*z + 2*w*x)],
#		      [(2*x*z + 2*w*y), (-2*w*x + 2*y*z), (w**2 -x**2 -y**2 + z**2)]])
	return M


def Cx(angle):
	# Rotation matrix
	rotate_x = np.array([[1., 0., 0.], [0., np.cos(angle), -np.sin(angle)], [0., np.sin(angle), np.cos(angle)]], dtype='float64')
	return rotate_x


def Cy(angle):
	# Rotation matrix
	rotate_y = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]], dtype='float64')
	return rotate_y


def Cz(angle):
	# Rotation matrix
	rotate_z = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]], dtype='float64')
	return rotate_z


def Eul_FromHMatrix(M, eul_order):
	# Matches rotationMatrix2Euler()
	cy = np.sqrt(M[0,0]*M[0,0] + M[1,0]*M[1,0])
	if (cy > 1e-6): # Not singular
		x = math.atan2(M[2,1], M[2,2])
		y = math.atan2(-M[2,0], cy)
		z = math.atan2(M[1,0], M[0,0])
	else:
		x = math.atan2(-M[1,2], M[1,1])
		y = math.atan2(-M[2,0], cy)
		z = 0
	return np.array([x, y, z])


def Eul_FromQuat(q, eul_order='xyz'):
	M = np.eye(4)
	Nq = q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2
	s = (2.0 / Nq) if (Nq > 0.0) else 0.0
	xs = q[0]*s;	  ys = q[1]*s;	 zs = q[2]*s
	wx = q[3]*xs;	  wy = q[3]*ys;	 wz = q[3]*zs
	xx = q[0]*xs;	  xy = q[0]*ys;	 xz = q[0]*zs
	yy = q[1]*ys;	  yz = q[1]*zs;	 zz = q[2]*zs
	M[0,0] = 1.0 - (yy + zz) 
	M[0,1] = xy - wz 
	M[0,2] = xz + wy
	M[1,0] = xy + wz
	M[1,1] = 1.0 - (xx + zz)
	M[1,2] = yz - wx
	M[2,0] = xz - wy
	M[2,1] = yz + wx
	M[2,2] = 1.0 - (xx + yy)
	return Eul_FromHMatrix(M, eul_order)


def ConvertRHSRotYUp2Zdown(quat):

# https://personal.utdallas.edu/~sxb027100/dock/quaternion.html

	x, y, z, w = quat
	# -90 deg rotation about +X
	angle = -np.pi / 2.0
	x2 = np.sin(angle / 2.0)
	y2 = 0.0
	z2 = 0.0
	w2 = np.cos(angle / 2.0)

	# q1 * q2:
	# rotate quat using quat multiply
	qxNew = w*x2 + x*w2 + y*z2 - z*y2
	qyNew = w*y2 - x*z2 + y*w2 + z*x2
	qzNew = w*z2 + x*y2 - y*x2 + z*w2
	qwNew = w*w2 - x*x2 - y*y2 - z*z2
	'''	
	# q2 * q1 :
	qxNew = w2*x + x2*w + y2*z - z2*y
	qyNew = w2*y - x2*z + y2*w + z2*x
	qzNew = w2*z + x2*y - y2*x + z2*w
	qwNew = w2*w - x2*x - y2*y - z2*z
	'''
	return np.array([qxNew, qyNew, qzNew, qwNew])


def euler_from_quaternion(x, y, z, w):
# https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return np.array([roll_x, pitch_y, yaw_z]) # in radians


def pos_quat2SE(quat_data):
	# Assumed quat_data order is (pos, quat)
	SO = R.from_quat(quat_data[3:7]).as_matrix()
	SE = np.matrix(np.eye(4))
	SE[0:3,0:3] = np.matrix(SO)
	SE[0:3,3]   = np.matrix(quat_data[0:3]).T
	return SE


def SE2kitti(SE):
	return np.array(SE[0:3,:]).reshape(1,12)


def invert_SE(se):
	'''Gets pose in SE format (R t
		               0 1)
	Returns se^-1 = (R^T -R^T*t
		          0     1  )''' 
	se = np.array(se)
	inv_se = np.eye(4)
	R_T = se[0:3, 0:3].T
	inv_se[0:3, 0:3] = R_T
	inv_se[0:3, -1] = - R_T @ se[0:3, -1]
	return np.array(inv_se)


def calc_initial_SE_motive2telloNED_inv(T_w_0):
	T_Yup2NED_inv = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
	T_Yup2NED = invert_SE(T_Yup2NED_inv)
	T_w_b0 = T_w_0 @ T_Yup2NED
	T_w_b0_inv = invert_SE(T_w_b0)
	return T_w_b0_inv


def calc_initial_SE_motive2telloNED_inv_v2(T_w_0):
	quat_inv_transform = np.array([-1, 0, 0, 1])
	pos_quat_inv = np.concatenate((np.array([0, 0, 0]), quat_inv_transform), axis=0).squeeze()
	SE_transform_inv = pos_quat2SE(pos_quat_inv)
	T_w_b0_inv = T_w_0 @ SE_transform_inv
	return T_w_b0_inv


def SE_motive2telloNED(SE_motive, T_w_b0_inv):
	T_Yup2NED_inv = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
	T_Yup2NED = invert_SE(T_Yup2NED_inv)
	T_w_bi = SE_motive @ T_Yup2NED
	T_b0_bi = T_w_b0_inv @ T_w_bi
	return T_b0_bi

def SE_motive2telloNED_v2(SE_motive, T_w_b0_inv):
	quat_inv_transform = np.array([-1, 0, 0, 1])
	pos_quat_inv = np.concatenate((np.array([0, 0, 0]), quat_inv_transform), axis=0).squeeze()
	SE_transform_inv = pos_quat2SE(pos_quat_inv)
	SE_tello_abs = SE_motive @ SE_transform_inv
	T_b0_bi = T_w_b0_inv @ SE_tello_abs
	return T_b0_bi

def SE_motive2tello_absNED(SE_motive):
	T_Yup2NED_inv = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
	T_Yup2NED = invert_SE(T_Yup2NED_inv)
	SE_tello_abs = SE_motive @ T_Yup2NED
	return SE_tello_abs

def SE_motive2tello_absNED_v2(SE_motive):
	quat_inv_transform = np.array([-1, 0, 0, 1])
	pos_quat_inv = np.concatenate((np.array([0, 0, 0]), quat_inv_transform), axis=0).squeeze()
	SE_transform_inv = pos_quat2SE(pos_quat_inv)
	SE_tello_abs = SE_motive @ SE_transform_inv
	return SE_tello_abs






def tello_go_xyz_speed_from_NED(tello, x, y, z, speed):
	tello.go_xyz_speed(int(x), int(-y), int(-z), speed)


def tello_rotate_clockwise_from_NED(tello, yaw):
	tello.rotate_counter_clockwise(yaw)


def tello_rotate_counter_clockwise_from_NED(tello, yaw):
	tello.rotate_counter_clockwise(yaw)


def mean_std_hovering(streamingClient, N_samples=10):
	states = np.zeros((N_samples, 3))	
	for idx in range(N_samples):
		state = telloState(streamingClient)
		se = state[-1]
		pos = se[0:3, -1]
		states[idx, :] = pos
	return np.mean(states, axis=0), np.std(states, axis = 0)
	


