import numpy as np
import time
from position import telloState

def runWaypoint(true_state, r_wd, dt, tello1):

	# psi must be defined
	current_orient_euler = true_state[3]

	# Vector projection along x-y plane (height component (z) is zero)
	r_wd_proj = np.array([r_wd[0], r_wd[1], 0])

	# Find the yaw angle (between the vector and the x-axis) through the dot
	# product formula. This gives the yaw angle required to face the waypoint.
	yaw_w = np.arctan2(r_wd_proj[1], r_wd_proj[0])		# in radians

	# Offset angle between drone heading and waypoint heading angle (yaw)
	yaw_w = yaw_w * 180/np.pi   # in degrees
	beta = yaw_w - (current_orient_euler[2] * 180/np.pi)
	if beta > 180:
		beta = beta - 360
	elif beta < -180:
		beta = beta + 360

	# Use the angle to find components of the vector projection in the forward/
	# back direction and the left/right direction.
	signal = np.array([np.linalg.norm(r_wd_proj) * np.sin(beta * np.pi/180),	# Lateral
		np.linalg.norm(r_wd_proj) * np.cos(beta * np.pi/180),	# Longitudinal
		r_wd[2],			# Vertical
		beta])				# yaw

	reference = np.array([0, 0, 0, 0])
	error = signal - reference

	try:
		controllerWaypoint(error, runWaypoint.prev_error, dt, tello1)
	except AttributeError:
		controllerWaypoint(error, error, dt, tello1)		# first run
	runWaypoint.prev_error = error
	return error


def controllerWaypoint(error, prev_error, dt, tello1):

	speed = 20
	# Numerical differentiation - first order difference scheme
	error_dot = (error - prev_error) / dt

	# PD constants and controller (Standard form)
	Kp = np.array([0.4, 0.7, 1.0, 1.0])	# lr, fb, ud, yaw
	Td = np.array([0, 0, 0, 0])
	pid_input = Kp * (error + Td * error_dot)

	# Longitudinal to laterial ratio
	ratio = pid_input[1] / pid_input[0]
	if ratio == 0:
		pass

	#print("############# PID_input = ", pid_input)

	# Maintain ratio between the limited controller inputs
	pid_input = controllerLimits(pid_input, -100.0, 100.0)
	#print("############# PID_input after limit = ", pid_input)

	if abs(ratio) > 1:
		pid_input[0] = (1 / ratio) * pid_input[1]
	else:
		pid_input[1] = ratio * pid_input[0]
	#print("############# PID_input after ratio = ", pid_input)
	pid_input[np.isnan(pid_input)] = 0

	#tello1.send_rc_control(left_right_velocity=int(pid_input[0]), forward_backward_velocity=int(pid_input[1]),
	#					   up_down_velocity=-int(pid_input[2]), yaw_velocity=int(pid_input[3]))
	tello1.go_xyz_speed()


def move2WP(tello, curr_state, WP_NED, speed=20):
	rel_WP = WP_NED - curr_state[1][0,:]
	tello.go_xyz_speed(int(rel_WP[0]), int(rel_WP[1]), int(rel_WP[2]), speed)


def controllerLimits(cont_input, min_limit, max_limit):
	limited_input = np.where(cont_input > max_limit, max_limit, cont_input)
	limited_input = np.where(limited_input < min_limit, min_limit, limited_input)
	return limited_input


def clamp100(x: int) -> int:
	return max(-100, min(100, x))


def control_loop(tello, streamingClient, WP_NED):
	distance_th = 28  # [cm]
	timeout = 10  # [s]
	speed = 20  # [cm/s]
	dt = 2

	start = time.time()

	true_state = telloState(streamingClient)
	current_position = true_state[1][0, :]
	r_wd = WP_NED - current_position
	distance_to_waypoint = np.linalg.norm(r_wd)

	while (time.time() - start < timeout) and (distance_to_waypoint > distance_th):
		# psi must be defined
		true_state = telloState(streamingClient)
		current_orient_euler = true_state[3][0, :]
		print('^^^^^^^^^^^^^^^^^^ ', current_orient_euler)
		current_position = true_state[1][0, :]
		r_wd = WP_NED - current_position
		distance_to_waypoint = np.linalg.norm(r_wd)
		print("^^^^^^^^^^^ Distance to WP: '", distance_to_waypoint)
		# Vector projection along x-y plane (height component (z) is zero)
		r_wd_proj = np.array([r_wd[0], r_wd[1], 0])
		R = np.linalg.norm(r_wd_proj)
		# Find the yaw angle (between the vector and the x-axis) through the dot
		# product formula. This gives the yaw angle required to face the waypoint.
		yaw_w = np.arctan2(r_wd_proj[1], r_wd_proj[0])		# in radians
		# Offset angle between drone heading and waypoint heading angle (yaw)
		yaw_w = yaw_w * 180/np.pi   # in degrees
		beta = yaw_w - (current_orient_euler[2] * 180/np.pi)
		if beta > 180:
			beta = beta - 360
		elif beta < -180:
			beta = beta + 360
		# Use the angle to find components of the vector projection in the forward/
		# back direction and the left/right direction.
		signal = np.array([R * np.sin(beta * np.pi/180),	# Lateral
						   R * np.cos(beta * np.pi/180),	# Longitudinal
						   r_wd[2],			# Vertical
						   beta])				# yaw
		reference = np.array([0, 0, 0, 0])
		error = signal - reference
		# P controller
		#Kp = np.array([0.4, 0.7, 0.8, 0.8])	# lr, fb, ud, yaw
		#pid = Kp * error
		#pid = controllerLimits(pid, -100, 100)
		#tello.send_rc_control(int(pid[0]), int(pid[1]), int(pid[2]), int(pid[3]))
		dx, dy, dz = controllerLimits(r_wd[0], -100, 100), controllerLimits(r_wd[1], -100, 100), controllerLimits(r_wd[2], -100, 100)
		#tello.go_xyz_speed(int(dx), int(dy), int(dz), speed)
		time.sleep(dt)
