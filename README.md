# Tello-Drone-Control

![](https://github.com/ilyak93/Tello-Drone-Control/blob/main/DJITelloPy/visualisation_pose_VO_planned/frame_planned_executed_predicted.gif)

Here you can see the tello-edu drone navigates itself to a simulated imaginary target a few meters in front, by using a simple carrot chasing navigation which I implemented for it. The recording shows the actual path (red), the planned movement by the navigation algorithm for each step (green) and also the estimation of the Visual Odometry (VO) model (Deep Neural Network based), (blue).

This is a simulation of a continuous flight (which is divided to stops distanced approximately 25 cm one from another.) during which a VO helps the drone to estimate its location, which is usually done by the GPS.

Here I only take a look at its performance (not that bad for the record),without taking the VO estimation for actual navigation, but of-course it can work as it is taking it into account, simply by replacing the real location by the VO estimation.

The real location in this toy example is gotten by using the pads, from which the drone can estimate the distance. Of-course a more reliable location estimation system will be used further (OptiTrack mockup, GPS and etc).

Threading events were used to implement the appropriate thread-based communication to make the recordings happen after each movement without any recording while the drone stops and waits (milliseconds/nanoseconds) for the next command to start. In more advanced version I've also worked on adding support for recording during the movement, but the key-points where the VO influence the navigation is in those stops (and in real life scenario continuously, a simulation of which is taking place, as was mentioned).


Further development of this module incorporated Motive OptiTrack Mockup for exact location estimation of the drone, illustration of which you can see down below:


![](https://github.com/ilyak93/Tello-Drone-Control/blob/main/DJITelloPy/visualisation_pose_VO_planned_with_Motive_OptiTrack/video_2023-03-31_20-20-10.gif)

Motive Perspective:
![](https://github.com/ilyak93/Tello-Drone-Control/blob/main/DJITelloPy/visualisation_pose_VO_planned_with_Motive_OptiTrack/2023-03-28-10-00-03.gif)

Center to center trajectory (Bottom to Top):
![](https://github.com/ilyak93/Tello-Drone-Control/blob/main/DJITelloPy/visualisation_pose_VO_planned_with_Motive_OptiTrack/1_traj.gif)

Left to center trajectory (Bottom to Top):
![](https://github.com/ilyak93/Tello-Drone-Control/blob/main/DJITelloPy/visualisation_pose_VO_planned_with_Motive_OptiTrack/2_traj.gif)

Right to center trajectory (Bottom to Top):
![](https://github.com/ilyak93/Tello-Drone-Control/blob/main/DJITelloPy/visualisation_pose_VO_planned_with_Motive_OptiTrack/3_traj.gif)

Center to center trajectory (Top to Bottom):
![](https://github.com/ilyak93/Tello-Drone-Control/blob/main/DJITelloPy/visualisation_pose_VO_planned_with_Motive_OptiTrack/4_traj.gif)

Center to center trajectories (Top to Bottom) statistics:
![](https://github.com/ilyak93/Tello-Drone-Control/blob/main/DJITelloPy/visualisation_pose_VO_planned_with_Motive_OptiTrack/traj_mean.gif)


