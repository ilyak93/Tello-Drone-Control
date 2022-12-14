# Applies a script over directories.

import os

script_name = 'gen_dataset_from_Aruco_images.py'
base_src_dir = 'C:/Users/vista/Desktop/DJI_Album/TELLO/OL_speed_60_190522_big_target/'
base_output_dir = 'C:/Users/vista/Desktop/DJI_Album/OL_speed_60_190522_big_target_proc/'

src_dirs = [os.path.join(base_src_dir, o) for o in os.listdir(base_src_dir)]

for src_dir in src_dirs:
    output_dir = os.path.join(base_output_dir, src_dir.split('/')[-1])
    os.system('python ' + script_name + ' ' + src_dir + ' ' + output_dir)
