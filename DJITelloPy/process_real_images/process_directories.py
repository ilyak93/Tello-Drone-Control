# Applies a script over directories.

import os

script_name = 'gen_dataset_from_Aruco_images.py'
base_src_dir = '/home/vista/ilya_tello_test/trajs_3D/dataset/OL_trajs_images_R25_btt_center_2/'
base_output_dir = '/home/vista/ilya_tello_test/trajs_3D/dataset/OL_trajs_images_R25_btt_center_2_proc/'

src_dirs = [os.path.join(base_src_dir, o) for o in os.listdir(base_src_dir)]

for src_dir in src_dirs:
    output_dir = os.path.join(base_output_dir, src_dir.split('/')[-1])
    os.system('python ' + script_name + ' ' + src_dir + ' ' + output_dir)


#TODO check the missing frame sometimes
