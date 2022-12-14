import pickle

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# Calibration images base path & extension:
calib_path = 'C:/Users/vista/Desktop/DJI_Album/TELLO/calib_images_640x448/'
calib_files_ext = 'png'
chessBoard_grid = (19, 13)
output_filename = "tello_640_448_calib_djitellopy.p"
cornerSubPix_window = (5, 5)
SHOW_IMAGES = False

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessBoard_grid[0]*chessBoard_grid[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chessBoard_grid[0], 0:chessBoard_grid[1]].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob(calib_path+'*.'+calib_files_ext)

cam_res = cv2.imread(images[0]).shape[::-1]
cam_res = cam_res[1:]
print(cam_res)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if SHOW_IMAGES:
        imgplot = plt.imshow(gray)
        plt.show()
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, chessBoard_grid, flags=None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,cornerSubPix_window,(-1,-1),criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        if SHOW_IMAGES:
            img = cv2.drawChessboardCorners(img, chessBoard_grid, corners2,ret)
            cv2.imshow('img',img)
            cv2.waitKey(100)
if SHOW_IMAGES:
    cv2.waitKey(0)
    cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, cam_res, None, None)
assert ret
print(mtx)
print(dist)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total reprojection error: {}".format(mean_error/len(objpoints)) )


# Save the camera calibration result to disk (we won't worry about rvecs / tvecs)
cam_calib = {"cam_matrix": mtx,
             "dist_coeffs": dist}
with open(output_filename, "wb") as f:
    pickle.dump(cam_calib, f)

