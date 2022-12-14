import sys

import cv2
import numpy as np
import pickle
# https://stackoverflow.com/questions/50928743/matplotlib-animation-not-displaying-in-pycharm/50929022
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from time import time, sleep
import os
import csv
import shutil


#https://learnopencv.com/augmented-reality-using-aruco-markers-in-opencv-c-python/

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}

aruco_type = "DICT_4X4_50"  # The actual we use.


cam_calib_fname =  'tello_640_448_calib_djitellopy.p'  # pickle calibration file in OpenCV format.
test_fname = sys.argv[1]
#test_fname = 'C:/Users/vista/Desktop/OL_speed_60_160522_big_target/8'
target_im_to_project = 'C:/Users/vista/Desktop/DJI_Album/OL_trajs_speed_50_processed/adv_best_pert_250422.png'
output_dir = sys.argv[2]
#output_dir = 'C:/Users/vista/Desktop/8/'
coords_filename = 'mask_coords.csv'
GT_pose_filename = 'pose_file.csv'
patch_pose_VO_filename = 'patch_pose_VO.csv'

img_suffix = 'png'
IS_VIDEO = False  # if False - an image sequence assumed.
UNDISTORT = False  # Usually False, as it doesn't work well / not needed
PROJECT_IM = False  # more for debug - project an image on the mask
SAVE_MASKS = True   # save also the patch mask frames
SAVE_I0_I1 = True   # Save the frames with dark and bright patches
MASK_BY_CENTERS = False  # else, by external corners
CROP_AROUND_CENTER = False  # else, resize the image
CALC_PATCH_MASK_BY_VECTORS = False  # experimental, doesn't work for 2D case in general perspective trasformation, and 3D case requires the camera poses.
PATCH_SCALE = 2.  # Relevant if CALC_PATCH_MASK_BY_VECTORS == True
OUTPUT_FRAMES_SIZE = (640, 448)  # frames either cropped or resized to this

class Viewer:
    def __init__(self):
        self.fig = None
        self.frame = None  # current frame to render
        self.calib_filename = cam_calib_fname
        self.world_corners = np.zeros((4, 3), np.float32)  # 3rd column is Z=0
        self.image_corners = np.zeros((4, 2), np.float32)
        self.downscale = 1
        if IS_VIDEO:
            self.cap = cv2.VideoCapture(test_fname)
        else:
            self.image_list = sorted([os.path.join(test_fname, o) for o in os.listdir(test_fname) if o.endswith(img_suffix)])
        sleep(1.)  # warm up
        self.rvec = None
        self.tvec = None
        self.K = None
        self.dist_coeffs = None
        self.img0 = None
        self.w = None
        self.h = None
        Viewer.set_calibration(self)
        Viewer.fix_distortion(self)
        #Viewer.set_camera_extrinsics(self)
        # Calculate the projection matrix:
        #self.projection_mtx = Viewer.projection_matrix(self)

    def detect_aruco_markers(self, frame, debug=False):
        # Assumed topleft->clockwise order, and corresponding 4 Aruco IDs in the images: [1, 0, 2, 3]
        # load the ArUCo dictionary and grab the ArUCo parameters
        orderDict = {'topleft': 1, 'topright': 0, 'bottomright': 2, 'bottomleft': 3}
        key_list = list(orderDict.keys())
        val_list = list(orderDict.values())
        arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
        arucoParams = cv2.aruco.DetectorParameters_create()
        # detect ArUco markers in the input frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (corners, ids, rejected) = cv2.aruco.detectMarkers(
                                                            gray, arucoDict, parameters=arucoParams,
                                                            cameraMatrix=self.K, distCoeff=self.dist_coeffs)
        # verify *at least* one ArUco marker was detected
        if len(corners) > 0 and debug:
            cv2.aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers

        ret_corners = {}
        if len(corners) == 4:
            for (markerCorner, markerID) in zip(corners, ids):
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners

                # convert each of the (x, y)-coordinate pairs to integers
                bottomRight = (bottomRight[0], bottomRight[1])
                topLeft = (topLeft[0], topLeft[1])

                id_pos_str = key_list[val_list[markerID[0]]]

                if MASK_BY_CENTERS:
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                else:
                    # defined by outmost corner. we need to know the IDs for that.
                    if id_pos_str == 'topleft':
                        cX, cY = int(topLeft[0]), int(topLeft[1])
                    elif id_pos_str == 'topright':
                        cX, cY = int(topRight[0]), int(topRight[1])
                    elif id_pos_str == 'bottomright':
                        cX, cY = int(bottomRight[0]), int(bottomRight[1])
                    elif id_pos_str == 'bottomleft':
                        cX, cY = int(bottomLeft[0]), int(bottomLeft[1])

                ret_corners[id_pos_str] = (cX, cY)
                #frame[cY, cX, 1] = 255

                if debug:
                    # draw the ArUco marker ID on the frame
                    cv2.putText(frame, str(markerID),
                                (int(topLeft[0]), int(topLeft[1]) - 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

                    cv2.imshow("", frame)
                    key = cv2.waitKey(1) & 0xFF
            # Return the 4 corners in the order: topleft -> clockwise

        #cv2.imshow("",frame)
        return ret_corners

    def grab_frame(self):
        """Grabs a frame from the camera"""
        ret, frame = self.cap.read()
        # return cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        return frame

    def set_calibration(self):
        """Loads the camera calibration parameters. Modifies the Intrinsics matrix if needed."""
        objects = []
        with (open(self.calib_filename, "rb")) as openfile:
            while True:
                try:
                    objects.append(pickle.load(openfile))
                except EOFError:
                    break
        # print(objects)
        K = objects[0]["cam_matrix"]
        dist = objects[0]["dist_coeffs"]
        # modify the intrinsic matrix according to downscale:
        K[0, 0] /= self.downscale
        K[1, 1] /= self.downscale
        K[0, 2] /= self.downscale
        K[1, 2] /= self.downscale
        self.K = K
        self.dist_coeffs = dist

    def set_camera_extrinsics(self):
        """Finds the rotation and translation vectors by solving the 3D-2D point correspondance"""
        _ret, self.rvec, self.tvec = cv2.solvePnP(self.world_corners, self.image_corners, self.K, self.dist_coeffs)

    def project_world2image(self, world_coords):
        """Maps 'world_coords' to image coords. 'world_coords' assumed float numpy array with 3 columns X,Y,Z. """
        return cv2.projectPoints(world_coords, self.rvec, self.tvec, self.K, self.dist_coeffs)[0].reshape(-1, 2)

    def projection_matrix(self):
        """computes the 3D projection matrix"""
        # Compute the 3D projection matrix from the model to the current frame
        R, _ = cv2.Rodrigues(self.rvec)
        projection = np.hstack((R, self.tvec))
        return np.dot(self.K, projection)

    def fix_distortion(self, alpha=0):
        """Calculates modified intrinsics and ROI for cropping image after undistortion"""
        self.K_new, self.roi = cv2.getOptimalNewCameraMatrix(
                                    self.K, self.dist_coeffs, (self.w, self.h), alpha, (self.w, self.h))

    def undistort_image(self, img, crop=False):
        """Applies undistortion to an image, plus cropping to avoid black padding"""
        if np.isnan(self.K_new).any():  # getOptimalNewCameraMatrix failed
            crop = False
            dst = cv2.undistort(img, self.K, self.dist_coeffs)
        else:
            dst = cv2.undistort(img, self.K, self.dist_coeffs, None, self.K_new)
            if crop:
            # crop the image
                x, y, w, h = self.roi
                return dst[y:y+h, x:x+w]
        return dst

    def close(self):
        """Releases VideoCapture"""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()

    def calc_3d_rect_from_3_3d_points(self, center_3d, bottom_center_3d, bottom_right_3d, w, h, markers_dist_from_plane=0):
        # Assumptions: 3 points not co-linear. Our rectangle origin is the bottom_right_3d point.
        # v1 is from it to bottom_center, v2 is from bottom_center_right to center.
        # v1 is used as the horizontal vector, v3 as the vertical.
        # w, h are the width, height in same units of the points.
        # If markers_dist_from_plane is passed, it is compensated from the rectangle to allow it to reside on the actual plane the markers are glued on.
        # Returned coordinates order is [tl, tr, br, bl].
        p1 = np.array(center_3d)
        p2 = np.array(bottom_center_3d)
        p3 = np.array(bottom_right_3d)
        v1 = p1 - p2
        v1 /= np.linalg.norm(v1)
        v2 = p1 - p3
        v2 /= np.linalg.norm(v2)
        n = np.cross(v2, v1)
        n /= np.linalg.norm(n)
        v3 = np.cross(v1, n)
        v3 /= np.linalg.norm(v3)

        br = p3
        bl = br + v1 * w
        tr = br + v3 * h
        tl = tr + v1 * w
        return np.array([tl, tr, br, bl]) - n*markers_dist_from_plane

    def calc_2d_rect_from_4_2d_points(self, p1, p2, p3, p4, s_factor=1.):
        # Assumed p1 -> tr, p2 -> bl, p3 -> br, p4 -> tl
        p1 = np.array(p1, dtype=np.float64)
        p2 = np.array(p2, dtype=np.float64)
        p3 = np.array(p3, dtype=np.float64)
        p3 = np.array(p3, dtype=np.float64)
        v1 = p1 - p3
        n_v1 = 1. #np.linalg.norm(v1)
        v1 /= n_v1
        v2 = p2 - p3
        n_v2 = 1. #np.linalg.norm(v2)
        v2 /= n_v2
        v3 = p4 - p3
        n_v3 = 1. # np.linalg.norm(v3)
        v3 /= n_v3

        br = p3 #- v2 * n_v2 * h_factor/2
        bl = br + v1 * n_v1 * s_factor
        tr = br + v2 * n_v2 * s_factor
        tl = br + v3 * n_v3 * s_factor
        return np.array([tl, tr, br, bl], dtype=np.int32)


def add_transparent_overlay(image, image2, alpha=0.5):
    overlay = image.copy()
    overlay[image2 > 0] = 255  # TODO: replace 255 with grid_mask value?
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


###############################################################################
viewer = Viewer()

target_im = cv2.imread(target_im_to_project)
h_t, w_t = target_im.shape[:2]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# copy patch_pose_VO as is:
src_patch_pose_VO_filename = os.path.join(test_fname, patch_pose_VO_filename)
dst_patch_pose_VO_filename = os.path.join(output_dir, patch_pose_VO_filename)
shutil.copyfile(src_patch_pose_VO_filename, dst_patch_pose_VO_filename)

# init the new pose file:
src_gt_pose_filename = os.path.join(test_fname, GT_pose_filename)
gt_pose_filename = os.path.join(output_dir, GT_pose_filename)
src_gt_pose_file = open(src_gt_pose_filename, 'r', newline='')
gt_pose_file = open(gt_pose_filename, 'w', newline='')
src_gt_pose_reader = csv.reader(src_gt_pose_file)
gt_pose_writer = csv.writer(gt_pose_file)


# init mask_coords file. coords order:  bl->br->tl->tr
coords_filename = os.path.join(output_dir, coords_filename)
coords_file = open(coords_filename, 'w', newline='')
coords_writer = csv.writer(coords_file)

# apply the projective transform to the image
idx = 0
while idx < len(viewer.image_list):
    # Capture frame-by-frame
    start = time()
    if IS_VIDEO:
        frame = viewer.grab_frame()
    else:
        frame = cv2.imread(viewer.image_list[idx])

    cv2.imshow("original", frame)
    # Undistort frame
    if UNDISTORT:
        frame = viewer.undistort_image(frame)
    cv2.imshow("undistort", frame)

    h, w = frame.shape[:2]
    if viewer.downscale > 1:
        frame = cv2.resize(frame, (int(w / viewer.downscale), int(h / viewer.downscale)), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]

    if CROP_AROUND_CENTER:
        crop_width = 640
        crop_height = 448
        mid_x, mid_y = int(w / 2), int(h / 2)
        cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
        frame = frame[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
        h, w = frame.shape[:2]
    else:
        crop_width = 640
        crop_height = 448
        frame = cv2.resize(frame, (int(crop_width), int(crop_height)), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]

    viewer.h, viewer.w = h, w
    # frame = add_transparent_overlay(frame, grid_image, alpha=0.4)

    # Regardless of detection success, we iterate over the original pose file:
    pos_row = next(src_gt_pose_reader)

    if True:  # idx % 2 == 0:
        corners = viewer.detect_aruco_markers(frame)

        # If we detected good:
        if len(list(corners.keys())) == 4:
            hsl = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
            Lchannel = hsl[:, :, 1]

            #Lmask = cv2.inRange(Lchannel, 200, 255)
            #dark_mask = cv2.inRange(Lchannel, 0, 60)
            #res = cv2.bitwise_and(frame, frame, mask=dark_mask)
            #cv2.namedWindow("Lchannel", cv2.WINDOW_NORMAL)
            #cv2.imshow("Lchannel", Lchannel)
            pts_src = np.array([[0, 0], [w_t-1, 0], [w_t-1, h_t-1], [0, h_t-1]], dtype=np.int32)

            if CALC_PATCH_MASK_BY_VECTORS:
                pts_dst = viewer.calc_2d_rect_from_4_2d_points(corners['bottomleft'], corners['topright'],
                                                               corners['bottomright'], corners['topleft'], PATCH_SCALE)
            else:
                pts_dst = np.zeros((4, 2), dtype=np.int32)
                pts_dst[0, :], pts_dst[1, :], pts_dst[2, :], pts_dst[3, :] = \
                                    corners['topleft'], corners['topright'], corners['bottomright'], corners['bottomleft']

            cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
            frame2 = frame.copy()
            for i in range(4):
                frame2 = cv2.circle(frame2, pts_dst[i], 3, (255,0,255))
            cv2.imshow("debug", frame2)

            # Order: bl->br->tl->tr
            mask_coords = list(pts_dst[3]) + list(pts_dst[2]) + list(pts_dst[0]) + list(pts_dst[1])

            # frame[pts_dst[:,1], pts_dst[:,0],  1] = 255  # for debug
            cv2.namedWindow("Arena0", cv2.WINDOW_NORMAL)
            cv2.imshow("Arena0", frame)

            # Calculate Homography
            H, status = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)

            # Prepare a mask representing region to copy from the warped image into the original frame.
            mask = np.zeros([frame.shape[0], frame.shape[1]], dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32([pts_dst]), (255, 255, 255), cv2.LINE_AA)

            # # Apply Antialiasing filter on the mask:
            # kernel_size = 9
            # kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
            # kernel /= (kernel_size * kernel_size)
            # mask = cv2.filter2D(mask, -1, kernel)

            mask[mask > 0] = 255

            # if ERODE_MASK:
            # # Erode the mask to not copy the boundary effects from the warping
            #     element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            #     mask = cv2.erode(mask, element, iterations=3)
            #
            # if DILATE_MASK:
            # # Dilate the mask to not copy the boundary effects from the warping
            #     element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            #     mask = cv2.dilate(mask, element, iterations=1)

            # Calculate histogram with mask and without mask
            # Check third argument for mask
            hist_full = cv2.calcHist([Lchannel], [0], None, [256], [0, 256])
            hist_mask = cv2.calcHist([Lchannel], [0], mask, [256], [0, 256])

            # Binarize the histogram:
            ret2, th2 = cv2.threshold(Lchannel[mask], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            roi = Lchannel[mask > 0]
            dark_vals = roi[roi < ret2]

            ret3, th3 = cv2.threshold(dark_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            dark_vals = dark_vals[dark_vals < ret3]

            bright_vals = roi[roi > ret2]
            ret4, th4 = cv2.threshold(bright_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            #bright_vals = bright_vals[bright_vals > ret4]

            curr_black = dark_vals.mean()  # dark_vals.min()
            curr_white = bright_vals.mean()  # bright_vals.max()

            # Normalize the target image to 0-1:
            target_im = (target_im - target_im.min()) / (target_im.max() - target_im.min())
            target_im = target_im * (curr_white - curr_black) + curr_black

            # Warp source image to destination based on homography
            warped_image = cv2.warpPerspective(target_im, H, (frame.shape[1], frame.shape[0]))

            # plt.subplot(121)
            # #plt.plot(hist_mask)
            # plt.plot(hist_full), plt.plot(hist_mask)
            # plt.xlim([0, 256])
            # plt.subplot(122)
            # plt.imshow(Lchannel)
            # plt.show()

            # Copy the mask into 3 channels.
            warped_image = warped_image.astype(float)
            mask3 = np.zeros_like(warped_image)
            for i in range(0, 3):
                mask3[:, :, i] = mask / 255

            # Copy the masked warped image into the original frame in the mask region.
            warped_image_masked = cv2.multiply(warped_image, mask3)
            frame_masked = cv2.multiply(frame.astype(float), 1 - mask3)

            if SAVE_I0_I1:
                I0_out = np.zeros_like(frame_masked)
                I1_out = I0_out.copy()
                I0_out[mask3 > 0 ] = curr_black
                I1_out[mask3 > 0] = curr_white
                I0_out = cv2.add(I0_out, frame_masked)
                I1_out = cv2.add(I1_out, frame_masked)

            if PROJECT_IM:
                im_out = cv2.add(warped_image_masked, frame_masked)
            else:
                im_out = frame.copy()

            cv2.namedWindow("Arena", cv2.WINDOW_NORMAL)
            cv2.imshow("Arena", im_out/255)
            cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
            cv2.imshow("mask", mask/255)

            if SAVE_I0_I1:
                cv2.imwrite(os.path.join(output_dir, 'I0_' + str(idx).zfill(5) + '.png'), I0_out)
                cv2.imwrite(os.path.join(output_dir, 'I1_' + str(idx).zfill(5) + '.png'), I1_out)
            else:
                cv2.imwrite(os.path.join(output_dir, str(idx).zfill(5) + '.png'), im_out)

            if SAVE_MASKS:
                np.save(os.path.join(output_dir, 'patch_mask_' + str(idx).zfill(5)), mask)
                coords_writer.writerow(mask_coords)

            gt_pose_writer.writerow(pos_row)

    end = time()
    idx += 1
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release and close stuff:
viewer.close()
src_gt_pose_file.close()
gt_pose_file.close()
coords_file.close()
