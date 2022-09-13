import cv2
import pickle


class ArucoDetector:
    def __init__(self, calib_filename, aruco_type=cv2.aruco.DICT_4X4_50):
        self.aruco_type = aruco_type
        self.aruco_dict = cv2.aruco.Dictionary_get(self.aruco_type)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.K = None
        self.dist_coeffs = None
        # Assumed topleft->clockwise order, and corresponding 4 Aruco IDs in the images: [1, 0, 2, 3]
        self.order_dict = {'topleft': 1, 'topright': 0, 'bottomright': 2, 'bottomleft': 3}
        self.mask_by_centers = False
        ArucoDetector.set_calibration(self, calib_filename)

    def detect_aruco_markers(self, frame, debug=False):
        key_list = list(self.order_dict.keys())
        val_list = list(self.order_dict.values())
        # detect ArUco markers in the input frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (corners, ids, rejected) = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params,
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

                if self.mask_by_centers:
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

                if debug:
                    # draw the ArUco marker ID on the frame
                    cv2.putText(frame, str(markerID),
                                (int(topLeft[0]), int(topLeft[1]) - 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

                    cv2.imshow("", frame)
                    key = cv2.waitKey(1) & 0xFF
            # Return the 4 corners in the order: topleft -> clockwise

        # cv2.imshow("",frame)
        return ret_corners

    def set_calibration(self, calib_filename):
        """Loads the camera calibration parameters. Modifies the Intrinsics matrix if needed."""
        objects = []
        with (open(calib_filename, "rb")) as openfile:
            while True:
                try:
                    objects.append(pickle.load(openfile))
                except EOFError:
                    break
        # print(objects)
        K = objects[0]["cam_matrix"]
        dist = objects[0]["dist_coeffs"]
        self.K = K
        self.dist_coeffs = dist

    def are_4_markers_detected(self, frame):
        corners = self.detect_aruco_markers(frame)
        return len(list(corners.keys())) == 4

