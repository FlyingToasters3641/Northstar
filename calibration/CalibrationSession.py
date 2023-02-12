import datetime
import os
from typing import List

import cv2
import numpy

from config.ConfigSource import FileConfigSource


class CalibrationSession:
    _all_charuco_corners: List[numpy.ndarray] = []
    _all_charuco_ids: List[numpy.ndarray] = []
    _imsize = None

    def __init__(self) -> None:
        # self._aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
        # self._aruco_params = cv2.aruco.DetectorParameters()
        # self._detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)

        # self._charuco_board = cv2.aruco.CharucoBoard(
        #     (12, 9), 0.030, 0.023, self._aruco_dict)
        # self._charuco_detector = cv2.aruco.CharucoDetector(self._charuco_board)

        self._aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000)
        self._aruco_params = cv2.aruco.DetectorParameters_create()
        self._charuco_board = cv2.aruco.CharucoBoard_create(
            12, 9, 0.030, 0.023, self._aruco_dict)

    def process_frame(self, image: cv2.Mat, save: bool) -> None:
        # Get image size
        if self._imsize == None:
            self._imsize = (image.shape[0], image.shape[1])

        # Detect tags
        #(corners, ids, rejected) = self._detector.detectMarkers(image)
        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, self._aruco_dict, parameters=self._aruco_params)

        if len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(image, corners)

            # Find Charuco corners
            #(retval, charuco_corners, charuco_ids) = cv2.aruco.interpolateCornersCharuco(
            #    corners, ids, image, self._charuco_board)
            #ret = cv2.aruco.interpolateCornersCharuco(
            #    corners, ids, image, self._charuco_board)
            (retval, charuco_corners, charuco_ids) = cv2.aruco.interpolateCornersCharuco(
                corners, ids, image, self._charuco_board)
            if retval:
                cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)

            #(corners, ids, charuco_corners, charuco_ids) = self._charuco_detector.detectBoard(image) #, corners, ids)
            #ret = self._charuco_detector.detectBoard(image) #, corners, ids)

            #res = self._charuco_detector.detectBoard(image) #, corners, ids)
            #cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)

            # Save corners
            if save:
                self._all_charuco_corners.append(charuco_corners)
                self._all_charuco_ids.append(charuco_ids)
                print("Saved calibration frame")

    def finish(self) -> None:
        if len(self._all_charuco_corners) == 0:
            print("ERROR: No calibration data")
            return

        if os.path.exists(FileConfigSource.CALIBRATION_FILENAME):
            os.remove(FileConfigSource.CALIBRATION_FILENAME)

        # toRemove = []
        # for i in range(0,len(self._all_charuco_ids)):
        #     if (self._all_charuco_corners[i] is None or self._all_charuco_ids[i] is None):
        #         toRemove.append(i)
        #     elif (len(self._all_charuco_corners[i]) < 4 or len(self._all_charuco_ids[i]) < 4):
        #         toRemove.append(i)
        #     else:
        #         print (f"%f %f", len(self._all_charuco_corners[i]), len(self._all_charuco_ids[i]))
        # for i in range(len(toRemove) - 1, 0, -1):
        #     self._all_charuco_corners.pop(toRemove[i])
        #     self._all_charuco_ids.pop(toRemove[i])

        (retval, camera_matrix, distortion_coefficients, rvecs, tvecs) = cv2.aruco.calibrateCameraCharuco(
            self._all_charuco_corners, self._all_charuco_ids, self._charuco_board, self._imsize, None, None)

        if retval:
            calibration_store = cv2.FileStorage(FileConfigSource.CALIBRATION_FILENAME, cv2.FILE_STORAGE_WRITE)
            calibration_store.write("calibration_date", str(datetime.datetime.now()))
            calibration_store.write("camera_resolution", self._imsize)
            calibration_store.write("camera_matrix", camera_matrix)
            calibration_store.write("distortion_coefficients", distortion_coefficients)
            calibration_store.release()
            print("Calibration finished")
        else:
            print("ERROR: Calibration failed")
