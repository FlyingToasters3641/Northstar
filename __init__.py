

import getopt
import sys
import time

import cv2
import ntcore

from calibration.CalibrationCommandSource import (CalibrationCommandSource,
                                                  NTCalibrationCommandSource,
                                                  ArgumentCalibrationCommandSource)
from calibration.CalibrationSession import CalibrationSession
from config.config import ConfigStore, LocalConfig, RemoteConfig
from config.ConfigSource import ConfigSource, FileConfigSource, NTConfigSource
from output.OutputPublisher import NTOutputPublisher, OutputPublisher
from output.overlay_util import *
from output.StreamServer import MjpegServer
from pipeline.Capture import GStreamerCapture
from pipeline.FiducialDetector import ArucoFiducialDetector
from pipeline.PoseEstimator import SquareTargetPoseEstimator

if __name__ == "__main__":
    config = ConfigStore(LocalConfig(), RemoteConfig())
    local_config_source: ConfigSource = FileConfigSource()
    remote_config_source: ConfigSource = NTConfigSource()
    #calibration_command_source: CalibrationCommandSource = NTCalibrationCommandSource()
    calibration_command_source: CalibrationCommandSource = ArgumentCalibrationCommandSource()

    capture = GStreamerCapture()
    fiducial_detector = ArucoFiducialDetector(cv2.aruco.DICT_APRILTAG_16h5)
    pose_estimator = SquareTargetPoseEstimator()
    output_publisher: OutputPublisher = NTOutputPublisher()
    stream_server = MjpegServer()
    calibration_session = CalibrationSession()

    local_config_source.update(config)
    ntcore.NetworkTableInstance.getDefault().setServer(config.local_config.server_ip)
    ntcore.NetworkTableInstance.getDefault().startClient4(config.local_config.device_id)
    stream_server.start(config)

    stop_calibration = False
    is_calibration_mode = False
    opts, args = getopt.getopt(sys.argv[1:],"cn",["calibrate","no-capture"])
    for opt, arg in opts:
        if opt in ("-c", "--calibrate"):
            is_calibration_mode = True

    frame_count = 0
    last_print = 0
    was_calibrating = False
    while True:
        try:
            remote_config_source.update(config)
            timestamp = time.time()
            success, image = capture.get_frame(config)
            if not success:
                time.sleep(0.5)
                continue

            fps = None
            frame_count += 1
            currTime = time.time()
            if currTime - last_print > 1:
                last_print = currTime
                fps = frame_count
                print("Running at", frame_count, "fps")
                frame_count = 0

            if calibration_command_source.get_calibrating(config) and not stop_calibration:
                # Calibration mode
                was_calibrating = True
                calibration_session.process_frame(image, calibration_command_source.get_capture_flag(config))
                time.sleep(.8)
            elif was_calibrating:
                # Finish calibration
                calibration_session.finish()
                sys.exit(0)

            elif config.local_config.has_calibration:
                # Normal mode
                image_observations = fiducial_detector.detect_fiducials(image, config)
                pose_observations = [pose_estimator.solve_fiducial_pose(x, config) for x in image_observations]
                [overlay_image_observation(image, x) for x in image_observations]
                [overlay_pose_observation(image, config, x) for x in pose_observations]
                output_publisher.send(config, timestamp, pose_observations, fps)

            else:
                # No calibration
                print("No calibration found")
                time.sleep(0.5)

            stream_server.set_frame(image)
        except KeyboardInterrupt:
            stop_calibration = True
