import getopt
import ntcore
import sys
import cv2
from config.config import ConfigStore


class CalibrationCommandSource:
    def get_calibrating(self) -> bool:
        return False

    def get_capture_flag(self) -> bool:
        return False


class NTCalibrationCommandSource(CalibrationCommandSource):
    _init_complete: bool = False
    _active_entry: ntcore.BooleanEntry
    _capture_flag_entry: ntcore.BooleanEntry

    def _init(self, config_store: ConfigStore):
        if not self._init_complete:
            nt_table = ntcore.NetworkTableInstance.getDefault().getTable(
                "/" + config_store.local_config.device_id + "/calibration")
            self._active_entry = nt_table.getBooleanTopic("active").getEntry(False)
            self._capture_flag_entry = nt_table.getBooleanTopic("capture_flag").getEntry(False)
            self._active_entry.set(False)
            self._capture_flag_entry.set(False)
            self._init_complete = True

    def get_calibrating(self, config_store: ConfigStore) -> bool:
        self._init(config_store)
        calibrating = self._active_entry.get()
        if not calibrating:
            self._capture_flag_entry.set(False)
        return calibrating

    def get_capture_flag(self, config_store: ConfigStore) -> bool:
        self._init(config_store)
        if self._capture_flag_entry.get():
            self._capture_flag_entry.set(False)
            return True
        return False

class ArgumentCalibrationCommandSource(CalibrationCommandSource):
    _calibrate: bool = False
    _noCapture: bool = False
    _stop: bool = False

    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:],"cn",["calibrate","no-capture"])
        for opt, arg in opts:
            if opt in ("-c", "--calibrate"):
                self._calibrate = True
            elif opt in ("-n", "--no--capture"):
                self._noCapture = True

    def get_calibrating(self, config_store: ConfigStore) -> bool:
        if cv2.pollKey() != -1:
            self._stop = True

        return self._calibrate and not self._stop

    def get_capture_flag(self, config_store: ConfigStore) -> bool:
        return not self._noCapture
