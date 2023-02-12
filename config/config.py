from dataclasses import dataclass
import numpy
import numpy.typing


@dataclass
class LocalConfig:
    device_id: str = ""
    server_ip: str = ""
    stream_port: int = 8000
    has_calibration: bool = False
    camera_matrix: numpy.typing.NDArray[numpy.float64] = numpy.array([])
    distortion_coefficients: numpy.typing.NDArray[numpy.float64] = numpy.array([])


@dataclass
class RemoteConfig:
    camera_id: int = 0
    camera_resolution_width: int = 1600
    camera_resolution_height: int = 1200
    camera_auto_exposure: int = 0
    camera_exposure: int = 35
    camera_gain: int = 0
    fiducial_size_m: float = 6.0 / 39.3701


@dataclass
class ConfigStore:
    local_config: LocalConfig
    remote_config: RemoteConfig
