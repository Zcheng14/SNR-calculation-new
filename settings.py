import scipy.constants as constant
import os
import numpy as np

#  data reading path
current_path = os.getcwd()  # "/mount/src/snr-calculation"

common_file_path = {
    'spot_size': current_path + "/data/separation.txt",
    'focal_plane_module': current_path + "/data/focal_plane_module.txt",
    'dichoric_reflection': current_path + "/data/dichoric_reflection.txt",
    'dichoric_transmission': current_path + "/data/dichoric_transmission.txt",
    'blue_grating': current_path + "/data/blue_grating_eff.txt",
    'red_grating': current_path + "/data/red_grating_eff.txt"
}

# setting
default_blue_wavelengths = np.arange(460, 511)
default_red_wavelengths = np.arange(610, 691)
cut = 510
field_points = np.array([0, 5, 10, 15, 20, 25, 30])  # mm

telescope = {
    "Canon 400mm f/2.8 telephoto lens": {
        "diameter": 392.5 / 2.9,  # mm
        "focal length": 392.5,  # mm
        "throughput": current_path + "/data/canon_throughput.txt"
    },
    "Custom": {
        "diameter": None,
        "focal length": None,
        "throughput": None
    }
}

fiber = {
    "50 micron core fiber": 0.05,  # mm
    "Custom": None
}

detector_camera = {
    "Nikkor 58mm f/0.95 S Noct": {
        "focal length": 59.67,  # mm
        "throughput": current_path + "/data/nikkor_throughput.txt",
        "vignetting": current_path + "/data/vignetting.txt"
    },
    "Custom": {
        "focal length": None,  # mm
        "throughput": None,
        "vignetting": None
    }
}

detector = {
    "QHY600": {
        "QE": current_path + "/data/QHY600_QE.txt",
        "pixel size": 3.76,  # um
        "readnoise": 1.683,
        "darkcurrent": 0.0022,
        "temperature": -20
    },
    "QHY4040": {
        "QE": current_path + "/data/QHY4040_QE.txt",
        "pixel size": 9,  # um
        "readnoise": 0.81,
        "darkcurrent": 0.16048,  # T=-20 degree
        "temperature": -20
    },
    "Custom": {
        "QE": None,
        "pixel size": [None, None],
        "readnoise": None,
        "darkcurrent": None,
        "temperature": None
    }
}


# collimator
eff_collimator = 0.9  # assume 90% reflexivity
f_collimator = 200  # mm

# fiber
eff_fiber_transmission = 0.96  # assume 4% reflectivity

# constant
speed_of_light = constant.speed_of_light  # ~3e8 m/s
h = constant.h  # ~6.626e-34 J/Hz

# grating
blue_AR_coating_paths = [current_path + '/data/grating_AR_blue_1.txt',
                         current_path + '/data/grating_AR_blue_2.txt',
                         current_path + '/data/grating_AR_blue_3.txt',
                         current_path + '/data/grating_AR_blue_4.txt']
red_AR_coating_paths = [current_path + '/data/grating_AR_red_1.txt',
                        current_path + '/data/grating_AR_red_2.txt',
                        current_path + '/data/grating_AR_red_3.txt',
                        current_path + '/data/grating_AR_red_4.txt']

line_density_blue = 3559  # lines/mm in blue channel
line_density_red = 2632  # lines/mm in red channel

incident_angle = 61.93  # degree
