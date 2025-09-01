import scipy.constants as constant
import os
import numpy as np

#  data reading path
current_path = os.getcwd()  # "/mount/src/snr-calculation"

common_file_path = {
    'focal_plane_module': current_path + "/data/focal_plane_module/focal_plane_module.txt",
    'dichoric_reflection': current_path + "/data/dichoric/dichoric_reflection.txt",
    'dichoric_transmission': current_path + "/data/dichoric/dichoric_transmission.txt",
    'blue_grating': current_path + "/data/grating/blue_grating_eff.txt",
    'red_grating': current_path + "/data/grating/red_grating_eff.txt"
}

# setting
default_blue_wavelengths = np.arange(460, 511)
default_red_wavelengths = np.arange(610, 691)

default_system = {
    "Custom design lens": {
        "spot_path": current_path + "/data/spot_size/spot_custom.txt",
        "wavelength_panel": [465,505,625,680],
        "wavelength_range": np.concatenate((np.arange(465, 506), np.arange(625, 681)), axis=None)
    },
    "Nikkor lens": {
        "spot_path": current_path + "/data/spot_size/spot_nikkor.txt",
        "wavelength_panel": [460,510,610,690],
        "wavelength_range": np.concatenate((default_blue_wavelengths, default_red_wavelengths), axis=None)
    },
}

cut = 510
field_points = np.array([0, 5, 10, 15, 20, 25, 30])  # mm

telescope = {
    "Canon 400mm f/2.8 telephoto lens": {
        "diameter": 392.5 / 2.9,  # mm
        "focal length": 392.5,  # mm
        "throughput": current_path + "/data/throughput/canon_throughput.txt"
    },
    #"Custom": {
    #    "diameter": None,
    #    "focal length": None,
    #    "throughput": None
    #}
}

fiber = {
    "50 micron core fiber": 0.05,  # mm
    "Custom": None
}

detector_camera = {
    "Custom design lens":{
        "focal length": 99.5,  # mm
        "throughput": current_path + "/data/throughput/customlens_throughput.txt",
        "vignetting": {
            "path": current_path + "/data/vignetting/vignetting_custom.txt",
            "wavelength_range": np.concatenate((np.arange(465, 506), np.arange(625, 681)), axis=None)
            },
        "emergent_angle0_blue": 59.89,
        "emergent_angle0_red": 59.61 #degree
    },
    "Nikkor 58mm f/0.95 S Noct": {
        "focal length": 59.67,  # mm
        "throughput": current_path + "/data/throughput/nikkor_throughput.txt",
        "vignetting": {
            "path": current_path + "/data/vignetting/vignetting_Nikkor.txt",
            "wavelength_range": np.concatenate((default_blue_wavelengths, default_red_wavelengths), axis=None)
            },
        "emergent_angle0_blue": 61.93,
        "emergent_angle0_red": 61.93 #degree
    },
    "Custom": {
        "focal length": None,  # mm
        "throughput": None,
        "vignetting": {
            "path": None,
            "wavelength_range": None
            }
    }
}

detector = {
    "QHY461": {
        "QE": current_path + "/data/QE/QHY461_QE.txt",
        "pixel size": 3.76,  # um
        "readnoise": 1.32,
        "darkcurrent": 2.667e-4,
        "temperature": -20
    },
    "QHY600": {
        "QE": current_path + "/data/QE/QHY600_QE.txt",
        "pixel size": 3.76,  # um
        "readnoise": 1.683,
        "darkcurrent": 0.0011,
        "temperature": -30
    },
    #"QHY4040": {
    #    "QE": current_path + "/data/QE/QHY4040_QE.txt",
    #    "pixel size": 9,  # um
    #    "readnoise": 0.81,
    #    "darkcurrent": 0.16048,  # T=-20 degree
    #    "temperature": -20
    #},
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
eff_fiber_transmission = 0.90  # assume 10% reflectivity

# constant
speed_of_light = constant.speed_of_light  # ~3e8 m/s
h = constant.h  # ~6.626e-34 J/Hz

# grating
blue_AR_coating_paths = [current_path + '/data/grating/grating_AR_blue_1.txt',
                         current_path + '/data/grating/grating_AR_blue_2.txt',
                         current_path + '/data/grating/grating_AR_blue_3.txt',
                         current_path + '/data/grating/grating_AR_blue_4.txt']
red_AR_coating_paths = [current_path + '/data/grating/grating_AR_red_1.txt',
                        current_path + '/data/grating/grating_AR_red_2.txt',
                        current_path + '/data/grating/grating_AR_red_3.txt',
                        current_path + '/data/grating/grating_AR_red_4.txt']

line_density_blue = 3559  # lines/mm in blue channel
line_density_red = 2632  # lines/mm in red channel

incident_angle = 61.93  # degree