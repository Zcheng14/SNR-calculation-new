# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import scipy.constants as constant
current_path = os.getcwd()
#%% 
"""
This is a demo script for exposure timme calculation with multiple sources.
"""
#%%
class Telescope():
    def __init__(self, focal_length=392.5, diameter=392.5 / 2.9):
        self.focal_length = focal_length # mm
        self.diameter = diameter # mm
        self.A = self.area()
        self.solid_angle = np.pi * (self.diameter / (2 * self.focal_length)) ** 2 * (
            180 / np.pi * 3600) ** 2  # arcsec^2
        self.throughput = self.read_throughput()  # throughput data
        
    def area(self):
        return np.pi * (self.diameter * 0.1 / 2) ** 2 # cm^2
    
    def read_throughput(self):
        file_path = current_path + "/data/throughput/canon_throughput.txt"
        data = np.loadtxt(file_path)  # wavelength (nm), throughput
        return data

class DetectorCamera():
    def __init__(self, camera_type="Custom design lens"):
        self.camera_type = camera_type
        self.load_camera_data()
        self.throughput = self.read_throughput()  # throughput data
        self.vignetting = self.read_vignetting()
        self.spot_size = self.read_spot_size()
    
    def load_camera_data(self):
        if self.camera_type == "Custom design lens":
            self.focal_length = 99.5  # mm
            self.wavelength_range = np.concatenate((np.arange(465, 506), np.arange(625, 681)), axis=None)
            self.blue_wavelength_panel = np.arange(465,506)
            self.red_wavelength_panel = np.arange(625,681)
            self.emergent_angle0_blue = 59.89 #degree
            self.emergent_angle0_red = 59.61 #degree
        elif self.camera_type == "Nikkor lens":
            self.focal_length = 59.67  # mm
            self.wavelength_range = np.concatenate((np.arange(460, 511), np.arange(610, 691)), axis=None)
            self.blue_wavelength_panel = np.arange(460,511)
            self.red_wavelength_panel = np.arange(610,691)
            self.emergent_angle0_blue = 61.93 #degree
            self.emergent_angle0_red = 61.93 #degree
        else:
            print("No data available for this camera type.")
            self.focal_length = None
            self.wavelength_range = None
            self.emergent_angle0_blue = None
            self.emergent_angle0_red = None
    
    def read_throughput(self):
        if self.camera_type == "Custom design lens":
            file_path = current_path + "/data/throughput/customlens_throughput.txt"
            data = np.loadtxt(file_path)  # wavelength (nm), throughput
        elif self.camera_type == "Nikkor lens":
            file_path = current_path + "/data/throughput/nikkor_throughput.txt"
            data = np.loadtxt(file_path)  # wavelength (nm), throughput
        else:
            print("No throughput data available for this camera type.")
            data = None
        return data
    
    def read_vignetting(self):
        if self.camera_type == "Custom design lens":
            file_path = current_path + "/data/vignetting/vignetting_custom.txt"
            data = np.loadtxt(file_path)
        elif self.camera_type == "Nikkor lens":
            file_path = current_path + "/data/vignetting/vignetting_Nikkor.txt"
            data = np.loadtxt(file_path)
        else:
            print("No vignetting data available for this camera type.")
            data = None
        return data

    def read_spot_size(self):
        if self.camera_type == "Custom design lens":
            file_path = current_path + "/data/spot_size/spot_custom.txt"
            data = np.loadtxt(file_path)
        elif self.camera_type == "Nikkor lens":
            file_path = current_path + "/data/spot_size/spot_nikkor.txt"
            data = np.loadtxt(file_path)
        else:
            print("No spot size data available for this camera type.")
            data = None
        return data
class Detector():
    def __init__(self, detector_type='QHY461'):
        self.detector_type = detector_type
        self.QE = self.read_QE()
        self.load_detector_data()
    
    def read_QE(self):
        if self.detector_type == 'QHY461':
            file_path = current_path + "/data/QE/QHY461_QE.txt"
            data = np.loadtxt(file_path)  # wavelength (nm), QE
        elif self.detector_type == 'QHY600':
            file_path = current_path + "/data/QE/QHY600_QE.txt"
            data = np.loadtxt(file_path)  # wavelength (nm), QE
        else:
            print("No QE data available for this detector type.")
            data = None
        return data
        
    def load_detector_data(self):
        if self.detector_type == 'QHY461':
            self.pixel_size = 3.76  # um
            self.readnoise = 1.32
            self.darkcurrent = 2.667e-4
            self.temperature = -20
        elif self.detector_type == 'QHY600':
            self.pixel_size = 3.76  # um
            self.readnoise = 1.683
            self.darkcurrent = 0.0011
            self.temperature = -30
        else:
            print("No data available for this detector type.")
            self.pixel_size = None
            self.readnoise = None
            self.darkcurrent = None
            self.temperature = None

class CommonSettings():
    def __init__(self):
        self.focal_plane_module = current_path + "/data/focal_plane_module/focal_plane_module.txt"
        self.dichoric_reflection = current_path + "/data/dichoric/dichoric_reflection.txt"
        self.dichoric_transmission = current_path + "/data/dichoric/dichoric_transmission.txt"
        self.blue_grating = current_path + "/data/grating/blue_grating_eff.txt"
        self.red_grating = current_path + "/data/grating/red_grating_eff.txt"
        self.cut = 510
        self.field_points = np.array([0, 5, 10, 15, 20, 25, 30]) # mm
        self.fiber_core_diameter = 0.05 # mm
        self.eff_collimator = 0.9  # assume 90% reflexivity
        self.f_collimator = 200  # mm
        self.eff_fiber_transmission = 0.90  # assume 10% reflectivity
        self.line_density_blue = 3559  # lines/mm in blue channel
        self.line_density_red = 2632  # lines/mm in red channel
        self.incident_angle = 61.93  # degree

        self.blue_AR_coating_paths = [current_path + '/data/grating/grating_AR_blue_1.txt',
                         current_path + '/data/grating/grating_AR_blue_2.txt',
                         current_path + '/data/grating/grating_AR_blue_3.txt',
                         current_path + '/data/grating/grating_AR_blue_4.txt']
        self.red_AR_coating_paths = [current_path + '/data/grating/grating_AR_red_1.txt',
                        current_path + '/data/grating/grating_AR_red_2.txt',
                        current_path + '/data/grating/grating_AR_red_3.txt',
                        current_path + '/data/grating/grating_AR_red_4.txt']

        self.speed_of_light = constant.speed_of_light  # ~3e8 m/s
        self.h = constant.h  # ~6.626e-34 J/Hz

def tot_efficiency(common_settings, telescope, camera, detector):
    # A function to calculate the total efficiency of the system
    """
    Parameters:
    -----------
    common_settings: CommonSettings object
    telescope: Telescope object
    camera: DetectorCamera object
    detector: Detector object
    Returns:
    -----------/data/spot_size/spot_size_Nikkor.txt
    wavelengths: array of wavelengths (nm)
    total_eff: array of total efficiency corresponding to wavelengths
    """
    wavelengths = camera.wavelength_range
    blue_panel = camera.blue_wavelength_panel
    red_panel = camera.red_wavelength_panel

    # focal plane module efficiency
    fpm = np.loadtxt(common_settings.focal_plane_module)
    fpm_wave = fpm[:, 0]
    fpm_eff = fpm[:, 1] * 0.01
    fpm_interp = 1- np.interp(wavelengths, fpm_wave, fpm_eff)

    # telescope throughput
    tel_wave = telescope.throughput[:, 0]
    tel_eff = telescope.throughput[:, 1]
    tel_interp = np.interp(wavelengths, tel_wave, tel_eff)

    # fiber
    fiber_eff = common_settings.eff_fiber_transmission

    # collimator
    collimator_eff = common_settings.eff_collimator

    # grating efficiency
    blue_grating = np.loadtxt(common_settings.blue_grating)
    red_grating = np.loadtxt(common_settings.red_grating)
    blue_wave = blue_grating[:, 0]
    blue_eff = blue_grating[:, 1] * 0.01
    red_wave = red_grating[:, 0]
    red_eff = red_grating[:, 1] * 0.01
    blue_grating_interp = np.interp(blue_panel, blue_wave, blue_eff)
    red_grating_interp = np.interp(red_panel, red_wave, red_eff)
    grating_interp = np.concatenate((blue_grating_interp, red_grating_interp), axis=None)

    # AR coating efficiency
    ar_blue_wave = np.arange(452, 521)
    ar_red_wave = np.arange(610, 701)
    ar_blue_eff = []
    for path in common_settings.blue_AR_coating_paths:
        data = np.loadtxt(path)
        data_interp = np.interp(ar_blue_wave, data[:, 0], data[:, 1])
        ar_blue_eff.append(data_interp)
    ar_blue_mean = np.mean(ar_blue_eff, axis=0) * 0.01
    ar_red_eff = []
    for path in common_settings.red_AR_coating_paths:
        data = np.loadtxt(path)
        data_interp = np.interp(ar_red_wave, data[:, 0], data[:, 1])
        ar_red_eff.append(data_interp)
    ar_red_mean = np.mean(ar_red_eff, axis=0) * 0.01
    ar_blue_interp = 1-np.interp(blue_panel, ar_blue_wave, ar_blue_mean)
    ar_red_interp = 1-np.interp(red_panel, ar_red_wave, ar_red_mean)
    ar_interp = np.concatenate((ar_blue_interp, ar_red_interp), axis=None)

    # vignetting
    vnet_wave = wavelengths
    vnet_eff = camera.vignetting * 0.01
    vnet_interp = vnet_eff

    # camera throughput
    cam_wave = camera.throughput[:, 0]
    cam_eff = camera.throughput[:, 1]
    cam_interp = np.interp(wavelengths, cam_wave, cam_eff)

    # Reflectivity and transmission of dichoric
    dr = np.loadtxt(common_settings.dichoric_reflection)
    dt = np.loadtxt(common_settings.dichoric_transmission)
    dr_wave_blue = dr[:,0]
    dr_y_blue = dr[:,1]*0.01

    dt_wave_red = dt[:,0]
    dt_y_red = dt[:,1]*0.01

    dr_interp_blue = np.interp(blue_panel, dr_wave_blue, dr_y_blue)
    dt_interp_red = np.interp(red_panel, dt_wave_red, dt_y_red)
    dichoric_interp = np.concatenate((dr_interp_blue, dt_interp_red), axis=None)


    # QE
    qe_wave = detector.QE[:, 0]
    qe_y = detector.QE[:, 1]
    qe_interp = np.interp(wavelengths, qe_wave, qe_y)

    # total efficiency
    total_eff = np.zeros_like(vnet_eff, dtype=float)

    for i in range(len(total_eff)):
        if camera.camera_type == "Custom design lens":
            total_eff[i] = (fpm_interp *
                            tel_interp *
                            fiber_eff**2 *
                            collimator_eff *
                            dichoric_interp *
                            grating_interp *
                            ar_interp *
                            vnet_interp[i] *
                            cam_interp *
                            qe_interp)
            
        elif camera.camera_type == "Nikkor lens":
            total_eff[i] = (1 *
                            tel_interp *
                            fiber_eff**2 *
                            collimator_eff *
                            dichoric_interp *
                            grating_interp *
                            ar_interp *
                            vnet_interp[i] *
                            cam_interp *
                            qe_interp)
    return wavelengths, total_eff

def dispersion(common_settings, camera):
    """
    This function calculates the dispersion (mm/nm) at different field points and wavelengths.
    Parameters:
    -----------
    common_settings: CommonSettings object
    camera: Camera object
    Returns:
    wavelengths: array of wavelengths (nm)
    dispersion: 2D array of dispersion (mm/nm) for each field point and wavelength
    """
    wavelengths = camera.wavelength_range
    emergent_angle0_blue = camera.emergent_angle0_blue
    emergent_angle0_red = camera.emergent_angle0_red
    line_density_blue = common_settings.line_density_blue
    line_density_red = common_settings.line_density_red
    field_points = common_settings.field_points
    incident_angle = math.radians(common_settings.incident_angle)

    e_angle0 = np.zeros_like(wavelengths)
    line_density = np.zeros_like(wavelengths)
    cut_index = np.where(wavelengths <= common_settings.cut)[0][-1]
    e_angle0[:cut_index+1] = emergent_angle0_blue
    e_angle0[cut_index+1:] = emergent_angle0_red
    e_angle0 = np.radians(e_angle0) # IMPORTANT: convert to radian
    line_density[:cut_index+1] = line_density_blue
    line_density[cut_index+1:] = line_density_red

    dispersion = np.zeros((len(field_points), len(wavelengths)))

    for i in range(len(field_points)):
        cos_gamma = np.cos(field_points[i] / common_settings.f_collimator)
        LHS = line_density * (wavelengths * 1e-6) / cos_gamma
        LHS = LHS - np.sin(incident_angle)
        LHS = np.arcsin(LHS)
        cos_beta = np.cos(LHS)
        beta = np.arccos(cos_beta)
        dl_db = camera.focal_length / (np.cos(beta-e_angle0))**2 / cos_gamma # mm/rad
        dispersion[i] = cos_gamma * cos_beta / (dl_db * line_density)
    return wavelengths, dispersion


def cal_signal(I, exposure_time, total_eff, telescope, wavelengths, common_settings):
    A = telescope.A  # cm^2
    solid_angle = telescope.solid_angle  # arcsec^2
    telescope_focal = telescope.focal_length  # mm
    signal = np.zeros_like(total_eff)
    for i in range(len(total_eff)):
        total_energy = I * A * solid_angle * exposure_time  # erg
        frequency = common_settings.speed_of_light / (wavelengths * 1e-9)
        photon_energy = (common_settings.h * frequency) / 1e-7
        signal[i] = total_energy / photon_energy * total_eff[i]
    return signal

def number_of_pixels(camera, detector, analysis_mode, intrinsic_broadening=None, wavelength=None):
    # intrinsic_broadening: km/s
    # wavelength: nm
    spot_size = camera.spot_size  # um
    pixel_size = detector.pixel_size  # um
    if analysis_mode == 'All wavelength':
        spot_area = (spot_size / 2) ** 2 * constant.pi
        pixel_area = pixel_size ** 2
        n_pixels = spot_area / pixel_area
    elif analysis_mode == 'Single wavelength':
        if intrinsic_broadening is None:
            print("Please provide intrinsic_broadening for single wavelength analysis.")
        else:
            width = intrinsic_broadening/(constant.speed_of_light / 1000) * (wavelength / 1000) # note: convert speed of light to km/s and wavelength to um
            

        

    pass
# %% test
telescope = Telescope()
camera = DetectorCamera(camera_type="Custom design lens")
detector = Detector(detector_type='QHY461')
common_settings = CommonSettings()
# %% check total efficiency
wavelengths, total_eff = tot_efficiency(common_settings, telescope, camera, detector)
fiber_slit_height = common_settings.field_points
color_list = ['red','orange','green','cyan','blue','brown','k']
plt.figure(figsize=(16,8),dpi=200)
# 2 panels for blue and red
plt.subplot(1,2,1)
for i in range(len(total_eff)):
    plt.plot(wavelengths[wavelengths<=common_settings.cut],total_eff[i][wavelengths<=common_settings.cut]*100,label = str(fiber_slit_height[i])+'mm', color=color_list[i])
plt.axvline(486,color='k',linestyle='--')
plt.axvline(501,color='k',linestyle='--')
plt.text(483,20,r'H$\beta$',fontsize=20) # Hb line blue channel
plt.text(495,20,r'[OIII]',fontsize=20) # OIII line blue channel
plt.xlim(wavelengths[wavelengths<=common_settings.cut][0],wavelengths[wavelengths<=common_settings.cut][-1]) # blue channel
plt.minorticks_on()
plt.xlabel(r"Wavelength(nm)",fontsize=20)
plt.ylabel(r"Throughput(%)",fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.title(r'Throughput of blue channel', fontsize=20)
plt.legend(fontsize=15,loc='upper left')

plt.subplot(1,2,2)
for i in range(len(total_eff)):
    plt.plot(wavelengths[wavelengths>common_settings.cut],total_eff[i][wavelengths>common_settings.cut]*100,label = str(fiber_slit_height[i])+'mm', color=color_list[i])
plt.axvline(656,color='k',linestyle='--')
plt.axvline(672,color='k',linestyle='--')
plt.text(653,17,r'H$\alpha$',fontsize=20) # Ha line red channel
plt.text(669,17,r'[SII]',fontsize=20) # SII line red channel
plt.xlim(wavelengths[wavelengths>common_settings.cut][0],wavelengths[wavelengths>common_settings.cut][-1]) # red channel
plt.minorticks_on()
plt.xlabel(r"Wavelength(nm)",fontsize=20)
plt.ylabel(r"Throughput(%)",fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.title(r'Throughput of red channel', fontsize=20)
plt.show()

# %%
