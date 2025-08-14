import settings
import data_files
import math
import numpy as np
import scipy.constants as constant

import streamlit as st

class ThroughputContributors:
    @staticmethod
    def load_eff_focal_plane_module(wavelengths):
        path = settings.common_file_path['focal_plane_module']
        return 1 - data_files.DataTextFile(path).interpolate(wavelengths) * 0.01

    @staticmethod
    def load_eff_telescope_tp(path, wavelengths):
        if path is not None:
            return data_files.DataTextFile(path).interpolate(wavelengths)
        else:
            return np.ones_like(wavelengths)

    @staticmethod
    def load_eff_fiber_transmission():
        return settings.eff_fiber_transmission

    @staticmethod
    def load_eff_collimator():
        return settings.eff_collimator

    @staticmethod
    def load_eff_dichoric(wavelengths):
        path_blue = settings.common_file_path['dichoric_reflection']
        path_red = settings.common_file_path['dichoric_transmission']
        return data_files.TwoDataTextFiles(path_blue, path_red).interpolate(wavelengths) * 0.01

    @staticmethod
    def load_eff_grating(wavelengths):
        blue_path = settings.common_file_path['blue_grating']
        red_path = settings.common_file_path['red_grating']
        return data_files.TwoDataTextFiles(blue_path, red_path).interpolate(wavelengths) * 0.01

    @staticmethod
    def load_eff_AR_coating_grating(wavelengths):
        return data_files.load_AR_coating(wavelengths)

    @staticmethod
    def load_eff_vignetting(path, wavelengths, detector_camera_choice):
        if path is not None:
            return data_files.VignettingFiles(path).interpolate(wavelengths, detector_camera_choice) * 0.01
        else:
            return np.ones([len(settings.field_points), len(wavelengths)])

    @staticmethod
    def load_eff_detector_camera_tp(path, wavelengths):
        if path is not None:
            return data_files.DataTextFile(path).interpolate(wavelengths)
        else:
            return np.ones_like(wavelengths)

    @staticmethod
    def load_QE(path, wavelengths):
        if path is not None:
            return data_files.DataTextFile(path).interpolate(wavelengths)
        else:
            return np.ones_like(wavelengths)


class Formula:
    @staticmethod
    def cos_gamma(field_point):
        return np.cos(field_point / settings.f_collimator)

    @staticmethod
    def cos_beta(line_density, wavelength, field_point):
        cos_gamma = Formula.cos_gamma(field_point)
        incident_angle = math.radians(settings.incident_angle)
        LHS = line_density * (wavelength * 1e-6) / cos_gamma
        LHS = LHS - np.sin(incident_angle)
        LHS = np.arcsin(LHS)
        cos_beta = np.cos(LHS)
        return cos_beta

    @staticmethod
    def resolution(wavelengths, detector_camera_choice, image_size, line_density, field_point):
        focal_length = settings.detector_camera[detector_camera_choice]['focal length']
        spot_size = image_size
        cos_beta = Formula.cos_beta(line_density, wavelengths, field_point)
        beta = np.arccos(cos_beta)
        cos_gamma = Formula.cos_gamma(field_point)
        dl_db = focal_length / (np.cos(beta - incident_angle))**2
        return (line_density * (wavelengths / 1000) * dl_db) / (spot_size * cos_beta * cos_gamma)


def find_spot_size(analysis_mode, default_setting, default_system, detector_camera_choice, d_fiber, wavelengths):
    if default_setting:
        path = settings.default_system[default_system]["spot_path"]
        return data_files.SpotSizeFiles(path).interpolate(wavelengths, default_system)
    else:
        spot_size = np.zeros((len(settings.field_points), len(wavelengths)))
        for i in range(len(settings.field_points)):
            focal_length = settings.detector_camera[detector_camera_choice]['focal length']
            f_collimator = settings.f_collimator
            cos_in_angle = math.cos(math.radians(settings.incident_angle))

            if analysis_mode == "All wavelength":
                cut_index = np.where(wavelengths <= settings.cut)[0][-1]
                wavelengths_blue = wavelengths[:cut_index + 1]
                cos_out_angle_blue = Formula.cos_beta(settings.line_density_blue, wavelengths_blue,
                                                      settings.field_points[i])
                wavelengths_red = wavelengths[cut_index + 1:]
                cos_out_angle_red = Formula.cos_beta(settings.line_density_red, wavelengths_red,
                                                     settings.field_points[i])
                cos_out_angle = np.concatenate((cos_out_angle_blue, cos_out_angle_red), axis=None)

            elif analysis_mode == "Single wavelength":
                if wavelengths <= settings.cut:
                    cos_out_angle = Formula.cos_beta(settings.line_density_blue, wavelengths, settings.field_points[i])
                else:
                    cos_out_angle = Formula.cos_beta(settings.line_density_red, wavelengths, settings.field_points[i])

            spot_size[i] = d_fiber * (focal_length / f_collimator) * (cos_in_angle / cos_out_angle)
    return spot_size * 1e3


def cal_num_pixel(analysis_mode, spot_size, detector_choice, wavelength, intrinsic_broadening):
    if analysis_mode == "All wavelength":
        spot_area = (spot_size / 2) ** 2 * constant.pi
        pixel_area = settings.detector[detector_choice]["pixel size"] ** 2
        num_pixel = spot_area / pixel_area
    elif analysis_mode == "Single wavelength":
        wavelength_width = intrinsic_broadening / settings.speed_of_light * (wavelength / 1000)
        spot_area = np.sqrt(spot_size ** 2 + wavelength_width ** 2) * spot_size / 4 * constant.pi
        pixel_area = settings.detector[detector_choice]["pixel size"] ** 2
        num_pixel = spot_area / pixel_area
    return num_pixel


class Analysis:
    def __init__(self, analysis_mode, continuum_mode, default_setting, default_system, telescope_choice, d_fiber, detector_camera_choice,
                detector_choice,
                wavelengths, intrinsic_broadening=None):
        self.analysis_mode = analysis_mode
        self.telescope_choice = telescope_choice
        self.d_fiber = d_fiber
        self.detector_camera_choice = detector_camera_choice
        self.detector_choice = detector_choice
        self.wavelengths = wavelengths
        self.continuum_mode = continuum_mode

        # eff
        self.eff_focal_plane_module = ThroughputContributors.load_eff_focal_plane_module(wavelengths)

        path = settings.telescope[telescope_choice]['throughput']
        self.eff_telescope_tp = ThroughputContributors.load_eff_telescope_tp(path, wavelengths)

        self.eff_fiber_transmission = ThroughputContributors.load_eff_fiber_transmission()

        self.eff_collimator = ThroughputContributors.load_eff_collimator()

        self.eff_dichoric = ThroughputContributors.load_eff_dichoric(wavelengths)

        self.eff_grating = ThroughputContributors.load_eff_grating(wavelengths)

        self.eff_AR_coating_grating = ThroughputContributors.load_eff_AR_coating_grating(wavelengths)

        path = settings.detector_camera[detector_camera_choice]['vignetting']['path']
        self.eff_vignetting = ThroughputContributors.load_eff_vignetting(path, wavelengths, detector_camera_choice)

        path = settings.detector_camera[detector_camera_choice]['throughput']
        self.eff_detector_camera_tp = ThroughputContributors.load_eff_detector_camera_tp(path, wavelengths)

        path = settings.detector[detector_choice]['QE']
        self.QE = ThroughputContributors.load_QE(path, wavelengths)

        self.spot_size = find_spot_size(analysis_mode, default_setting, default_system, detector_camera_choice, d_fiber, wavelengths)
        self.num_pixel = cal_num_pixel(analysis_mode, self.spot_size, detector_choice, wavelengths,
                                       intrinsic_broadening)

    def cal_signal(self, I, exposure_time):
        self.eff = np.zeros_like(self.eff_vignetting)
        for i in range(len(settings.field_points)):
            self.eff[i] = (self.eff_focal_plane_module *
                           self.eff_telescope_tp *
                           self.eff_fiber_transmission ** 2 *
                           self.eff_collimator *
                           self.eff_dichoric *
                           self.eff_grating *
                           self.eff_AR_coating_grating *
                           self.eff_vignetting[i] *
                           self.eff_detector_camera_tp)

        radius_telescope = settings.telescope[self.telescope_choice]['diameter'] * 0.1 / 2
        self.A = constant.pi * radius_telescope ** 2  # cm^2

        radius_fiber = self.d_fiber / 2
        telescope_focal = settings.telescope[self.telescope_choice]["focal length"]
        self.solid_angle = constant.pi * (radius_fiber / telescope_focal) ** 2 * (
                206264.5 ** 2)  # 206264.5 radian^2 to arcsec^2

        photon_counts = np.zeros_like(self.eff_vignetting)
        signal = np.zeros_like(self.eff_vignetting)
        for i in range(len(settings.field_points)):
            total_energy = I * self.A * self.solid_angle * exposure_time  # erg

            frequency = settings.speed_of_light / (self.wavelengths * 1e-9)
            photon_energy = (settings.h * frequency) / 1e-7

            photon_counts[i] = total_energy / photon_energy * self.eff[i]

            signal[i] = photon_counts[i] * self.QE

        return signal

    def __set_line_density(self):
        if self.analysis_mode == "All wavelength":
            line_density = np.zeros_like(self.wavelengths)

            cut_index = np.where(self.wavelengths <= settings.cut)[0][-1]
            line_density[:cut_index + 1] = settings.line_density_blue
            line_density[cut_index + 1:] = settings.line_density_red

        elif self.analysis_mode == "Single wavelength":
            if self.wavelengths <= 510:
                line_density = settings.line_density_blue
            else:
                line_density = settings.line_density_red

        return line_density

    def __cal_dispersion(self, field_points, line_density):
        dispersion = np.zeros_like(self.eff_vignetting)  # d\lambda/dl
        for i in range(len(field_points)):
            cos_gamma = Formula.cos_gamma(field_points)
            cos_beta = Formula.cos_beta(line_density, self.wavelengths, field_points[i])
            detectr_focal = settings.detector_camera[self.detector_camera_choice]["focal length"]
            dispersion[i] = cos_gamma[i] * cos_beta / (detectr_focal * line_density)
        return dispersion

    def cal_continuum(self, I_continuum, exposure_time):
        self.eff = np.zeros_like(self.eff_vignetting)
        for i in range(len(settings.field_points)):
            self.eff[i] = (self.eff_focal_plane_module *
                           self.eff_telescope_tp *
                           self.eff_fiber_transmission ** 2 *
                           self.eff_collimator *
                           self.eff_dichoric *
                           self.eff_grating *
                           self.eff_AR_coating_grating *
                           self.eff_vignetting[i] *
                           self.eff_detector_camera_tp)

        radius_telescope = settings.telescope[self.telescope_choice]['diameter'] * 0.1 / 2
        self.A = constant.pi * radius_telescope ** 2  # cm^2

        radius_fiber = self.d_fiber / 2
        telescope_focal = settings.telescope[self.telescope_choice]["focal length"]
        self.solid_angle = constant.pi * (radius_fiber / telescope_focal) ** 2 * (
                206264.5 ** 2) # 206264.5 radian^2 to arcsec^2
        photon_counts_con = np.zeros_like(self.eff_vignetting)
        continuum = np.zeros_like(self.eff_vignetting)
        I_continuum_erg = 10**(I_continuum/(-2.5))*3631/(3.34*10**4)/5000**2 # transfer AB mag to flux, assume λ=5000Å
        field_points = settings.field_points
        line_density = self.__set_line_density()
        dispersion = self.__cal_dispersion(field_points, line_density)
        pixel_size_1d = settings.detector[self.detector_choice]["pixel size"]
        for i in range(len(settings.field_points)):
            total_energy = I_continuum_erg * self.A * self.solid_angle * exposure_time  # erg
            frequency = settings.speed_of_light / (self.wavelengths * 1e-9)
            photon_energy = (settings.h * frequency) / 1e-7

            photon_counts_con[i] = total_energy / photon_energy * self.eff[i] * dispersion[i] * (pixel_size_1d * 1e4)
            continuum[i] = photon_counts_con[i] * self.QE
        return continuum

    def cal_sky_background(self, surface_brightness, exposure_time):
        field_points = settings.field_points

        manga_count = 90 * 10 ** ((surface_brightness - 21.5) / (-2.5))

        manga_exposuretime = 900
        manga_disperson = 1.4
        sdss_mirror = ((2.5 / 2) ** 2 * constant.pi - (1.3 / 2) ** 2 * constant.pi) * 1e4
        manga_fiberarea = constant.pi
        manga_efficiency = 0.32

        # electrons/s/Angstrom/arcsec^2/cm^2
        sky_brightness = manga_count / (
                manga_exposuretime * manga_disperson * sdss_mirror * manga_fiberarea * manga_efficiency)

        line_density = self.__set_line_density()
        dispersion = self.__cal_dispersion(field_points, line_density)

        pixel_size_1d = settings.detector[self.detector_choice]["pixel size"]
        sky_brightness = sky_brightness * dispersion * (pixel_size_1d * 1e4)  # electrons/s/pixel/arcsec^2/cm^2

        self.sky_noise_per_pixel = sky_brightness * exposure_time * self.solid_angle * self.A * self.eff * self.QE  # number of electrons

        sky_noise = self.sky_noise_per_pixel * (self.spot_size / pixel_size_1d)
        return sky_noise

    def cal_read_noise(self):
        self.readnoise_per_pixel = settings.detector[self.detector_choice]["readnoise"] ** 2
        readnoise = self.readnoise_per_pixel * self.num_pixel
        return readnoise

    def cal_dark_noise(self, is_temperature_change, exposure_time, temperature_change=None):
        darkcurrent = settings.detector[self.detector_choice]["darkcurrent"]
        if is_temperature_change is True:
            self.darknoise_per_pixel = darkcurrent * (2 ** (temperature_change / 10)) * exposure_time
        else:
            self.darknoise_per_pixel = darkcurrent * exposure_time
        darknoise = self.darknoise_per_pixel * self.num_pixel
        return darknoise

    @staticmethod
    def cal_SNR(signal, sky_noise, readnoise, darknoise, continuum, continuum_mode):
        if continuum_mode == 'No':
            SNR = signal / np.sqrt(signal + sky_noise + readnoise + darknoise)
        else:
            SNR = signal / np.sqrt(signal + sky_noise + readnoise + darknoise + continuum)
        return SNR

    def cal_resolution(self):
        line_density = self.__set_line_density()
        resolution = np.zeros_like(self.spot_size)
        for i, f in enumerate(settings.field_points):
            resolution[i] = Formula.resolution(self.wavelengths, self.detector_camera_choice, self.spot_size[i], line_density, f)

        return resolution
