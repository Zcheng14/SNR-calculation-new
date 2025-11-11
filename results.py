import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

import settings


class Graph:
    def __init__(self, x, y, colors):
        field_points = settings.field_points

        x1 = x[x <= settings.cut]
        x2 = x[x > settings.cut]

        self.fig, [self.blue, self.red] = plt.subplots(nrows=1, ncols=2, figsize=(32, 16), layout="constrained")

        self.x1 = x1
        self.x2 = x2

        for i, (f, c) in enumerate(zip(field_points, colors)):
            self.blue.plot(self.x1, y[i][:len(x1)], label=str(f) + "mm", color=c,
                           linewidth=3)
            self.red.plot(self.x2, y[i][len(x1):], label=str(f) + "mm", color=c,
                          linewidth=3)

    def plot(self):
        return self.fig

    def set_special_labels(self, height):
        h_beta = 486
        if self.x1[-1] >= h_beta >= self.x1[0]:
            self.blue.axvline(x=h_beta, color='black', linestyle='--')
            self.blue.text(((h_beta - self.x1[0]) / (self.x1[-1] - self.x1[0])), height, "H$\\beta$", ha='center',
                           va='center', fontsize=48,
                           transform=self.blue.transAxes)

        o_iii = 501
        if self.x1[-1] >= o_iii >= self.x1[0]:
            self.blue.axvline(x=o_iii, color='black', linestyle='--')
            self.blue.text(((o_iii - self.x1[0]) / (self.x1[-1] - self.x1[0])), height, "[O III]", ha='center',
                           va='center', fontsize=48,
                           transform=self.blue.transAxes)

        h_alpha = 656
        if self.x2[-1] >= h_alpha >= self.x2[0]:
            self.red.axvline(x=h_alpha, color='black', linestyle='--')
            self.red.text(((h_alpha - self.x2[0]) / (self.x2[-1] - self.x2[0])), height, "H$\\alpha$", ha='center',
                          va='center', fontsize=48,
                          transform=self.red.transAxes)

        s_ii = 672
        if self.x2[-1] >= s_ii >= self.x2[0]:
            self.red.axvline(x=s_ii, color='black', linestyle='--')
            self.red.text(((s_ii - self.x2[0]) / (self.x2[-1] - self.x2[0])), height, "[S II]", ha='center',
                          va='center',
                          fontsize=48,
                          transform=self.red.transAxes)

    def set_major_axis(self, length, width):
        self.blue.set_xlim(self.x1[0], self.x1[-1])
        xticks = self.blue.get_xticks()
        self.blue.set_xticks(xticks)

        self.red.set_xlim(self.x2[0], self.x2[-1])
        xticks = self.red.get_xticks()
        self.red.set_xticks(xticks)

        self.blue.tick_params(axis='y', length=length, direction='in', width=width)
        self.blue.tick_params(axis='x', length=length, direction='in', width=width)
        self.red.tick_params(axis='y', length=length, direction='in', width=width)
        self.red.tick_params(axis='x', length=length, direction='in', width=width)

    def set_axis_label(self, y_label, fontsize):
        self.blue.set_xlabel('Wavelength (nm)', fontsize=fontsize)
        self.blue.set_ylabel(y_label, fontsize=fontsize)
        self.blue.tick_params(axis='x', labelsize=fontsize)
        self.blue.tick_params(axis='y', labelsize=fontsize)

        self.red.set_xlabel('Wavelength (nm)', fontsize=fontsize)
        self.red.set_ylabel(y_label, fontsize=fontsize)
        self.red.tick_params(axis='x', labelsize=fontsize)
        self.red.tick_params(axis='y', labelsize=fontsize)

    def set_legend(self, column, fontsize, location):
        self.blue.legend(ncol=column, title='Fiber slit\'s height', fontsize=fontsize, title_fontsize=fontsize,
                         loc=location)

    def set_title(self, blue_title, red_title, fontsize):
        self.blue.set_title(blue_title, fontsize=fontsize)
        self.red.set_title(red_title, fontsize=fontsize)


def print_result_info_multi_wavelength(channel, wavelengths, signal, num_pixel, sky_noise_per_pixel, detector_choice,
                                       darknoise_per_pixel,
                                       temperature_change,
                                       readnoise_per_pixel):
    cut_index = np.where(wavelengths <= settings.cut)[0][-1]
    if channel == 'blue':
        st.write('Blue channel')
        st.markdown(
            f"- Average photon counts from target: {np.mean(signal[:, :cut_index + 1]):.2e} e$^-$")
        st.markdown(f"- Average PSF area: {np.mean(num_pixel[:, :cut_index + 1]):.2e} pixel")
        st.markdown(
            f"- Average sky counts: {np.mean(sky_noise_per_pixel[:, :cut_index + 1]):.2e} e$^-$/1D pixel")
    elif channel == 'red':
        st.write('Red channel')
        st.markdown(
            f"- Average photon counts from target: {np.mean(signal[:, cut_index + 1:]):.2e} e$^-$")
        st.markdown(f"- Average PSF area: {np.mean(num_pixel[:, cut_index + 1:]):.2e} pixel")
        st.markdown(
            f"- Average sky counts: {np.mean(sky_noise_per_pixel[:, cut_index + 1:]):.2e} e$^-$/1D pixel")

    if st.session_state.temperature_change_toggle is True:
        st.markdown(
            f"- Dark counts: {darknoise_per_pixel:.2e} e$^-$/pixel at {settings.detector[detector_choice]['temperature'] + temperature_change} C$^\circ$")
    else:
        st.markdown(
            f"- Dark counts: {darknoise_per_pixel:.2e} e$^-$/pixel at {settings.detector[detector_choice]['temperature']} C$^\circ$")

    st.markdown(f"- Read noise: {readnoise_per_pixel:.2e} e$^-$/pixel")


def print_result_info_single_wavelength(wavelengths, SNR, resolution, signal, num_pixel, sky_noise_per_pixel, detector_choice, darknoise_per_pixel, temperature_change, readnoise_per_pixel):
    def round_up_to_string(data, num_decimal):
        data = np.squeeze(data)
        data = np.around(data, decimals=num_decimal)
        data = ", ".join(str(x) for x in data)
        return data

    field_point_str = ", ".join(str(x) for x in settings.field_points)
    st.header(f'The results are reported for slit heights of [{field_point_str}] mm, respectively.',
              divider='rainbow')

    SNR = round_up_to_string(SNR, 2)
    st.markdown(f"- The SNR of an emission line at {np.squeeze(wavelengths)} nm: [{SNR}]")

    resolution_single = round_up_to_string(resolution, 0)
    st.markdown(f"- The resolution (R) at {np.squeeze(wavelengths)} nm: [{resolution_single}]")

    signal = round_up_to_string(signal, 0)
    st.markdown(f"- Photon counts from target: [{signal}] e$^-$")

    num_pixel = round_up_to_string(num_pixel, 0)
    st.markdown(f"- PSF area: [{num_pixel}] pixel")

    sky_noise_per_pixel = round_up_to_string(sky_noise_per_pixel, 0)
    st.markdown(f"- sky counts: [{sky_noise_per_pixel}] e$^-$/1D pixel")

    if st.session_state.temperature_change_toggle is True:
        st.markdown(
            f"- Dark counts: {darknoise_per_pixel:.2e} e$^-$/pixel at {settings.detector[detector_choice]['temperature'] + temperature_change} C$^\circ$")
    else:
        st.markdown(
            f"- Dark counts: {darknoise_per_pixel:.2e} e$^-$/pixel at {settings.detector[detector_choice]['temperature']} C$^\circ$")
    st.markdown(f"- Read noise: {readnoise_per_pixel:.2e} e$^-$/pixel")

'''
class ThroughputContributorGraph(Graph):
    def __init__(self, x1, x2, contributors, contributor_labels, colors):
        self.fig, [self.blue, self.red] = plt.subplots(nrows=1, ncols=2, figsize=(32, 16), layout="constrained")

        self.x1 = x1
        self.x2 = x2

        ax1_eff.plot(x1, eff_dichoric[: cut[-1] + 1] * 100, label="Dichoric reflection (blue)/transmission (red)",
                     color='#ffa500', linewidth=3)
        ax2_eff.plot(x2, eff_dichoric[cut[-1] + 1:] * 100, label="Dichoric reflection (blue)/transmission (red)",
                     color='#ffa500',
                     linewidth=3)

        ax1_eff.plot(x1, eff_vignetting[0][: cut[-1] + 1] * 100, label="Spectrograph vignetting (30mm)",
                     color='#00FF00', linewidth=3, zorder=100)
        ax2_eff.plot(x2, eff_vignetting[0][cut[-1] + 1:] * 100, label="Spectrograph vignetting (30mm)", color='#00FF00',
                     linewidth=3)

        ax1_eff.plot(x1, eff_vignetting[6][: cut[-1] + 1] * 100, label="Spectrograph vignetting (30mm)",
                     color='#00FF00', linewidth=3, linestyle='dashed', zorder=101)
        ax2_eff.plot(x2, eff_vignetting[6][cut[-1] + 1:] * 100, label="Spectrograph vignetting (30mm)", color='#00FF00',
                     linewidth=3, linestyle='dashed')

        eff_item = [eff_telescope_tp, eff_detector_camera_tp, eff_focal_plane_module, QE]
        labels_red = ["Canon lens throughput", "Nikkor lens throughput", "Focal plane module", "QHY600M QE"]
        labels_blue = ["Canon lens throughput", "Nikkor lens throughput", "Focal plane module", "QHY600M QE"]
        c = ['#FF0000', '#00FFFF', '#0000FF', '#800080']

        for i, contributor in enumerate(contributors):
            if contributor == ''
            ax1_eff.plot(x1, contributors[i][: len(x1)] * 100, label=labels_red[i], color=c[i], linewidth=3,zorder=100)
            ax2_eff.plot(x2, eff_item[i][len(x1):] * 100, color=c[i], linewidth=3)

        index_470 = np.where(wavelength == 470)[0][0]
        index_505 = np.where(wavelength == 505)[0][0]
        index_630 = np.where(wavelength == 630)[0][0]
        index_680 = np.where(wavelength == 680)[0][0]

        ax1_eff.plot(wavelength[index_470:index_505], eff_grating[index_470:index_505] * 100, label="Grating",
                     color='#000000', linewidth=3)
        ax2_eff.plot(wavelength[index_630:index_680], eff_grating[index_630:index_680] * 100, label="Grating",
                     color='#000000', linewidth=3)
'''
