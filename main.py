import streamlit as st

from results import Graph, print_result_info_multi_wavelength, print_result_info_single_wavelength
from analyses import Analysis
from input_panels import initialize, Panel
from overviews import Overview
import numpy as np

st.set_page_config(layout="wide")

def show_result():
    analysis = Analysis(analysis_mode, continuum_mode, spot_size_mode, default_setting, default_system, telescope_choice, d_fiber, detector_camera_choice,
                        detector_choice, wavelengths, intrinsic_broadening)

    signal = analysis.cal_signal(I, exposure_time, detector_camera_choice)
    continuum = analysis.cal_continuum(I_continuum, exposure_time, detector_camera_choice)
    sky_background = analysis.cal_sky_background(surface_brightness, exposure_time)
    read_noise = analysis.cal_read_noise()
    dark_noise = analysis.cal_dark_noise(is_temperature_change, exposure_time, temperature_change)
    SNR = analysis.cal_SNR(signal, sky_background, read_noise, dark_noise, continuum, continuum_mode)
    resolution = analysis.cal_resolution()

    spot_size = analysis.spot_size
    #kk = spot_size[:, [20, 41, -35, -19]]
    #np.savetxt("bb.txt", kk)

    num_pixel = analysis.num_pixel

    sky_noise_per_pixel = analysis.sky_noise_per_pixel
    darknoise_per_pixel = analysis.darknoise_per_pixel
    readnoise_per_pixel = analysis.readnoise_per_pixel


    if analysis_mode == "All wavelength":
        colors = ['#FF0000', '#ffa500', '#00FF00', '#00FFFF', '#0000FF', '#800080', '#000000']

        SNR_graph = Graph(wavelengths, SNR, colors)
        SNR_graph.set_special_labels(0.1)
        SNR_graph.set_major_axis(20, 1)
        SNR_graph.set_axis_label("SNR", 48)
        SNR_graph.set_legend(2, 36, 'upper left')
        SNR_graph.set_title("SNR of blue channel", "SNR of red channel", 52)

        resolution_graph = Graph(wavelengths, resolution, colors)
        resolution_graph.set_special_labels(0.1)
        resolution_graph.set_major_axis(20, 1)
        resolution_graph.set_axis_label("Resolution", 48)
        resolution_graph.set_legend(2, 36, 'upper left')
        resolution_graph.set_title("Resolution of blue channel", "Resolution of red channel", 52)

        throughput = analysis.eff * analysis.QE * 100
        throughput_graph = Graph(wavelengths, throughput, colors)
        throughput_graph.set_special_labels(0.1)
        throughput_graph.set_major_axis(20, 1)
        throughput_graph.set_axis_label("Throughput (%)", 48)
        throughput_graph.set_legend(2, 36, 'upper left')
        throughput_graph.set_title("Throughput of blue channel", "Throughput of red channel", 52)

        ## Show calculation
        with st.expander("Result", expanded=True):
            blue_result, red_result = st.columns(2)

            st.pyplot(SNR_graph.plot())
            st.pyplot(resolution_graph.plot())
            st.pyplot(throughput_graph.plot())

            with blue_result:
                print_result_info_multi_wavelength('blue',
                                                   wavelengths,
                                                   signal,
                                                   num_pixel,
                                                   sky_noise_per_pixel,
                                                   detector_choice,
                                                   darknoise_per_pixel,
                                                   temperature_change,
                                                   readnoise_per_pixel)
            with red_result:
                print_result_info_multi_wavelength('red',
                                                   wavelengths,
                                                   signal,
                                                   num_pixel,
                                                   sky_noise_per_pixel,
                                                   detector_choice,
                                                   darknoise_per_pixel,
                                                   temperature_change,
                                                   readnoise_per_pixel)

    elif analysis_mode == "Single wavelength":
        with st.expander("Result", expanded=True):
            print_result_info_single_wavelength(wavelengths, SNR, resolution, signal, num_pixel, sky_noise_per_pixel, detector_choice, darknoise_per_pixel, temperature_change, readnoise_per_pixel)


if __name__ == "__main__":
    initialize()

    with st.sidebar:
        input = Panel()

        analysis_mode = input.analysis_mode
        continuum_mode = input.continuum_mode
        spot_size_mode = input.spot_size_mode
        wavelengths = input.wavelengths
        I = input.I
        I_continuum = input.I_continuum
        exposure_time = input.exposure_time
        surface_brightness = input.surface_brightness

        if analysis_mode == "Single wavelength":
            intrinsic_broadening = input.intrinsic_broadening
        else:
            intrinsic_broadening = None

        default_setting = input.default_setting
        if default_setting is True:
            default_system = input.default_system
        else:
            default_system = None

        telescope_choice = input.telescope_choice
        fiber_choice = input.fiber_choice
        d_fiber = input.d_fiber
        detector_camera_choice = input.detector_camera_choice
        detector_choice = input.detector_choice
        
        is_temperature_change = input.is_temperature_change
        if is_temperature_change is True:
            temperature_change = input.temperature_change
        else:
            temperature_change = None

    with st.expander("Setting overview", expanded=True):
        setting_overview = Overview(default_setting,
                                    analysis_mode,
                                    continuum_mode,
                                    spot_size_mode,
                                    wavelengths,
                                    exposure_time,
                                    I,
                                    I_continuum,
                                    intrinsic_broadening,
                                    surface_brightness,
                                    telescope_choice,
                                    fiber_choice,
                                    detector_camera_choice,
                                    detector_choice,
                                    is_temperature_change,
                                    temperature_change)

    not_ready = None
    if telescope_choice is None or fiber_choice is None or detector_camera_choice is None or detector_choice is None:
        not_ready = True
    else:
        not_ready = False
    st.button('Analyse!', on_click=show_result, disabled=not_ready, type="primary")
