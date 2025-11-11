import streamlit as st
import numpy as np

import settings


class Overview:
    def __init__(self, default_setting, analysis_mode, continuum_mode, spot_size_mode, wavelengths, exposure_time, I, I_continuum, intrinsic_broadening,
                 surface_brightness,
                 telescope_choice, fiber_choice, detector_camera_choice, detector_choice, is_temperature_change,
                 temperature_change):

        self.__basic_overview(analysis_mode, continuum_mode, spot_size_mode, wavelengths, exposure_time, I, I_continuum, intrinsic_broadening, surface_brightness)

        col1, col2 = st.columns(2)

        with col1:
            self.__telescope_overview(telescope_choice)
            self.__fiber_overview(fiber_choice)
            self.__spot_size_overview(default_setting, spot_size_mode)

        with col2:
            self.__detector_camera_overview(detector_camera_choice)
            self.__detector_overview(detector_choice, is_temperature_change, temperature_change)

    @staticmethod
    def __basic_overview(analysis_mode, continuum_mode, spot_size_mode, wavelengths, exposure_time, I, I_continuum, intrinsic_broadening, surface_brightness):
        with st.container(border=True):
            if analysis_mode == "Single wavelength":
                st.markdown(f"- Wavelength: {np.squeeze(wavelengths)} nm")
            st.markdown(f"- Exposure time: {exposure_time} s")
            if analysis_mode == "All wavelength":
                st.markdown(f"- Emission-line flux: {I} erg/cm$^2$/s/arcsec$^2$")
                if continuum_mode == 'Yes':
                    st.markdown(f"- Continuum Flux: {I_continuum} erg/cm$^2$/s/$\AA$/arcsec$^2$")
            elif analysis_mode == "Single wavelength":
                st.markdown(
                    f"- Emission-line flux: {I} erg/cm$^2$/s/arcsec$^2$\n  - Intrinsic broadening: {intrinsic_broadening} km/s")
            st.markdown(f"- Sky background surface brightness: {surface_brightness} mag/arcsec$^2$")
            
    @staticmethod
    def __telescope_overview(telescope_choice):
        with st.container(border=True):
            if telescope_choice is None:
                st.markdown(f"- Telescope: {chr(0x274C)}")
                st.markdown(f"- Focal length: {chr(0x274C)}")
                st.markdown(f"- Diameter: {chr(0x274C)}")
                st.markdown(f"- Throughput: {chr(0x274C)}")
            else:
                st.write(f'Telescope: {telescope_choice}')
                if settings.telescope[telescope_choice]['focal length'] is None:
                    st.markdown(f"- Focal length: {chr(0x274C)}")
                else:
                    st.markdown(f"- Focal length: {settings.telescope[telescope_choice]['focal length']} mm")
                if settings.telescope[telescope_choice]['diameter'] is None:
                    st.markdown(f"- Diameter: {chr(0x274C)}")
                else:
                    st.markdown(f"- Diameter: {round(settings.telescope[telescope_choice]['diameter'], 2)} mm")
                if settings.telescope[telescope_choice]['throughput'] is None:
                    st.markdown(f"- Throughput: {chr(0x274C)}")
                else:
                    st.markdown(f"- Throughput: {chr(0x2705)}")

    @staticmethod
    def __fiber_overview(fiber_choice):
        with st.container(border=True):
            if fiber_choice is None:
                st.markdown(f'Fiber: {chr(0x274C)}')
                st.markdown(f"- Diameter: {chr(0x274C)}")
            else:
                st.write(f'Fiber: {fiber_choice}')
                if settings.fiber[fiber_choice] is None:
                    st.markdown(f"- Diameter: {chr(0x274C)}")
                else:
                    st.markdown(f"- Diameter: {settings.fiber[fiber_choice]} mm")

    @staticmethod
    def __detector_camera_overview(detector_camera_choice):
        with st.container(border=True):
            if detector_camera_choice is None:
                st.markdown(f'Spectrograph camera: {chr(0x274C)}')
                st.markdown(f"- Focal length: {chr(0x274C)}")
                st.markdown(f"- Throughput: {chr(0x274C)}")
                st.markdown(f"- Vignetting: {chr(0x274C)}")
            else:
                st.write(f'Spectrograph camera: {detector_camera_choice}')
                if settings.detector_camera[detector_camera_choice]['focal length'] is None:
                    st.markdown(f"- Focal length: {chr(0x274C)}")
                else:
                    st.markdown(
                        f"- Focal length: {settings.detector_camera[detector_camera_choice]['focal length']} mm")
                if settings.detector_camera[detector_camera_choice]['throughput'] is None:
                    st.markdown(f"- Throughput: {chr(0x274C)}")
                else:
                    st.markdown(f"- Throughput: {chr(0x2705)}")
                if settings.detector_camera[detector_camera_choice]['vignetting'] is None:
                    st.markdown(f"- Vignetting: {chr(0x274C)}")
                else:
                    st.markdown(f"- Vignetting: {chr(0x2705)}")

    @staticmethod
    def __detector_overview(detector_choice, is_temperature_change, temperature_change=None):
        with st.container(border=True):
            if detector_choice is None:
                st.markdown(f'Detector: {chr(0x274C)}')
                st.markdown(f"- Pixel size: {chr(0x274C)}")
                st.markdown(f"- Read noise: {chr(0x274C)}")
                st.markdown(f"- Dark current: {chr(0x274C)}")
                st.markdown(f"- QE: {chr(0x274C)}")
            else:
                st.write(f'Detector: {detector_choice}')
                if settings.detector[detector_choice]['pixel size'] is None:
                    st.markdown(f"- Pixel size: {chr(0x274C)}")
                else:
                    st.markdown(
                        f"- Pixel size: {settings.detector[detector_choice]['pixel size']} mm x {settings.detector[detector_choice]['pixel size']} mm")
                if settings.detector[detector_choice]['readnoise'] is None:
                    st.markdown(f"- Read noise: {chr(0x274C)}")
                else:
                    st.markdown(f"- Read noise: {settings.detector[detector_choice]['readnoise']} e$^-$")
                if settings.detector[detector_choice]['darkcurrent'] is None:
                    st.markdown(f"- Dark current: {chr(0x274C)}")
                else:
                    if is_temperature_change is False:
                        st.markdown(
                            f"- Dark current: {settings.detector[detector_choice]['darkcurrent']:.2e} e$^-$/pixel/s at {settings.detector[detector_choice]['temperature']} C$^\circ$")
                    elif is_temperature_change is True:
                        st.markdown(
                            f"- Dark current: {settings.detector[detector_choice]['darkcurrent']:.2e} e$^-$/pixel/s at {settings.detector[detector_choice]['temperature']} C$^\circ$\n  - Temperature change: {temperature_change}C$^\circ$")
                if settings.detector[detector_choice]['QE'] is None:
                    st.markdown(f"- QE: {chr(0x274C)}")
                else:
                    st.markdown(f"- QE: {chr(0x2705)}")

    @staticmethod
    def __spot_size_overview(default_setting, spot_size_mode):
        with st.container(border=True):
            st.markdown('Spot size')
            #if default_setting is True:
            if spot_size_mode == "Simulation":
                st.markdown(f"- Theoretical calculation: {chr(0x274C)}")
                st.markdown(f"- Simulation: {chr(0x2705)}")
            else:
                st.markdown(f"- Theoretical calculation: {chr(0x2705)}")
                st.markdown(f"- Simulation: {chr(0x274C)}")
