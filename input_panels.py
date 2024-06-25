import streamlit as st
import numpy as np

import settings

if "temperature_change_toggle" not in st.session_state:
    st.session_state.temperature_change_toggle = False
if "default_index" not in st.session_state:
    st.session_state.default_index = 0
if "disable_all" not in st.session_state:
    st.session_state.disable_all = True
if "default_setting" not in st.session_state:
    st.session_state.default_setting = True

def default_setting_callback():
    if st.session_state.default_setting is True:
        st.session_state.default_index = 0
        st.session_state.disable_all = True
    elif st.session_state.default_setting is False:
        st.session_state.default_index = None
        st.session_state.disable_all = False
    st.session_state.temperature_change_toggle = False


class Panel:
    def __init__(self):
        self.analysis_mode = st.radio("Choose the analysis mode:", ["All wavelength", "Single wavelength"])

        self.__basic_panel()

        self.default_setting = st.checkbox("Use default setting(Canon + 50 micron core fiber + Nikon + QHY600)?",
                                           on_change=default_setting_callback, key="default_setting")

        self.__telescope_panel()
        self.__fiber_panel()
        self.__detector_camera_panel()
        self.__detector_panel()

    def __basic_panel(self):
        if self.analysis_mode == "All wavelength":
            with st.container(border=True):
                wavelength_col1, wavelength_col2 = st.columns(2)
                with wavelength_col1:
                    wavelength_red = st.slider('Select a range of wavelength (nm)', 460, 510, value=(460, 510),
                                               disabled=False)
                with wavelength_col2:
                    wavelength_blue = st.slider('Select a range of wavelength (nm)', 610, 690, value=(610, 690),
                                                disabled=False)
                self.wavelengths = np.concatenate((np.arange(wavelength_red[0], wavelength_red[1] + 1),
                                                   np.arange(wavelength_blue[0], wavelength_blue[1] + 1)), axis=None)
                self.exposure_time = st.number_input("Enter exposure time (s)", value=900)
                self.I = st.number_input('Enter emission-line flux (erg/cm$^2$/s/arcsec$^2$):', format='%g',
                                         value=1e-17)
                self.surface_brightness = st.number_input('Enter sky background surface brightness (mag/arcsec$^2$):',
                                                          value=21.5, format="%0.3f")

        elif self.analysis_mode == "Single wavelength":
            with st.container(border=True):
                wavelengths = st.number_input("Enter the wavelength (nm):")
                self.wavelengths = np.array(wavelengths, ndmin=1)
                self.exposure_time = st.slider("Exposure time (s)", min_value=0, max_value=1800, value=900)
                self.I = st.number_input("Enter the emission-line flux (erg/cm$^2$/s/arcsec$^2$):", format='%g',
                                         value=1e-17)
                self.intrinsic_broadening = st.number_input("Enter the intrinsic broadening (km/s):", format='%g',
                                                            value=0)
                self.surface_brightness = st.number_input('Enter sky background surface brightness (mag/arcsec$^2$):',
                                                          value=21.5, format="%0.3f")

    def __telescope_panel(self):
        with st.container(border=True):
            self.telescope_choice = st.selectbox("Choose the telescope:", list(settings.telescope.keys()),
                                                 index=st.session_state.default_index,
                                                 placeholder="Select telescope...",
                                                 disabled=st.session_state.disable_all)
            if self.telescope_choice == "Custom":
                telescope_popover = False
            else:
                telescope_popover = True
            with st.popover("Parameters of custom telescope", disabled=telescope_popover):
                settings.telescope['Custom']['focal length'] = st.number_input(
                    "Enter the focal length (mm) of the telescope:")
                settings.telescope['Custom']['diameter'] = st.number_input(
                    "Enter the dimater (mm) of the enterance pupil of the telescope?", format="%0.4f")

    def __fiber_panel(self):
        with st.container(border=True):
            self.fiber_choice = st.selectbox("Choose the fiber:", list(settings.fiber.keys()),
                                        index=st.session_state.default_index, placeholder="Select fiber...",
                                        disabled=st.session_state.disable_all)
            if self.fiber_choice == "Custom":
                fiber_popover = False
            else:
                fiber_popover = True
            with st.popover("Parameters of custom fiber", disabled=fiber_popover):
                settings.fiber['Custom'] = st.number_input("Enter the diameter (mm) of the core of the fiber:")

        if self.fiber_choice is None:
            self.d_fiber = None
        else:
            self.d_fiber = settings.fiber[self.fiber_choice]

    def __detector_camera_panel(self):
        with st.container(border=True):
            self.detector_camera_choice = st.selectbox("Choose the spectrograph camera:",
                                                  list(settings.detector_camera.keys()),
                                                  index=st.session_state.default_index,
                                                  placeholder="Select spectrograph camera...",
                                                  disabled=st.session_state.disable_all)
            if self.detector_camera_choice == "Custom":
                detector_camera_popover = False
            else:
                detector_camera_popover = True
            with st.popover("Parameters of custom spectrograph camera", disabled=detector_camera_popover):
                settings.detector_camera['Custom']['focal length'] = st.number_input(
                    "What is the focal length (mm) of the spectrograph camera?")

    def __detector_panel(self):
        with st.container(border=True):
            detector_col1, detector_col2 = st.columns([0.7, 0.3])
            with detector_col1:
                self.detector_choice = st.selectbox("Choose the detector:", list(settings.detector.keys()),
                                               index=st.session_state.default_index, placeholder="Select detector...",
                                               disabled=st.session_state.disable_all)
            with detector_col2:
                if self.detector_choice is None or self.detector_choice == "Custom":
                    temperature_disabled = True
                else:
                    temperature_disabled = False
                self.is_temperature_change = st.toggle('$\Delta$ T', disabled=temperature_disabled,
                                                  key="temperature_change_toggle")
            if self.detector_choice == "Custom" or self.detector_choice is None:
                if self.detector_choice == "Custom":
                    detector_popover = False
                else:
                    detector_popover = True
                with st.popover("Parameters of custom detector", disabled=detector_popover):  #
                    settings.detector['Custom']['pixel size'] = st.number_input(
                        "Enter the pixel size (um):")
                    settings.detector['Custom']['readnoise'] = st.number_input(
                        "Enter the read noise (e$^-$) of the detector:",
                        format="%.3f")
                    settings.detector['Custom']['darkcurrent'] = st.number_input(
                        "Enter the dark current (e$^-$/pixel/s) of the detector:", format="%.4f")
                    settings.detector['Custom']['temperature'] = st.number_input(
                        "Enter the temperature (C$^\circ$) of the detector:")
            else:
                with st.popover(f"Temperature change of {self.detector_choice}", disabled=not self.is_temperature_change):
                    with st.container(border=True):
                        st.write(f"The default temperature is {settings.detector[self.detector_choice]['temperature']}")
                    self.temperature_change = st.number_input(
                        "Enter the change of temperature (C$^\circ$):", format="%d", step=5)
