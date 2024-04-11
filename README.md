# Introduction
The code utilizes the Python language to estimate the signal-to-noise ratio (SNR) and spectral resolution of the AMASE-P system. To use the app, please visit [https://snr-calculation-cheung-yiu-hung.streamlit.app/](https://snr-calculation-cheung-yiu-hung.streamlit.app/). Select the desired settings from the sidebar on the website. If you are unable to see the sidebar, simply expand it by clicking the ">" symbol located on the top left-hand corner.

# Assumption
In the calculation of the noise sky background part, the code assumes that the data from MaNGA is 21.5 mag/arcsec<sup>2</sup>.

# Setting
## Default setting
The default setting of the system uses the Canon 400mm f/2.8 telephoto lens + 50 micron core fiber + Nikkor 58mm f/0.95 S Noct lens + QHY600M CMOS detector. The default setting takes into account various factors such as lens throughput, vignetting of the Nikkor lens, detector quantum efficiency (QE), etc. The spot size is based on Zemax simulation analysis rather than theoretical values.

## Custom setting
If a custom design is required, the current code can analyze custom telescopes, fibers, spectrograph cameras, and detectors. Other components such as gratings and dichroic beam splitters are not expected to be changed. The limitation of the custom setting is that ***the effectiveness will not be considered***, and the spot size calculation is based on theoretical calculations.
