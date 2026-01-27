# Updating on 2025.12.10
New features:
1. Updating the sky background estimation.
2. the demo_multi.py can be used to calculate the SNR for multiple targets after giving exposure time.

# Updating on 2025.09.01
New feature:
1. Update the efficiency of fiber transmission.(from 0.96 to 0.9)
# Updating on 2025.08.21
New features:
1. Add the simulation spot size data of custom lens.
2. Add the throughput of custom lens.

# Updating on 2025.08.15
New features:
1. Provide an option of spot size during calculation: Simulation or Theoretical calculation. The results of Theoretical calculation can be displayed rightly. ***[For Custom lens, the current simulation data is wrong, since the Focal length has changed from 105 mm to 99.5 mm]***
2. In the new design of the spectrograph with the custom lens, we align the custom lens with the output angle at 663nm in the red and 491nm in the blue. Now the dispersion formula is cos(beta-beta_0), where beta_0 is the emergent angle for 491nm and 663nm respectively. 

# Updating on 2025.08.14
To use the new app, please visit [https://snr-calculation-new-zcheng.streamlit.app/](https://snr-calculation-new-zcheng.streamlit.app/). Some new features:
1. The default setting of the system has been changed to the Custom design lens. ***[The throughput of Spectrograph camera is NOT be considered NOW.]***
2. The default exposure time is 1800s.
3. Provide an option of whether to include the continuum flux.

# Introduction
The code utilizes the Python language to estimate the signal-to-noise ratio (SNR) and spectral resolution of the AMASE-P system. To use the app, please visit [https://snr-calculation-cheung-yiu-hung.streamlit.app/](https://snr-calculation-cheung-yiu-hung.streamlit.app/). Select the desired settings from the sidebar on the website. If you are unable to see the sidebar, simply expand it by clicking the ">" symbol located on the top left-hand corner.

# Assumption
In the calculation of the noise sky background part, the code assumes that the data from MaNGA is 21.5 mag/arcsec<sup>2</sup>.

# Setting
## Default setting
The default setting of the system uses the Canon 400mm f/2.8 telephoto lens + 50 micron core fiber + Nikkor 58mm f/0.95 S Noct lens + QHY600M CMOS detector. The default setting takes into account various factors such as lens throughput, vignetting of the Nikkor lens, detector quantum efficiency (QE), etc. The spot size is based on Zemax simulation analysis rather than theoretical values.

## Custom setting
If a custom design is required, the current code can analyze custom telescopes, fibers, spectrograph cameras, and detectors. Other components such as gratings and dichroic beam splitters are not expected to be changed. The limitation of the custom setting is that ***the effectiveness will not be considered***, and the spot size calculation is based on theoretical calculations.
