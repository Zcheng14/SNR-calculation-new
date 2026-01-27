#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import demo_multi_func as demo_func
from demo_multi_func import Telescope, DetectorCamera, Detector, CommonSettings
from demo_multi_func import tot_efficiency, cal_dispersion
from demo_multi_func import cal_sky_background, cal_signal, cal_continuum
from demo_multi_func import number_of_pixels, cal_read_noise, cal_dark_current, cal_SNR
current_path = os.getcwd()


# %% test the code
# ----------------------------------
# Initialize objects
# 1. Telescope: the default is the Canon lens
# 2. Camera: options are "Custom design lens" and "Nikkor lens"
# 3. Detector: options are 'QHY461' and 'QHY600'
# 4. CommonSettings: default settings

# This script uses the "Custom design lens" and 'QHY461' as an example
# And this script is suitable for multiple targets with different fiber slit heights in "All wavelength" analysis mode.
# If you want to do "Single wavelength" analysis mode, please go to the website of ETC.


# The input information includes:
# Header of the fits file:
# 1. emission line flux in erg/cm^2/s/arcsec^2
# 2. Exposure time in seconds: selected from [180, 900, 1800, 3600]
# 3. Sky surface brightness in mag/arcsec^2
# 4. If continuum is considered, 
# 5. continuum surface brightness in erg/cm^2/s/A/arcsec^2: if no continuum, set it to 0

# ----------------------------------
#%% construct the example fits file with multiple sources

fluxes_list = [5e-17, 1e-16, 5e-16]  # erg/cm^2/s/arcsec^2
exposure_time_list = [900,1800,3600]  # seconds
sky_mag_list = [20.5,21.5,22.5]  # mag/arcsec^2
if_continuum = [False, False, False]
continuum_list = [1e-18, 1e-18, 1e-18]  # erg/cm^2/s/A/arcsec^2

# save to a fits file
from astropy.io import fits
from astropy.table import Table
data = Table()
data['fluxes'] = fluxes_list
data['exposure_time'] = exposure_time_list
data['sky_mag'] = sky_mag_list
data['if_continuum'] = if_continuum
data['continuum'] = continuum_list
data.write('./data/multi_sources.fits', format='fits', overwrite=True)

#%%
telescope = Telescope()
camera = DetectorCamera(camera_type="Custom design lens")
detector = Detector(detector_type='QHY461')
common_settings = CommonSettings()


wavelengths, total_eff = tot_efficiency(common_settings, telescope, camera, detector)
dispersion_wavelengths, dispersion_values = cal_dispersion(common_settings, camera)

# read the fits file
data = fits.open('./data/multi_sources.fits')[1].data
fluxes_list = data['fluxes']
exposure_time_list = data['exposure_time']
sky_mag_list = data['sky_mag']
if_continuum_list = data['if_continuum']
continuum_list = data['continuum']


SNR_results = np.zeros((len(fluxes_list), len(common_settings.field_points), len(wavelengths)))  # store SNR results
SNR_mean_results = np.zeros((len(fluxes_list), len(common_settings.field_points)))  # store mean SNR results

for i in range(len(fluxes_list)):
    flux = fluxes_list[i]
    exposure_time = exposure_time_list[i]
    sky_mag = sky_mag_list[i]
    if_continuum = if_continuum_list[i]
    continuum_value = continuum_list[i]
    sky_per_pixel, sky_noise = cal_sky_background(sky_mag, exposure_time, total_eff, telescope, wavelengths, common_settings, camera, detector)
    signal_per_pixel, signal = cal_signal(flux, exposure_time, total_eff, telescope, wavelengths, common_settings)
    if if_continuum is True:
        continuum = cal_continuum(continuum_value, exposure_time, total_eff, telescope, wavelengths, common_settings, camera, detector)
    else:
        continuum = 0
    n_pixels = number_of_pixels(camera, detector, analysis_mode='All wavelength')
    read_noise = cal_read_noise(detector, n_pixels)
    dark_current = cal_dark_current(detector, exposure_time, n_pixels)
    SNR = cal_SNR(signal, sky_noise, read_noise, dark_current, continuum_mode=if_continuum, continuum=continuum)
    SNR_results[i] = SNR
    SNR_mean_results[i] = np.nanmean(SNR, axis=1)

#%% Save SNR results to a txt file
# ---------------------------------
# the txt file will have the following columns:
# 1. source flux
# 2. proposed exposure time (s)
# 3. fiber slit height (mm)
# 4. mean SNR (across all wavelengths)
# 5. SNR at Hb (486 nm)
# 6. SNR at OIII (501 nm)
# 7. SNR at Ha (656 nm)
# 8. SNR at SII (672 nm)
# 9. SNR at each wavelength (nm)
# ---------------------------------
output_file = './data/multi_sources_SNR_results.txt'
with open(output_file, 'w') as f:
    # write header
    header = 'Source_Flux\tExposure_Time(s)\tFiber_Slit_Height(mm)\tMean_SNR\tSNR_Hb(486nm)\tSNR_OIII(501nm)\tSNR_Ha(656nm)\tSNR_SII(672nm)'
    for wl in wavelengths:
        header += f'\tSNR_{wl:.1f}nm'
    f.write(header + '\n')
    
    # write data
    for i in range(len(fluxes_list)):
        for j in range(len(common_settings.field_points)):
            line = f'{fluxes_list[i]}\t{exposure_time_list[i]}\t{common_settings.field_points[j]}\t{SNR_mean_results[i,j]:.2f}'
            # find indices for specific wavelengths
            idx_Hb = np.argmin(np.abs(wavelengths - 486))
            idx_OIII = np.argmin(np.abs(wavelengths - 501))
            idx_Ha = np.argmin(np.abs(wavelengths - 656))
            idx_SII = np.argmin(np.abs(wavelengths - 672))
            line += f'\t{SNR_results[i,j,idx_Hb]:.2f}\t{SNR_results[i,j,idx_OIII]:.2f}\t{SNR_results[i,j,idx_Ha]:.2f}\t{SNR_results[i,j,idx_SII]:.2f}'
            for k in range(len(wavelengths)):
                line += f'\t{SNR_results[i,j,k]:.2f}'
            f.write(line + '\n')

#%% print txt file
with open(output_file, 'r') as f:
    content = f.read()
    print(content)


# %%
"""
# Plot total efficiency
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
"""