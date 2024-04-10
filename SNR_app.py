import streamlit as st
import math
import scipy.constants as constant
import numpy as np
import matplotlib.pyplot as plt

#  data reading path

spot_size_path="/mount/src/snr-calculation/data/separation.txt"
focal_plane_module_path="/mount/src/snr-calculation/data/focal_plane_module.txt"
dichoric_reflection_path="/mount/src/snr-calculation/data/dichoric_reflection.txt"
dichoric_transmission_path="/mount/src/snr-calculation/data/dichoric_transmission.txt"


## setting
field_point=np.array([0,5,10,15,20,25,30]) #mm

telescope={
    "Canon 400mm f/2.8 telephoto lens":{
        "diameter": 392.5/2.9, # mm
        "focal length": 392.5, # mm
        "throughput": "/mount/src/snr-calculation/data/canon_throughput.txt"
    },
    "Custom":{
        "diameter": None,
        "focal length": None,
        "throughput": None
    }
}

fiber={
    "50 micron core fiber": 0.05, #mm
    "Custom": None
}

detector_camera={
    "Nikkor 58mm f/0.95 S Noct":{
        "focal length": 59.67, # mm
        "throughput": "/mount/src/snr-calculation/data/nikkor_throughput.txt",
        "vignetting": "/mount/src/snr-calculation/data/vignetting.txt"
    },
    "Custom":{
        "focal length": None, # mm
        "throughput": None,
        "vignetting": None
    }
}

detector={
    "QHY600":{
        "QE" : "/mount/src/snr-calculation/data/QHY600_QE.txt",
        "pixel size": [3.76,3.76], # um
        "readnoise": 1.683,
        "darkcurrent": 0.0022,
        "temperature": -20
    },
    "QHY4040":{
        "QE" : "/mount/src/snr-calculation/data/QHY4040_QE.txt",
        "pixel size": [9,9], # um
        "readnoise": 0.81,
        "darkcurrent": 0.16048, # T=-20 degree
        "temperature": -20
    },
    "Custom":{
        "QE" : None,
        "pixel size": [None, None],
        "readnoise": None,
        "darkcurrent": None,
        "temperature": None
    }
}

# collimator
eff_collimator=0.9 # assume 90% reflexivity
f_collimator=200 # mm

# fiber
eff_fiber_transmission=0.96 # assume 4% reflectivity

# constant
speed_of_light=constant.speed_of_light # ~3e8 m/s
h=constant.h # ~6.626e-34 J/Hz

# grating
line_density_blue = 3559 # lines/mm in blue channel
line_density_red = 2632 # lines/mm in red channel
        
incident_angle = 61.93 # degree

def cos_gamma(field_point):
    return np.cos(field_point/200)

def cos_beta(line_density,wavelength,field_point):
    return np.cos(np.arcsin(line_density*(wavelength*1e-6)/cos_gamma(field_point)-np.sin(math.radians(incident_angle))))

def resolution(wavelength,image_size,line_density,field_point):
    return line_density*(wavelength/1000)*detector_camera[detector_camera_choice]['focal length']/(image_size*cos_beta(line_density,wavelength,field_point)*cos_gamma(field_point))

## interface
def default_setting_callback():
    if default_setting:
        default_index = 0
        disable_all = True
    else:
        default_index = None
        disable_all = False
    return (default_index, disable_all)

with st.sidebar:
    analysis_mode = st.radio("Choose the analysis mode:", ["All wavelength","Single wavelength"])
    if analysis_mode == "All wavelength":
        with st.container(border=True):
            wavelength_col1 , wavelength_col2 = st.columns(2)
            with wavelength_col1:
                wavelength_red = st.slider('Select a range of wavelength (nm)', 460, 510, value=(460, 510), disabled=True)
            with wavelength_col2:
                wavelength_blue = st.slider('Select a range of wavelength (nm)', 610, 690, value=(610, 690), disabled=True)
            wavelength = np.concatenate((np.arange(wavelength_red[0],wavelength_red[1]+1),np.arange(wavelength_blue[0],wavelength_blue[1]+1)), axis=None) #nm
            cut = np.where(wavelength <= 510)[0]
            exposure_time = st.slider("Exposure time (s)", min_value=0, max_value=1800, value=900)
            I = st.number_input('Enter emission-line flux (erg/cm$^2$/s/arcsec$^2$):', format='%g', value=1e-17)
            surface_brightness = st.number_input('Enter sky background surface brightness (mag/arcsec$^2$):', value = 21.5, format="%0.3f")
            
    elif analysis_mode == "Single wavelength":
        with st.container(border=True):
            wavelength = st.number_input("Enter the wavelength (nm):")
            wavelength = np.array(wavelength, ndmin=1)
            exposure_time = st.slider("Exposure time (s)", min_value=0, max_value=1800, value=900)
            I = st.number_input("Enter the emission-line flux (erg/cm$^2$/s/arcsec$^2$):", format='%g',value = 1e-17)
            intrinsic_broadening = st.number_input("Enter the intrinsic broadening (km/s):", format='%g',value = 0)
            surface_brightness = st.number_input('Enter sky background surface brightness (mag/arcsec$^2$):', value = 21.5, format="%0.3f")
    
    default_setting = st.checkbox("Use default setting(Canon + 50 micron core fiber + Nikon + QHY600)?", value=True, on_change=default_setting_callback)
    
    with st.container(border=True):
        telescope_choice = st.selectbox("Choose the telescope:",list(telescope.keys()), index=default_setting_callback()[0], placeholder="Select telescope...", disabled=default_setting_callback()[1])
        if telescope_choice == "Custom":
            telescope_popover = False
        else:
            telescope_popover = True
        with st.popover("Parameters of custom telescope", disabled=telescope_popover):
            telescope['Custom']['focal length'] = st.number_input("Enter the focal length (mm) of the telescope:")
            telescope['Custom']['diameter'] = st.number_input("Enter the dimater (mm) of the enterance pupil of the telescope?", format="%0.4f")

    with st.container(border=True):
        fiber_choice = st.selectbox("Choose the fiber:",list(fiber.keys()), index=default_setting_callback()[0], placeholder="Select fiber...", disabled=default_setting_callback()[1])
        if fiber_choice == "Custom":
            fiber_popover = False
        else:
            fiber_popover = True
        with st.popover("Parameters of custom fiber", disabled=fiber_popover):
            fiber['Custom'] = st.number_input("Enter the diameter (mm) of the core of the fiber:")
    if fiber_choice == None:
        d_fiber = None
    else:
        d_fiber = fiber[fiber_choice]

    with st.container(border=True):
        detector_camera_choice = st.selectbox("Choose the spectrograph camera:",list(detector_camera.keys()), index=default_setting_callback()[0], placeholder="Select spectrograph camera...", disabled=default_setting_callback()[1])
        if detector_camera_choice == "Custom":
            detector_camera_popover = False
        else:
            detector_camera_popover = True
        with st.popover("Parameters of custom spectrograph camera", disabled=detector_camera_popover):
            detector_camera['Custom']['focal length'] = st.number_input("What is the focal length (mm) of the spectrograph camera?")

    with st.container(border=True):
        detector_choice = st.selectbox("Choose the detector:",list(detector.keys()), index=default_setting_callback()[0], placeholder="Select detector...", disabled=default_setting_callback()[1])
        if detector_choice == "Custom":
            detector_popover = False
        else:
            detector_popover = True
        with st.popover("Parameters of custom detector", disabled=detector_popover):
            detector['Custom']['pixel size'][0] = st.number_input("Enter the pixel size (um) in the dispersion direction:")
            detector['Custom']['pixel size'][1] = st.number_input("Enter the pixel size (um) in the spatial direction:")
            detector['Custom']['readnoise'] = st.number_input("Enter the read noise (e$^-$) of the detector:", format="%.3f")
            detector['Custom']['darkcurrent'] = st.number_input("Enter the dark current (e$^-$/pixel/s) of the detector:", format="%.4f")
            detector['Custom']['temperature'] = st.number_input("Enter the temperature (C$^\circ$) of the detector:")

with st.expander("Setting overview", expanded = True):
    with st.container(border=True):
        st.markdown(f"- Exposure time: {exposure_time} s")
        st.markdown(f"- Emission-line flux: {I} erg/cm$^2$/s/arcsec$^2$")
        st.markdown(f"- Sky background surface brightness: {surface_brightness} mag/arcsec$^2$")
    col1, col2 = st.columns(2) 
    with col1:
        with st.container(border=True):
            if telescope_choice is None:
                st.markdown(f"- Telescope: {chr(0x274C)}")
                st.markdown(f"- Focal length: {chr(0x274C)}")
                st.markdown(f"- Diameter: {chr(0x274C)}")
                st.markdown(f"- Throughput: {chr(0x274C)}")
            else:
                st.write(f'Telescope: {telescope_choice}')
                if telescope[telescope_choice]['focal length'] is None:
                    st.markdown(f"- Focal length: {chr(0x274C)}")
                else:
                    st.markdown(f"- Focal length: {telescope[telescope_choice]['focal length']} mm")
                if telescope[telescope_choice]['diameter'] is None:
                    st.markdown(f"- Diameter: {chr(0x274C)}")
                else:
                    st.markdown(f"- Diameter: {round(telescope[telescope_choice]['diameter'], 2)} mm")
                if telescope[telescope_choice]['throughput'] is None:
                    st.markdown(f"- Throughput: {chr(0x274C)}")
                else:
                    st.markdown(f"- Throughput: {chr(0x2705)}")

        with st.container(border=True):
            if fiber_choice is None:
                st.markdown(f'Fiber: {chr(0x274C)}')
                st.markdown(f"- Diameter: {chr(0x274C)}")
            else:
                st.write(f'Fiber: {fiber_choice}')
                if fiber[fiber_choice] is None:
                    st.markdown(f"- Diameter: {chr(0x274C)}")
                else:
                    st.markdown(f"- Diameter: {fiber[fiber_choice]} mm")

    with col2:
        with st.container(border=True):
            if detector_camera_choice is None:
                st.markdown(f'Spectrograph camera: {chr(0x274C)}')
                st.markdown(f"- Focal length: {chr(0x274C)}")
                st.markdown(f"- Throughput: {chr(0x274C)}")
                st.markdown(f"- Vignetting: {chr(0x274C)}")
            else:
                st.write(f'Spectrograph camera: {detector_camera_choice}')
                if detector_camera[detector_camera_choice]['focal length'] is None:
                    st.markdown(f"- Focal length: {chr(0x274C)}")
                else:
                    st.markdown(f"- Focal length: {detector_camera[detector_camera_choice]['focal length']} mm")
                if detector_camera[detector_camera_choice]['throughput'] is None:
                    st.markdown(f"- Throughput: {chr(0x274C)}")
                else:
                    st.markdown(f"- Throughput: {chr(0x2705)}")
                if detector_camera[detector_camera_choice]['vignetting'] is None:
                    st.markdown(f"- Vignetting: {chr(0x274C)}")
                else:
                    st.markdown(f"- Vignetting: {chr(0x2705)}")

        with st.container(border=True):
            if detector_choice is None:
                st.markdown(f'Detector: {chr(0x274C)}')
                st.markdown(f"- Pixel size: {chr(0x274C)}")
                st.markdown(f"- Read noise: {chr(0x274C)}")
                st.markdown(f"- Dark current: {chr(0x274C)}")
                st.markdown(f"- QE: {chr(0x274C)}")
            else:
                st.write(f'Detector: {detector_choice}')
                if detector[detector_choice]['pixel size'] is None:
                    st.markdown(f"- Pixel size: {chr(0x274C)}")
                else:
                    st.markdown(f"- Pixel size: {detector[detector_choice]['pixel size'][0]} mm x {detector[detector_choice]['pixel size'][1]} mm")
                if detector[detector_choice]['readnoise'] is None:
                    st.markdown(f"- Read noise: {chr(0x274C)}")
                else:
                    st.markdown(f"- Read noise: {detector[detector_choice]['readnoise']} e$^-$")
                if detector[detector_choice]['darkcurrent'] is None:
                    st.markdown(f"- Dark current: {chr(0x274C)}")
                else:
                    st.markdown(f"- Dark current: {detector[detector_choice]['darkcurrent']} e$^-$/pixel/s at {detector[detector_choice]['temperature']} C$^\circ$")
                if detector[detector_choice]['QE'] is None:
                    st.markdown(f"- QE: {chr(0x274C)}")
                else:
                    st.markdown(f"- QE: {chr(0x2705)}")

def focal_plane_module_file_reading(path):
    data = np.loadtxt(path)
    eff_focal_plane_module = np.interp(wavelength, data[:,0], data[:,1])
    return eff_focal_plane_module
    
            
def QE_file_reading(detector_choice):
    if detector[detector_choice]['QE'] is not None:
        data = np.loadtxt(detector[detector_choice]['QE'], delimiter=" ")
        QE = np.interp(wavelength, data[:,0], data[:,1])
    else:
        QE = np.ones_like(wavelength)
    return QE
     
def telescope_throughput_file_reading(telescope_choice):
    if telescope[telescope_choice]['throughput'] is not None:    
        data = np.loadtxt(telescope[telescope_choice]['throughput'])
        eff_telescope_tp = np.interp(wavelength, np.concatenate((np.arange(460,511),np.arange(610,691)), axis=None), data)
    else:
        eff_telescope_tp = np.ones_like(wavelength)
    return eff_telescope_tp

def spectrograph_camera_throughput_file_reading(detector_camera_choice):
    if detector_camera[detector_camera_choice]['throughput'] is not None:    
        data = np.loadtxt(detector_camera[detector_camera_choice]['throughput'])
        eff_detector_camera_tp = np.interp(wavelength, np.concatenate((np.arange(460,511),np.arange(610,691)), axis=None), data)
    else:
        eff_detector_camera_tp = np.ones_like(wavelength)
    return eff_detector_camera_tp

def spot_size_file_reading(default_setting):
    if default_setting:
        data = np.loadtxt(spot_size_path)
        spot_size = np.zeros((len(field_point),len(wavelength)))
        for i in range(len(field_point)):
            spot_size[i] = np.interp(wavelength, np.concatenate((np.arange(460,511),np.arange(610,691)), axis=None), data[i])
    else:
        spot_size = np.zeros((len(field_point),len(wavelength)))
        for i in range(len(field_point)):
            if analysis_mode == "All wavelength":
                spot_size[i] = d_fiber*(detector_camera[detector_camera_choice]['focal length']/f_collimator)*(math.cos(math.radians(incident_angle))/np.concatenate((cos_beta(line_density_blue,wavelength[: cut[-1] + 1],field_point[i]),cos_beta(line_density_red,wavelength[cut[-1] + 1:],field_point[i])), axis=None))
            elif analysis_mode == "Single wavelength":
                if wavelength <= 510:
                    line_density = line_density_blue
                else:
                    line_density = line_density_red
                spot_size[i] = d_fiber*(detector_camera[detector_camera_choice]['focal length']/f_collimator)*(math.cos(math.radians(incident_angle))/cos_beta(line_density,wavelength,field_point[i]))
        spot_size = spot_size*1e3
    return spot_size

def vignetting_file_reading(detector_camera_choice, spot_size):
    if detector_camera[detector_camera_choice]['vignetting'] is not None:
        data = np.loadtxt(detector_camera[detector_camera_choice]['vignetting'])
        data = data*0.01
        eff_vignetting = np.zeros_like(spot_size)
        for i in range(len(field_point)):
            eff_vignetting[i] = np.interp(wavelength, np.concatenate((np.arange(460,511),np.arange(610,691)), axis=None), data[i])
    else:
        eff_vignetting = np.ones_like(spot_size)
    return eff_vignetting

def dichoric_file_reading(ref_path = None, pass_path = None):
    if analysis_mode == "All wavelength":
        data_blue = np.loadtxt(ref_path)
        eff_dichoric_blue = np.interp(wavelength[: cut[-1] + 1], data_blue[:,0], data_blue[:,1])

        data_red = np.loadtxt(pass_path)
        eff_dichoric_red = np.interp(wavelength[cut[-1] + 1 :], data_red[:,0], data_red[:,1])

        eff_dichoric = np.concatenate((eff_dichoric_blue,eff_dichoric_red), axis=None)
    elif analysis_mode == "Single wavelength":
        if wavelength <= 510:
            data = np.loadtxt(ref_path)
        else:
            data = np.loadtxt(pass_path)
        eff_dichoric = np.interp(wavelength, data[:,0], data[:,1])
    return eff_dichoric
        
def main():
    # canon throughput
    eff_telescope_tp = telescope_throughput_file_reading(telescope_choice)

    # nikkor throughput
    eff_detector_camera_tp = spectrograph_camera_throughput_file_reading(detector_camera_choice)
    
    # spot size and number of pixel
    spot_size = spot_size_file_reading(default_setting)
    
    if analysis_mode == "All wavelength":
        num_pixel = (spot_size/2)**2*constant.pi/(detector[detector_choice]["pixel size"][0]*detector[detector_choice]["pixel size"][1])
    elif analysis_mode == "Single wavelength":
        num_pixel = np.sqrt(spot_size**2+intrinsic_broadening**2)*spot_size/4*constant.pi/(detector[detector_choice]["pixel size"][0]*detector[detector_choice]["pixel size"][1])

    # vignetting
    eff_vignetting = vignetting_file_reading(detector_camera_choice, spot_size)

    # QE
    QE = QE_file_reading(detector_choice)

    # focal plane module
    eff_focal_plane_module = 1-focal_plane_module_file_reading(focal_plane_module_path)*0.01

    # dichoric

    eff_dichoric = dichoric_file_reading(ref_path = dichoric_reflection_path, pass_path = dichoric_transmission_path)*0.01

    ## Signal
    eff=np.zeros_like(spot_size)
    for i in range(7):
        eff[i]=eff_telescope_tp*eff_detector_camera_tp*eff_vignetting[i]*eff_collimator*eff_dichoric*eff_fiber_transmission**2*eff_focal_plane_module

    A=(constant.pi/4)*(telescope[telescope_choice]["diameter"]**2*0.1**2) ## cm^2
    solid_angle=(constant.pi/4)*(d_fiber/telescope[telescope_choice]["focal length"])**2*(206264.5**2) #206264.5 radian^2 to arcsec^2
    
    photon_counts = np.zeros(np.shape(eff))
    signal = np.zeros(np.shape(eff))
    for i in range(np.shape(eff)[0]):
        photon_counts[i] = I*A*solid_angle/(h*(speed_of_light/(wavelength*1e-9))/1e-7)*exposure_time*eff[i]
        signal[i] = photon_counts[i]*QE


    ## Sky background
    manga_count=90*10**((surface_brightness-21.5)/(-2.5))

    manga_exposuretime=900
    manga_disperson=1.4
    sdss_mirror=((2.5/2)**2*constant.pi-(1.3/2)**2*constant.pi)*1e4
    manga_fiberarea=constant.pi
    manga_efficiency=0.32

    sky_brightness=(manga_count)/(manga_exposuretime*manga_disperson*sdss_mirror*manga_fiberarea*manga_efficiency) # electrons/s/Angstrom/arcsec^2/cm^2
    
    if analysis_mode == "All wavelength":
        dispersion1 = np.zeros([len(field_point),len(wavelength[: cut[-1] + 1])], dtype='float') # d\lambda/dl
        dispersion2 = np.zeros([len(field_point),len(wavelength[cut[-1] + 1:])], dtype='float')

        for i in range(len(field_point)):
            dispersion1[i] = cos_gamma(field_point)[i]*cos_beta(line_density_blue,wavelength[: cut[-1] + 1],field_point[i])/(detector_camera[detector_camera_choice]["focal length"]*line_density_blue)
            dispersion2[i] = cos_gamma(field_point)[i]*cos_beta(line_density_red,wavelength[cut[-1] + 1 :],field_point[i])/(detector_camera[detector_camera_choice]["focal length"]*line_density_red)

        dispersion = np.concatenate((dispersion1,dispersion2), axis=1)
        
    elif analysis_mode == "Single wavelength":
        dispersion = np.zeros([len(field_point),len(wavelength)])
        if wavelength <= 510:
            line_density = line_density_blue
        else:
            line_density = line_density_red
        for i in range(len(field_point)):
            dispersion[i] = cos_gamma(field_point)[i]*cos_beta(line_density,wavelength,field_point[i])/(detector_camera[detector_camera_choice]["focal length"]*line_density)

    sky_brightness = sky_brightness*dispersion*(detector[detector_choice]["pixel size"][0]*1e4) # electrons/s/pixel/arcsec^2/cm^2

    sky_noise_per_pixel = sky_brightness*exposure_time*solid_angle*A*eff # number of electrons
    
    sky_noise = sky_noise_per_pixel*(spot_size/detector[detector_choice]["pixel size"][0])


    ## Read noise
    readnoise_per_pixel = detector[detector_choice]["readnoise"]**2
    readnoise = readnoise_per_pixel*num_pixel
    
    ## dark noise
    darknoise_per_pixel = detector[detector_choice]["darkcurrent"]*exposure_time
    darknoise = darknoise_per_pixel*num_pixel

    ## SNR
    SNR=signal/np.sqrt(signal+sky_noise+readnoise+darknoise)
    
    if analysis_mode == "All wavelength":
        ## Plot
        fig, [ax1, ax2] = plt.subplots(nrows=1,ncols=2,figsize=(30,15), layout="constrained")
        c=['#FF0000','#ffa500','#FFFF00','#00FF00','#00FFFF','#0000FF','#800080']

        x1=wavelength[: cut[-1] + 1]
        x2=wavelength[cut[-1] + 1:]

        for f,f_n in zip(field_point,range(len(field_point))):
            y1=SNR[f_n][:cut[-1] + 1]
            y2=SNR[f_n][cut[-1] + 1:]
            ax1.plot(x1,y1, label=str(f)+"mm", color=c[f_n])
            ax2.plot(x2,y2, label=str(f)+"mm", color=c[f_n])

        ax1.set_xlim(wavelength[0], wavelength[cut[-1]])    
        xticks = ax1.get_xticks()

        special_x1 = 486
        ax1.axvline(x=special_x1, color='black', linestyle='--')
        special_label1 = "H$\\beta$"
        special_x2 = 501
        ax1.axvline(x=special_x2, color='black', linestyle='--')
        special_label2 = "O[III]"

        xticks = list(xticks)
        xticks=[int(i) for i in xticks]
        xticks.remove(500) 
        xticks.extend([special_x1, special_x2])

        xtick_labels = []
        for tick in xticks:
            if tick == special_x1:
                xtick_labels.append(special_label1)
            elif tick == special_x2:
                xtick_labels.append(special_label2)
            else:
                xtick_labels.append(str(tick))

        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xtick_labels)

        ax2.set_xlim(wavelength[cut[-1] + 1], wavelength[-1])
        xticks = ax2.get_xticks()

        special_x1 = 656
        ax2.axvline(x=special_x1, color='black', linestyle='--')
        special_label1 = "H$\\alpha$"
        special_x2 = 672
        ax2.axvline(x=special_x2, color='black', linestyle='--')
        special_label2 = "S[II]"

        xticks = list(xticks)
        xticks=[int(i) for i in xticks]
        xticks.remove(660)
        xticks.remove(670)
        xticks.extend([special_x1, special_x2])

        xtick_labels = []
        for tick in xticks:
            if tick == special_x1:
                xtick_labels.append(special_label1)
            elif tick == special_x2:
                xtick_labels.append(special_label2)
            else:
                xtick_labels.append(str(tick))

        ax2.set_xticks(xticks)
        ax2.set_xticklabels(xtick_labels)

        ax1.set_xlabel('Wavelength (nm)', fontsize=36)
        ax1.set_ylabel('SNR', fontsize=36)
        ax1.tick_params(axis='x', labelsize=36)
        ax1.tick_params(axis='y', labelsize=36)

        ax2.set_xlabel('Wavelength (nm)', fontsize=36)
        ax2.set_ylabel('SNR', fontsize=36)
        ax2.tick_params(axis='x', labelsize=36)
        ax2.tick_params(axis='y', labelsize=36)

        ax1.legend(title='Fiber slit\'s height', fontsize=25, title_fontsize=25)
        ax2.legend(title='Fiber slit\'s height', fontsize=25, title_fontsize=25)

        ax1.set_title('SNR of red channel', fontsize=36)
        ax2.set_title('SNR of blue channel', fontsize=36)

        ## Resolution
        fig_r, [ax1_r, ax2_r] = plt.subplots(nrows=1, ncols=2, figsize=(30, 15), layout="constrained")
        for f, i in zip(field_point, range(len(field_point))):
            resolution_red = resolution(x1, spot_size[i][:cut[-1] + 1], 3559, f)
            ax1_r.plot(x1, resolution_red, label=str(f) + "mm", color=c[i])
            resolution_blue = resolution(x2, spot_size[i][cut[-1] + 1:], 2632, f)
            ax2_r.plot(x2, resolution_blue, label=str(f) + "mm", color=c[i])

        ax1_r.set_xlim(wavelength[0], wavelength[cut[-1]])
        xticks = ax1_r.get_xticks()

        special_x1 = 486
        ax1_r.axvline(x=special_x1, color='black', linestyle='--')
        special_label1 = "H$\\beta$"
        special_x2 = 501
        ax1_r.axvline(x=special_x2, color='black', linestyle='--')
        special_label2 = "O[III]"

        xticks = list(xticks)
        xticks = [int(i) for i in xticks]
        xticks.remove(500)
        xticks.extend([special_x1, special_x2])

        xtick_labels = []
        for tick in xticks:
            if tick == special_x1:
                xtick_labels.append(special_label1)
            elif tick == special_x2:
                xtick_labels.append(special_label2)
            else:
                xtick_labels.append(str(tick))

        ax1_r.set_xticks(xticks)
        ax1_r.set_xticklabels(xtick_labels)

        ax2_r.set_xlim(wavelength[cut[-1] + 1], wavelength[-1])
        xticks = ax2_r.get_xticks()

        special_x1 = 656
        ax2_r.axvline(x=special_x1, color='black', linestyle='--')
        special_label1 = "H$\\alpha$"
        special_x2 = 672
        ax2_r.axvline(x=special_x2, color='black', linestyle='--')
        special_label2 = "S[II]"

        xticks = list(xticks)
        xticks = [int(i) for i in xticks]
        xticks.remove(660)
        xticks.remove(670)
        xticks.extend([special_x1, special_x2])

        xtick_labels = []
        for tick in xticks:
            if tick == special_x1:
                xtick_labels.append(special_label1)
            elif tick == special_x2:
                xtick_labels.append(special_label2)
            else:
                xtick_labels.append(str(tick))

        ax2_r.set_xticks(xticks)
        ax2_r.set_xticklabels(xtick_labels)

        ax1_r.set_xlabel('Wavelength (nm)', fontsize=36)
        ax1_r.set_ylabel('Resolution', fontsize=36)
        ax1_r.tick_params(axis='x', labelsize=36)
        ax1_r.tick_params(axis='y', labelsize=36)

        ax2_r.set_xlabel('Wavelength (nm)', fontsize=36)
        ax2_r.set_ylabel('Resolution', fontsize=36)
        ax2_r.tick_params(axis='x', labelsize=36)
        ax2_r.tick_params(axis='y', labelsize=36)

        ax1_r.legend(title='Fiber slit\'s height', fontsize=25, title_fontsize=25)
        ax2_r.legend(title='Fiber slit\'s height', fontsize=25, title_fontsize=25)

        ax1_r.set_title('Resolution of red channel', fontsize=36)
        ax2_r.set_title('Resolution of blue channel', fontsize=36)

        ## Show calculation
        with st.expander("Result", expanded=True):
            red_result, blue_result = st.columns(2) 
            st.pyplot(fig)
            st.pyplot(fig_r)
            with red_result:
                st.write('Red channel')
                st.markdown(f"- Average photon counts from target: {round(np.mean(photon_counts[:,:cut[-1] + 1]))}")
                st.markdown(f"- Average PSF area: {round(np.mean(num_pixel[:,:cut[-1] + 1]))} pixel")
                st.markdown(f"- Average sky counts: {round(np.mean(sky_noise_per_pixel[:,:cut[-1] + 1]))} e$^-$/1D pixel")
                st.markdown(f"- Dark counts: {round(darknoise_per_pixel)} e$^-$/pixel at {detector[detector_choice]['temperature']} C$^\circ$")
                st.markdown(f"- Read noise: {round(readnoise_per_pixel)} e$^-$/pixel")
            with blue_result:
                st.write('Blue channel')
                st.markdown(f"- Average photon counts from target: {round(np.mean(photon_counts[:,cut[-1] + 1:]))}")
                st.markdown(f"- Average PSF area: {round(np.mean(num_pixel[:,cut[-1] + 1:]))} pixel")
                st.markdown(f"- Average sky counts: {round(np.mean(sky_noise_per_pixel[:,cut[-1] + 1:]))} e$^-$/1D pixel")
                st.markdown(f"- Dark counts: {round(darknoise_per_pixel)} e$^-$/pixel at {detector[detector_choice]['temperature']} C$^\circ$")
                st.markdown(f"- Read noise: {round(readnoise_per_pixel)} e$^-$/pixel")
    elif analysis_mode == "Single wavelength":
        with st.container(border=True):
            resolution_single = np.zeros_like(spot_size)
            if wavelength <= 510:
                for f,i in zip(field_point , range(len(field_point))):
                    resolution_single[i] = resolution(wavelength, spot_size[i], line_density_blue, f)
            else:
                for f,i in zip(field_point , range(len(field_point))):
                    resolution_single[i] = resolution(wavelength, spot_size[i], line_density_red, f)

            def round_up_to_string(data, num_decimal):
                data = np.squeeze(data)
                data = np.around(data, decimals=num_decimal)
                data = ", ".join(str(x) for x in data)
                return data
            
            SNR = round_up_to_string(SNR, 2)
            st.write(f"The SNR of {np.squeeze(wavelength)}: [{SNR}]")

            resolution_single = round_up_to_string(resolution_single, 0)
            st.write(f"The resolution of {np.squeeze(wavelength)}: [{resolution_single}]")
            
            photon_counts = round_up_to_string(photon_counts, 0)
            st.markdown(f"- Photon counts from target: [{photon_counts}]")
            
            num_pixel = round_up_to_string(num_pixel, 0)
            st.markdown(f"- PSF area: [{num_pixel}] pixel")
            
            sky_noise_per_pixel = round_up_to_string(sky_noise_per_pixel, 0)
            st.markdown(f"- sky counts: [{sky_noise_per_pixel}] e$^-$/1D pixel")
            
            st.markdown(f"- Dark counts: {round(darknoise_per_pixel)} e$^-$/pixel at {detector[detector_choice]['temperature']} C$^\circ$")
            st.markdown(f"- Read noise: {round(readnoise_per_pixel)} e$^-$/pixel")

not_ready=None    
if telescope_choice == None or fiber_choice == None or detector_camera_choice == None or detector_choice == None:
    not_ready = True
else:
    not_ready = False
st.button('Analyse!', on_click=main, disabled=not_ready, type="primary")