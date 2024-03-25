import streamlit as st
import math
import scipy.constants as constant
import numpy as np
import matplotlib.pyplot as plt
import os

# data reading path

current_path = os.getcwd()

spot_size_path="/home/davidbear/SNR calculation/data/separation.txt"
focal_plane_module_path="/home/davidbear/SNR calculation/data/focal_plane_module.txt"
dichoric_reflection_path="/home/davidbear/SNR calculation/data/dichoric_reflection.txt"
dichoric_transmission_path="/home/davidbear/SNR calculation/data/dichoric_transmission.txt"


## setting
wavelength=np.concatenate((np.arange(460,511),np.arange(610,691)), axis=None) #nm
field_point=np.array([0,5,10,15,20,25,30]) #mm

telescope={
    "Canon 400mm f/2.8 telephoto lens":{
        "diameter": 392.5/2.9, # mm
        "focal length": 392.5, # mm
        "throughput": "/home/davidbear/SNR calculation/data/canon_throughput.txt"
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
        "throughput": "/home/davidbear/SNR calculation/data/nikkor_throughput.txt",
        "vignetting": "/home/davidbear/SNR calculation/data/vignetting.txt"
    },
    "Custom":{
        "focal length": None, # mm
        "throughput": None,
        "vignetting": None
    }
}

detector={
    "QHY600":{
        "QE" : "/home/davidbear/SNR calculation/data/QE_132.txt",
        "pixel size": [3.76,3.76], # um
        "readnoise": 1.683,
        "darkcurrent": 0.0022 # T=-20 degree
    },
    "Custom":{
        "QE" : None,
        "pixel size": [None, None],
        "readnoise": None,
        "darkcurrent": None
    }
}

# collimator
eff_collimator=0.9 # assume 90% reflexivity
f_collimator=200 # mm

# fiber
eff_fiber_transmission=0.96 # assume 4% reflectivity

# constant
speed_of_light=constant.speed_of_light # ~3e8 m/s 
frequency=constant.speed_of_light/(wavelength*1e-9)
h=constant.h # ~6.626e-34 J/Hz

# grating
line_density_blue = 3559 # lines/mm in blue channel
line_density_red = 2632 # lines/mm in red channel
        
incident_angle = 61.93 # degree

def cos_gamma(field_point):
    return np.cos(field_point/200)

def cos_beta(line_density,wavelength,field_point):
    return np.cos(np.arcsin(line_density*(wavelength*1e-6)/cos_gamma(field_point)-np.sin(math.radians(incident_angle))))


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
    with st.container(border=True):
        exposure_time = st.slider("Exposure time (s)", min_value=0, max_value=1800, value=900)
        I = st.number_input('Enter flux density (erg/cm$^2$/s/arcsec$^2$):', format='%g', value=5.665e-18)
        surface_brightness = st.number_input('Enter surface brightness (mag/arcsec$^2$):', value = 21.5, format="%0.3f")

    default_setting = st.checkbox("Use default setting(Canon + 50 micron core fiber + Nikon + QHY600)?", value=True, on_change=default_setting_callback)

    with st.container(border=True):
        telescope_choice = st.selectbox("Which telescope do you use?",list(telescope.keys()), index=default_setting_callback()[0], placeholder="Select telescope...", disabled=default_setting_callback()[1])
        if telescope_choice == "Custom":
            telescope_popover = False
        else:
            telescope_popover = True
        with st.popover("Parameters of custom telescope", disabled=telescope_popover):
            telescope['Custom']['diameter'] = st.number_input("What is the dimater (mm) of the enterance pupil of the telescope?", format="%0.4f")
            telescope['Custom']['focal length'] = st.number_input("What is the focal length (mm) of the telescope?")

    with st.container(border=True):
        fiber_choice = st.selectbox("Which fiber do you use?",list(fiber.keys()), index=default_setting_callback()[0], placeholder="Select fiber...", disabled=default_setting_callback()[1])
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
        detector_camera_choice = st.selectbox("Which spectrograph camera do you use?",list(detector_camera.keys()), index=default_setting_callback()[0], placeholder="Select spectrograph camera...", disabled=default_setting_callback()[1])
        if detector_camera_choice == "Custom":
            detector_camera_popover = False
        else:
            detector_camera_popover = True
        with st.popover("Parameters of custom spectrograph camera", disabled=detector_camera_popover):
            detector_camera['Custom']['focal length'] = st.number_input("What is the focal length (mm) of the spectrograph camera?")

    with st.container(border=True):
        detector_choice = st.selectbox("Which detector do you use?",list(detector.keys()), index=default_setting_callback()[0], placeholder="Select detector...", disabled=default_setting_callback()[1])
        if detector_choice == "Custom":
            detector_popover = False
        else:
            detector_popover = True
        with st.popover("Parameters of custom detector", disabled=detector_popover):
            detector['Custom']['pixel size'][0] = st.number_input("Enter the pixel size (um) in the dispersion direction:")
            detector['Custom']['pixel size'][1] = st.number_input("Enter the pixel size (um) in the spatial direction:")
            detector['Custom']['darkcurrent'] = st.number_input("What is the dark current (e/pixel/s) of the detector?")
            detector['Custom']['readnoise'] = st.number_input("What is the read noise (e) of the detector?")
            
def main():
    # canon throughput
    if telescope[telescope_choice]['throughput'] != None:    
        eff_telescope_tp=np.zeros(132, dtype='float')
        with open(telescope[telescope_choice]['throughput'], "r") as file:    
            for line,i in zip(file,range(132)):
                line=line.split()
                eff_telescope_tp[i]=float(line[0])
    else:
        eff_telescope_tp=np.ones(132)


    # nikkor throughput

    if detector_camera[detector_camera_choice]['throughput'] != None:
        eff_detector_camera_tp=np.zeros(132, dtype='float')
        with open(detector_camera[detector_camera_choice]['throughput'], "r") as file:   
            for line,i in zip(file,range(132)):
                line=line.split()
                eff_detector_camera_tp[i]=float(line[0])            
    else:
        eff_detector_camera_tp=np.ones(132)

    # spot size
    spot_size=np.zeros([7,132], dtype='float')
    if default_setting: 
        with open(spot_size_path, 'r') as file:
            j=0
            for line in file:
                line=line.split()
                spot_size[j]=line
                j=j+1
    else:
        for i in range(7):
            spot_size[i]=d_fiber*(detector_camera[detector_camera_choice]['focal length']/f_collimator)*(math.cos(math.radians(incident_angle))/np.concatenate((cos_beta(line_density_blue,wavelength[:51],field_point[i]),cos_beta(line_density_red,wavelength[51:],field_point[i])), axis=None))
        spot_size=spot_size*1e3

    num_pixel=np.ceil(((spot_size/2)**2*constant.pi/(detector[detector_choice]["pixel size"][0]*detector[detector_choice]["pixel size"][1])))

    # vignetting

    if detector_camera[detector_camera_choice]['vignetting'] != None:
        eff_vignetting=np.zeros([7,132], dtype='float')
        with open(detector_camera[detector_camera_choice]['vignetting'], 'r') as file:   
            j=0
            for line in file:
                line=line.split()
                eff_vignetting[j]=line
                j=j+1

        eff_vignetting=eff_vignetting*0.01

    else:
        eff_vignetting=np.ones([7,132])

    # QE
    if detector[detector_choice]['QE'] != None:
        QE=np.zeros(132, dtype='float')
        with open(detector[detector_choice]['QE'], 'r') as file:
            for line,i in zip(file,range(132)):
                line=line.split()
                QE[i]=line[0]
    else:
        QE=np.ones(132)

    # focal plane module

    eff_focal_plane_module=np.zeros(132)
    with open(focal_plane_module_path, "r") as file:    
        i=0
        for line in file:
            line=line.split()
            if int(line[0])==wavelength[i]:
                eff_focal_plane_module[i]=1-float(line[1])*0.01
                if wavelength[i]==wavelength[-1]:
                    break
                else:
                    i = i+1

    # dichoric

    eff_dichoric=np.zeros(132)

    # blue channel
    with open(dichoric_reflection_path, "r") as file:
        i=0
        for line in file:
            line=line.split()
            if int(line[0])==wavelength[:51][i]:
                eff_dichoric[i]=float(line[1])*0.01
                if wavelength[:51][i]==wavelength[:51][-1]:
                    break
                else:
                    i = i+1

    # red channel
    with open(dichoric_transmission_path, "r") as file:
        i=0
        for line in file:
            line=line.split()
            if int(line[0])==wavelength[51:][i]:
                eff_dichoric[i+51]=float(line[1])*0.01
                if wavelength[51:][i]==wavelength[51:][-1]:
                    break
                else:
                    i = i+1          


    ## Signal
    eff=np.zeros([7,132], dtype='float')
    for i in range(7):
        eff[i]=eff_telescope_tp*eff_detector_camera_tp*eff_vignetting[i]*eff_collimator*eff_dichoric*eff_fiber_transmission**2*eff_focal_plane_module

    A=(constant.pi/4)*(telescope[telescope_choice]["diameter"]**2*0.1**2) ## cm^2
    solid_angle=(constant.pi/4)*(d_fiber/telescope[telescope_choice]["focal length"])**2*(206264.5**2) #206264.5 radian^2 to arcsec^2

    signal=np.zeros(np.shape(eff))
    for i in range(np.shape(eff)[0]):
        signal[i]=I*A*solid_angle/(h*(speed_of_light/(wavelength*1e-9))/1e-7)*exposure_time*eff[i]*QE[i]


    ## Sky background
    manga_count=90*10**((surface_brightness-21.5)/(-2.5))

    manga_exposuretime=900
    manga_disperson=1.4
    sdss_mirror=((2.5/2)**2*constant.pi-(1.3/2)**2*constant.pi)*1e4
    manga_fiberarea=constant.pi
    manga_efficiency=0.32

    sky_brightness=(manga_count)/(manga_exposuretime*manga_disperson*sdss_mirror*manga_fiberarea*manga_efficiency) # electrons/s/Angstrom/arcsec^2/cm^2

    dispersion1=np.zeros([np.shape(eff)[0],51], dtype='float') # d\lambda/dl
    dispersion2=np.zeros([np.shape(eff)[0],81], dtype='float')
    for i in range(np.shape(eff)[0]):
        dispersion1[i]=cos_gamma(field_point)[i]*cos_beta(line_density_blue,wavelength[:51],field_point[i])/(detector_camera[detector_camera_choice]["focal length"]*line_density_blue)
        dispersion2[i]=cos_gamma(field_point)[i]*cos_beta(line_density_red,wavelength[51:],field_point[i])/(detector_camera[detector_camera_choice]["focal length"]*line_density_red)

    dispersion=np.concatenate((dispersion1,dispersion2), axis=1)

    sky_brightness=sky_brightness*dispersion*(detector[detector_choice]["pixel size"][0]*1e4) # electrons/s/pixel/arcsec^2/cm^2

    sky_noise=sky_brightness*exposure_time*(spot_size/detector[detector_choice]["pixel size"][0])*solid_angle*A*eff # number of electrons


    ## Read noise
    readnoise=detector[detector_choice]["readnoise"]**2*num_pixel


    ## dark noise
    darknoise=detector[detector_choice]["darkcurrent"]*num_pixel*exposure_time

    ## SNR
    SNR=signal/np.sqrt(signal+sky_noise+readnoise+darknoise)


    ## Plot
    fig, [ax1, ax2] = plt.subplots(nrows=1,ncols=2,figsize=(30,15), layout="constrained")
    c=['#FF0000','#ffa500','#FFFF00','#00FF00','#00FFFF','#0000FF','#800080']

    x1=wavelength[:51]
    x2=wavelength[51:]

    for f,f_n in zip(field_point,range(len(field_point))):
        y1=SNR[f_n][:51]
        y2=SNR[f_n][51:]
        ax1.plot(x1,y1, label=str(f)+"mm", color=c[f_n])
        ax2.plot(x2,y2, label=str(f)+"mm", color=c[f_n])

    ax1.legend()
    ax2.legend()
    st.pyplot(fig)

not_ready=None    
if telescope_choice == None or fiber_choice == None or detector_camera_choice == None or detector_choice == None:
    not_ready = True
else:
    not_ready = False
st.button('Analyse!', on_click=main, disabled=not_ready)