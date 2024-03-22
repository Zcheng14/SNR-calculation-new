#!/usr/bin/env python
# coding: utf-8

from astropy.io import fits
import math
import scipy.constants as constant
import numpy as np
import matplotlib.pyplot as plt
import os

# data reading path

current_path = os.getcwd()

Canon_tp_path=current_path+"/data/canon_throughput.txt"
Nikon_tp_path=current_path+"/data/nikkor_throughput.txt"
spot_size_path=current_path+"/data/separation.txt"
vignetting_path=current_path+"/data/vignetting.txt"
QE_path=current_path+"/data/QE_132.txt"
focal_plane_module_path=current_path+"/data/focal_plane_module.txt"
dichoric_reflection_path=current_path+"/data/dichoric_reflection.txt"
dichoric_transmission_path=current_path+"/data/dichoric_transmission.txt"


# # setting
wavelength=np.concatenate((np.arange(460,511),np.arange(610,691)), axis=None) #nm
field_point=np.array([0,5,10,15,20,25,30]) #mm





telescope={
    "Canon 400mm f/2.8 telephoto lens":{
        "diameter": 392.5/2.9, # mm
        "focal length": 392.5, # mm
        "throughput": current_path+"/data/canon_throughput.txt"
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
        "throughput": current_path+"/data/nikkor_throughput.txt",
        "vignetting": current_path+"/data/vignetting.txt"
    },
    "Custom":{
        "focal length": None, # mm
        "throughput": None,
        "vignetting": None
    }
}

detector={
    "QHY600":{
        "QE" : current_path+"/data/QE_132.txt",
        "pixel size": (3.76,3.76), # um
        "readnoise": 1.683,
        "darkcurrent": 0.0022 # T=-20 degree
    },
    "Custom":{
        "QE" : None,
        "pixel size": None,
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
c=constant.speed_of_light # ~3e8 m/s 
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


# # Input

try:
    exposure_time = int(input("Enter exposure time (seconds): "))
except ValueError:
    print("Please input an integer for the exposure time.")
    exit()
    
try:
    I = float(input("Enter flux density (erg/cm^2/s/arcsec^2): "))
except ValueError:
    print("Please input a float for the flux density.")
    exit()
    
try:
    surface_brightness = float(input("Enter surface brightness (mag/arcsec^2): "))
except ValueError:
    print("Please input a float for the surface brightness.")
    exit()





try:
    default_setting = input("Use default setting(Canon + 50 micron core fiber + Nikon + QHY600)? Enter 0 (False) or 1 (True):")
    default_setting = bool(int(default_setting))
except ValueError:
    print("Please input 0 or 1.")
    exit()





if default_setting:
    telescope_choice="Canon 400mm f/2.8 telephoto lens"
    d_fiber=0.05 # mm
    detector_camera_choice="Nikkor 58mm f/0.95 S Noct"
    detector_choice="QHY600"
else:
    for key_n in range(len(telescope)):
        print(list(telescope.keys())[key_n]+ " (" + str(key_n) + ")")
    choice_value=int(input("Which telescope do you use? Type the corresponding integer:"))
    telescope_choice=list(telescope.keys())[choice_value]
    if list(telescope.keys())[choice_value] == "Custom":
        telescope['Custom']['diameter'] = float(input("What is the dimater (mm) of the enterance pupil of the telescope?"))
        telescope['Custom']['focal length'] = float(input("What is the focal length (mm) of the telescope?"))
    
    choice_value=None
    
    for key_n in range(len(fiber)):
        print(list(fiber.keys())[key_n]+ " (" + str(key_n) + ")")
    choice_value=int(input("Which fiber do you use? Type the corresponding integer:"))
    if list(fiber.keys())[choice_value] == "Custom":
        fiber['Custom']=float(input("What is the diameter (mm) of the core of the fiber?"))
    d_fiber=fiber[list(fiber.keys())[choice_value]]
    
    choice_value=None
    
    for key_n in range(len(detector_camera)):
        print(list(detector_camera.keys())[key_n]+ " (" + str(key_n) + ")")
    choice_value=int(input("Which spectrograph camera do you use? Type the corresponding integer:"))
    detector_camera_choice=list(detector_camera.keys())[choice_value]
    if list(detector_camera.keys())[choice_value] == "Custom":
        detector_camera['Custom']['focal length'] = float(input("What is the focal length (mm) of the spectrograph camera?"))
        
    choice_value=None
    
    for key_n in range(len(detector)):
        print(list(detector.keys())[key_n]+ " (" + str(key_n) + ")")
    choice_value=int(input("Which detector do you use? Type the corresponding integer:"))
    detector_choice=list(detector.keys())[choice_value]
    if list(detector.keys())[choice_value] == "Custom":
        user_input = input("Enter the pixel size (um) in the format (dispersion, spatial) e.g.(3.76,3.76): ")
        user_input = user_input.strip()
        user_input = user_input.strip('()')
        user_input = user_input.split(',')
        detector['Custom']['pixel size'] = tuple(float(v) for v in user_input)
        detector['Custom']['darkcurrent'] = float(input("What is the dark current (e/pixel/s) of the detector?"))
        detector['Custom']['readnoise'] = float(input("What is the read noise (e) of the detector?"))


# # text file reading

# canon throughput

if telescope_choice == "Canon 400mm f/2.8 telephoto lens":    
    eff_telescope_tp=np.zeros(132, dtype='float')
    with open(Canon_tp_path, "r") as file:    
        for line,i in zip(file,range(132)):
            line=line.split()
            eff_telescope_tp[i]=float(line[0])
else:
    eff_telescope_tp=np.ones(132)





# nikkor throughput

if detector_camera_choice == "Nikkor 58mm f/0.95 S Noct":
    eff_detector_camera_tp=np.zeros(132, dtype='float')
    with open(Nikon_tp_path, "r") as file:   
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

if detector_camera_choice == "Nikkor 58mm f/0.95 S Noct":
    eff_vignetting=np.zeros([7,132], dtype='float')
    with open(vignetting_path, 'r') as file:   
        j=0
        for line in file:
            line=line.split()
            eff_vignetting[j]=line
            j=j+1

    eff_vignetting=eff_vignetting*0.01
    
else:
    eff_vignetting=np.ones([7,132])




# QE
if detector_choice=="QHY600":
    QE=np.zeros(132, dtype='float')
    with open(QE_path, 'r') as file:
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


# # Signal




eff=np.zeros([7,132], dtype='float')
for i in range(7):
        eff[i]=eff_telescope_tp*eff_detector_camera_tp*eff_vignetting[i]*eff_collimator*eff_dichoric*eff_fiber_transmission**2*eff_focal_plane_module





A=(constant.pi/4)*(telescope[telescope_choice]["diameter"]**2*0.1**2) ## cm^2
solid_angle=(constant.pi/4)*(d_fiber/telescope[telescope_choice]["focal length"])**2*(206264.5**2) #206264.5 radian^2 to arcsec^2




signal=np.zeros(np.shape(eff))
for i in range(np.shape(eff)[0]):
    signal[i]=I*A*solid_angle/(h*(c/(wavelength*1e-9))/1e-7)*exposure_time*eff[i]*QE[i]


# # Sky background




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



# # Read noise




readnoise=detector[detector_choice]["readnoise"]**2*num_pixel


# # dark noise




darknoise=detector[detector_choice]["darkcurrent"]*num_pixel*exposure_time


# # SNR




SNR=signal/np.sqrt(signal+sky_noise+readnoise+darknoise)


# # Plot




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
plt.show()