#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from astropy.io import fits
import math
import scipy.constants as constant
import numpy as np
import matplotlib.pyplot as plt
import os


# In[ ]:


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


# # Input

# In[ ]:


exposure_time=900 # s
I=5.665e-18 # erg/cm^2/s/arcsec^2


# In[ ]:


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


# # setting

# In[ ]:


wavelength=np.concatenate((np.arange(460,511),np.arange(610,691)), axis=None)
field_point=np.array([0,5,10,15,20,25,30]) #mm


# In[ ]:


detector_camera={
    "Nikkor 58mm f/0.95 S Noct":{
        "focal length": 59.67 # mm
    }
}

detector={
    "QHY600":{
        "pixel size": 3.76, # um
        "readnoise": 1.683,
        "darkcurrent": 0.0022 # T=-20 degree
    }
}


# In[ ]:


detector_camera_choice="Nikkor 58mm f/0.95 S Noct"
detector_choice="QHY600"


# In[ ]:


# Canon lens
d_telescope=392.5/2.9 # mm
f_telescope=392.5 # mm

# collimator
eff_collimator=0.9 # assume 0.9 reflexivity

# fiber
d_fiber=0.05 # mm
eff_fiber_transmission=0.96 # 4% reflectivity

# constant
c=constant.speed_of_light # ~3e8 m/s 
frequency=constant.speed_of_light/(wavelength*1e-9)
h=constant.h # ~6.626e-34 J/Hz

## grating
#3559 lines/mm in blue channel
#2632 lines/mm in red channel
        
incident_angle = 61.93 # degree


def cos_gamma(field_point):
    return np.cos(field_point/200)

def cos_beta(line_density,wavelength,field_point):
    return np.cos(np.arcsin(line_density*(wavelength*1e-6)/cos_gamma(field_point)-np.sin(math.radians(incident_angle))))


# # text file reading

# In[ ]:


# canon throughput

eff_canon_tp=np.zeros(132, dtype='float')
with open(Canon_tp_path, "r") as file:    
    for line,i in zip(file,range(132)):
        line=line.split()
        eff_canon_tp[i]=float(line[0])


# In[ ]:


# nikkor throughput

eff_nikon_tp=np.zeros(132, dtype='float')
with open(Nikon_tp_path, "r") as file:   
    for line,i in zip(file,range(132)):
        line=line.split()
        eff_nikon_tp[i]=float(line[0])


# In[ ]:


# spot size

spot_size=np.zeros([7,132], dtype='float')
with open(spot_size_path, 'r') as file:
    j=0
    for line in file:
        line=line.split()
        spot_size[j]=line
        j=j+1

num_pixel=(spot_size/detector[detector_choice]["pixel size"])**2


# In[ ]:


# vignetting

eff_vignetting=np.zeros(132, dtype='float')
with open(vignetting_path, 'r') as file:   
    for line,i in zip(file,range(132)):
        line=line.split()
        eff_vignetting[i]=float(line[0])*0.01

# the code to produce vignetting text file        
'''wavelength=np.concatenate((np.arange(460,511),np.arange(610,691)), axis=None)
for i in wavelength:
    path="/home/davidbear/SNR calculation/gia4/gia_"+str(i)+"_0_Vignetting.txt"
    with open(path, 'r', encoding='utf-16') as file:
        j=0
        for index,line in enumerate(file):
            if index in range(17) and index!=13:
                continue
            elif index==13:
                match = re.search(r"\d+\.\d+", line)
                if match:
                    number = float(match.group())
                    print(number)
                    with open("/home/davidbear/SNR calculation/data/vignetting.txt", "a") as text:
                        text.write(str(number)+"\n")
                else:
                    print("Number not found")
                continue'''


# In[ ]:


# QE
QE=np.zeros(132, dtype='float')
with open(QE_path, 'r') as file:
    for line,i in zip(file,range(132)):
        line=line.split()
        QE[i]=line[0]

# the code to produce QE text file
'''with open(QE_path, 'r') as file:
    QE_data=np.zeros([600,2], dtype='float')
    QE=np.zeros(132, dtype='float')
    for line,i in zip(file,range(600)):
        line=line.split()
        QE_data[i]=np.array([float(x) for x in line])
    for w,i in zip(wavelength,range(i)):
        difference=abs(QE_data[:, 0]-w)
        QE[i]=QE_data[np.argmin(difference)][1]
        with open("/home/davidbear/SNR calculation/data/QE_132.txt", "a") as text:
            text.write(str(QE_data[np.argmin(difference)][1])+"\n")'''


# In[ ]:


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


# In[ ]:


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

# In[ ]:


eff=np.zeros([1,132], dtype='float')
for i in range(np.shape(eff)[0]):
    eff[i]=eff_canon_tp*eff_nikon_tp*eff_vignetting[i]*eff_collimator*eff_dichoric*eff_fiber_transmission**2*eff_focal_plane_module


# In[ ]:


A=(constant.pi/4)*(d_telescope**2*0.1**2) ## cm^2
solid_angle=(constant.pi/4)*(d_fiber/f_telescope)**2*(206264.5**2) #206264.5 radian to arcsec


# In[ ]:


signal=np.zeros(np.shape(eff))
for i in range(np.shape(eff)[0]):
    signal[i]=I*A*solid_angle/(h*(c/(wavelength*1e-9))/1e-7)*exposure_time*eff[i]*QE[i]


# # Sky background

# In[ ]:


manga_count=90
manga_exposuretime=900
manga_disperson=1.4
sdss_mirror=((2.5/2)**2*constant.pi-(1.3/2)**2*constant.pi)*1e4
manga_fiberarea=constant.pi
manga_efficiency=0.32


# In[ ]:


sky_brightness=(manga_count)/(manga_exposuretime*manga_disperson*sdss_mirror*manga_fiberarea*manga_efficiency)
#print(sky_brightness) # electrons/s/Angstrom/arcsec^2/cm^2


# In[ ]:


dispersion1=np.zeros([np.shape(eff)[0],51], dtype='float') # d\lambda/dl
dispersion2=np.zeros([np.shape(eff)[0],81], dtype='float')
for i in range(np.shape(eff)[0]):
    dispersion1[i]=cos_gamma(field_point)[i]*cos_beta(3559,wavelength[:51],field_point[i])/(detector_camera[detector_camera_choice]["focal length"]*3559)
    dispersion2[i]=cos_gamma(field_point)[i]*cos_beta(2632,wavelength[51:],field_point[i])/(detector_camera[detector_camera_choice]["focal length"]*2632)

dispersion=np.concatenate((dispersion1,dispersion2), axis=None)


# In[ ]:


sky_brightness=sky_brightness*dispersion*(detector[detector_choice]["pixel size"]*1e4)
# print(sky_brightness) # electrons/s/pixel/arcsec^2/cm^2


# In[ ]:


sky_noise=sky_brightness*exposure_time*np.sqrt(num_pixel)*solid_angle*A # number of electrons
# sqrt(num_pixel) as only counts 1D pixel


# # Read noise

# In[ ]:


readnoise=detector[detector_choice]["readnoise"]**2*num_pixel


# # dark noise

# In[ ]:


darknoise=detector[detector_choice]["darkcurrent"]*num_pixel*exposure_time


# # SNR

# In[ ]:


SNR=signal/np.sqrt(signal+sky_noise+readnoise+darknoise)


# # Plot

# In[ ]:


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

# In[ ]:


'''
fig, ax = plt.subplots()
ax.set_title('Signal to noise ratio')
(line1, ) = ax.plot(x, y1, label='0', markersize=1, color=c[0])
(line2, ) = ax.plot(x, y2, label='5', markersize=1, color=c[1])
(line3, ) = ax.plot(x, y3, label='10', markersize=1, color=c[2])
(line4, ) = ax.plot(x, y4, label='15', markersize=1, color=c[3])
(line5, ) = ax.plot(x, y5, label='20', markersize=1, color=c[4])
(line6, ) = ax.plot(x, y6, label='25', markersize=1, color=c[5])
(line7, ) = ax.plot(x, y7, label='30', markersize=1, color=c[6])
leg = ax.legend(fancybox=True, shadow=True)

lines = [line1, line2, line3, line4, line5, line6, line7]
map_legend_to_ax = {}  # Will map legend lines to original lines.

pickradius = 5  # Points (Pt). How close the click needs to be to trigger an event.

for legend_line, ax_line in zip(leg.get_lines(), lines):
    legend_line.set_picker(pickradius)  # Enable picking on the legend line.
    map_legend_to_ax[legend_line] = ax_line


def on_pick(event):
    # On the pick event, find the original line corresponding to the legend
    # proxy line, and toggle its visibility.
    legend_line = event.artist

    # Do nothing if the source of the event is not a legend line.
    if legend_line not in map_legend_to_ax:
        return

    ax_line = map_legend_to_ax[legend_line]
    visible = not ax_line.get_visible()
    ax_line.set_visible(visible)
    # Change the alpha on the line in the legend, so we can see what lines
    # have been toggled.
    legend_line.set_alpha(1.0 if visible else 0.2)
    fig.canvas.draw()


fig.canvas.mpl_connect('pick_event', on_pick)

# Works even if the legend is draggable. This is independent from picking legend lines.
#leg.set_draggable(True)
ax.grid(which = "both")
ax.minorticks_on()
plt.ylabel('S/N')
plt.xlabel('Wavelength (nm)')
plt.show()
'''

