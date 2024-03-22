# Introduction
The code utilizes the Python language to estimate the signal-to-noise ratio (SNR) of the AMASE-P system. Simply run the SNR.py file directly and follow the instructions.

# Assumption
In the calculation of the noise sky background part, the code assumes that the data from MaNGA is 21.5 mag/arcsec<sup>2</sup>.

# General Usage
## Default setting
The default setting of the system uses the Canon 400mm f/2.8 telephoto lens + 50 micron core fiber + Nikkor 58mm f/0.95 S Noct lens + QHY600M CMOS detector. The default setting takes into account various factors such as lens throughput, vignetting of the Nikkor lens, detector quantum efficiency (QE), etc. The spot size is based on Zemax simulation analysis rather than theoretical values.

## Custom setting
If a custom design is required, the current code can analyze custom telescopes, fibers, spectrograph cameras, and detectors. Other components such as gratings and dichroic beam splitters are not expected to be changed. The limitation of the custom setting is that **the effectiveness will not be considered**, and the spot size calculation is based on theoretical calculations.

### Advanced custom setting
Basic estimation using the custom setting is sufficient. However, if you have advanced information such as the throughput of a new camera and want more accurate results, you can modify the code in the 'setting' part. For example, to add a new detector, you can add a new dictionary in the code. Taking the detector as an example. Suppose you have a detector called "New detector," then you can modify the code as follows and **please do not change the code in "Custom"**:
```
detector={
    "QHY600":{
        "QE" : current_path+"/data/QE_132.txt",
        "pixel size": (3.76,3.76), # um
        "readnoise": 1.683,
        "darkcurrent": 0.0022 # T=-20 degree
    },
    "New detector":{
        "QE" : path_of_QE,
        "pixel size": (5,5),
        "readnoise": 1.8,
        "darkcurrent": 0.004
    },
    "Custom":{
        "QE" : None,
        "pixel size": None,
        "readnoise": None,
        "darkcurrent": None
    }
}
```
In this example, the format of the QE data can refer to the \data file. If a different format is needed, such as a throughput with a new range of wavelengths instead of the default range, you may need to modify the code for 'text file reading' part because the default reading file code may not work in that case.
