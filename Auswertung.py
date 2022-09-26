#%% Packete

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import welch
from numpy.fft import fft

from tvb.simulator.lab import *

from tvb.datatypes.cortex import Cortex
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.projections import ProjectionMatrix, ProjectionSurfaceEEG
from tvb.datatypes.sensors import SensorsEEG

white_matter = connectivity.Connectivity.from_file('connectivity_76.zip')
tavg_label = white_matter.region_labels

#%% Messwerte laden

raw1 = np.load('Messreihen/tavg_raw1.npy')
raw2 = np.load('Messreihen/tavg_raw2.npy')
raw3 = np.load('Messreihen/tavg_raw3.npy')
raw4 = np.load('Messreihen/tavg_raw4.npy')
time = np.load('Messreihen/tavg_time.npy')

raw11 = np.load('Messreihen/tavg_raw1_stim.npy')
raw22 = np.load('Messreihen/tavg_raw2_stim.npy')
raw33 = np.load('Messreihen/tavg_raw3_stim.npy')
raw44 = np.load('Messreihen/tavg_raw4_stim.npy')
time1 = np.load('Messreihen/tavg_time_stim.npy')

#%% Messdaten kürzen, normieren und zentrieren

delete = 1000
raw1 = raw1[delete:]
raw2 = raw2[delete:]
raw3 = raw3[delete:]
raw4 = raw4[delete:]
time = time[delete:]

raw11 = raw11[delete:]
raw22 = raw22[delete:]
raw33 = raw33[delete:]
raw44 = raw44[delete:]
time1 = time1[delete:]

raw1 -= np.mean(raw1,0) # Mittelwert bereinigt
raw1 /= (np.max(raw1,0) - np.min(raw1,0)) # Normiert
raw2 -= np.mean(raw2,0) # Mittelwert bereinigt
raw2 /= (np.max(raw2,0) - np.min(raw2,0)) # Normiert
raw3 -= np.mean(raw3,0) # Mittelwert bereinigt
raw3 /= (np.max(raw3,0) - np.min(raw3,0)) # Normiert

raw11 -= np.mean(raw11,0) # Mittelwert bereinigt
raw11 /= (np.max(raw11,0) - np.min(raw11,0)) # Normiert
raw22 -= np.mean(raw22,0) # Mittelwert bereinigt
raw22 /= (np.max(raw22,0) - np.min(raw22,0)) # Normiert
raw33 -= np.mean(raw33,0) # Mittelwert bereinigt
raw33 /= (np.max(raw33,0) - np.min(raw33,0)) # Normiert

#%% Plot Time

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(time[:], raw3[:, :] + np.r_[:76])
plt.title('Temporally Averaged Neuronal Activity', fontsize=12)
plt.xlabel('Time [ms]', fontsize=12)
plt.yticks(range(76), white_matter.region_labels[:], fontsize=8)
plt.yticks(range(76), white_matter.region_labels[:], fontsize=10)

plt.subplot(1, 2, 2)
plt.plot(time[:], raw33[:, :] + np.r_[:76])
plt.title('Temporally Averaged Neuronal Activity', fontsize=12)
plt.xlabel('Time [ms]', fontsize=12)
plt.yticks(range(76), white_matter.region_labels[:], fontsize=8)
plt.yticks(range(76), white_matter.region_labels[:], fontsize=10)
plt.show()


#%% Welch

f0, Pxx_den0  = welch(raw3,fs=128,nperseg=1000,axis=0,scaling='spectrum')
f1, Pxx_den1  = welch(raw33,fs=128,nperseg=1000,axis=0,scaling='spectrum')
N = len(raw1[0,:])
n = np.arange(N)

#%% FFT

fft0 = fft(raw3, axis=0)
fft1 = fft(raw33, axis=0)
N = len(raw1)
n = np.arange(N)
T = N/128
freq = n/T 


#%% Plot FFT and PDS

k0 = 0# Kanal
k1 = 3

stim = [1,2,23,24,35,36,47,48,59,60]

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(f0, Pxx_den0[:,k0:k1])
plt.xlim([0, 64])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.legend(tavg_label)
plt.title(label="PDS no stimulus",fontsize=12)


plt.subplot(2, 2, 2)
plt.plot( freq,np.abs(fft0[:,k0:k1]))
plt.xlim([0, 64])
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.legend(tavg_label)
plt.title(label="FFT no stimulus",fontsize=12)

plt.subplot(2, 2, 3)
plt.plot(f1, Pxx_den1[:,k0:k1])
plt.xlim([0, 64])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.legend(tavg_label)
plt.title(label="PDS with stimulus",fontsize=12)

plt.subplot(2, 2, 4)
plt.plot( freq,np.abs(fft1[:,k0:k1]))
plt.xlim([0, 64])
plt.xlabel('frequency [Hz]')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.legend(tavg_label)
plt.title(label="FFT with stimulus",fontsize=12)
plt.show()
