#%% Packete

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tvb.simulator.lab import *

from tvb.datatypes.cortex import Cortex
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.projections import ProjectionMatrix, ProjectionSurfaceEEG
from tvb.datatypes.sensors import SensorsEEG

#%% Messwerte laden

raw1= np.load('Messreihen/tavg_raw1.npy')
time = np.load('Messreihen/tavg_time.npy')

#%% Plotten der raw Messwerte EEG
plt.figure()
plt.plot(time, raw1[:,1], 'k', alpha=0.1)
plt.grid(True)
plt.xlabel('Time (ms)')
plt.ylabel("Temporal average")

#%% Display the local connectivity kernel definining the local coupling strength for one focal point

default_cortex = Cortex.from_file(region_mapping_file='regionMapping_16k_76.txt')
white_matter = connectivity.Connectivity.from_file('connectivity_76.zip')
local_coupling_strength = np.array([2 ** -10])
default_cortex.region_mapping_data.connectivity = white_matter
default_cortex.coupling_strength = local_coupling_strength


#plt.figure()
#ax = plt.subplot(111, projection='3d')
#x, y, z = default_cortex.vertices.T
#ax.plot_trisurf(x, y, z, triangles=default_cortex.triangles, alpha=0.1, edgecolor='none')

#%% Plot Sensorpositionen

conn = connectivity.Connectivity.from_file('connectivity_76.zip')
conn.configure()
skin = surfaces.SkinAir.from_file()
skin.configure()
sens_eeg = sensors.SensorsEEG.from_file('eeg_unitvector_62.txt.bz2')
sens_eeg.configure()

plt.figure()
ax = plt.subplot(111, projection='3d')

# ROI centers as black circles
x, y, z = conn.centres.T
plt.plot(x, y, z, 'ko')

# EEG sensors as blue x's
x, y, z = sens_eeg.sensors_to_surface(skin).T
plt.plot(x, y, z, 'bx')

# Plot boundary surface
x, y, z = skin.vertices.T
ax.plot_trisurf(x, y, z, triangles=skin.triangles, alpha=0.1, edgecolor='none')

# Plot Cortex
x, y, z = default_cortex.vertices.T
ax.plot_trisurf(x, y, z, triangles=default_cortex.triangles, alpha=0.1, edgecolor='none')

#%% Plotten der Connectivity Matrix

white_matter.configure()
white_matter.summary_info
plot_connectivity(connectivity = white_matter)
