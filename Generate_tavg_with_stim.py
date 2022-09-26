#%% Packete
import csv
import numpy as np
from numpy import zeros, newaxis

import pandas as pd 
from matplotlib import pyplot as plt

from tvb.simulator.lab import *

from tvb.datatypes.cortex import Cortex
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.projections import ProjectionMatrix, ProjectionSurfaceEEG
from tvb.datatypes.sensors import SensorsEEG

#%% Simulationsparameter
# Dauer der Simulation

Time = 30e3 #in ms

# Model
oscillator =  models.JansenRit()

# Connectivitymatrix/Conectom 
white_matter = connectivity.Connectivity.from_file('connectivity_76.zip')
labels = white_matter.region_labels

# Übertragungsfunktion
white_matter_coupling = coupling.SigmoidalJansenRit()

# Integrator
heunint = integrators.EulerDeterministic(dt=2e-3)

#%% Defining the Stimuli

# configure stimulus spatial pattern
weighting = np.zeros((76, )) # Array for 76 regins
weighting[[1,2,23,24,35,36,47,48,59,60]] = 1 # Definiere Gewicht

eqn_t = equations.Sinusoid()


stimulus = patterns.StimuliRegion(
    temporal=eqn_t,
    connectivity=white_matter,
    weight=weighting)

#Plot Stimuli
#Configure space and time

stimulus.configure_space()

stimulus.configure_time(np.arange(0., 65e3, 2**-1))

#And take a look
plot_pattern(stimulus)

#%% Monitore 

# EEG Abtastrate

#fsamp = 7.8125 #128 H
#fsamp =  1.953125 #512 Hz
fsamp = 0.244140625 # 4096 Hz => 0.244140625 ms


# Erstellen der Monitore
tavg = monitors.TemporalAverage(period=fsamp)
Counter =       monitors.ProgressLogger(period=2**-1)

#tavg Monitor
mon = (tavg,
        Counter,
)
#%% Alle Informationen für Simulation zusammen führen und ausführen
sim = simulator.Simulator(
    model=oscillator,
    connectivity=white_matter,
    coupling=white_matter_coupling,
    integrator=heunint,
    stimulus=stimulus,
    monitors=mon,
    simulation_length=Time,
).configure()

tavg, _ = sim.run()
  
# Wenn Surfase nicht auf default ist, dann wird für jedes Voxel ein Wert berechnet und nicht für jeden Knoten
#%% Daten speichern
#tavg ist ein Tuble Zeit und vier VIO (variables to watch/intrest)

tavg_time = tavg[0][:]
tavg_raw = np.squeeze(tavg[1][:])
tavg_raw = tavg_raw[:,0,:]


np.save('Messreihen/tavg_time_stim', tavg_time)
np.save('Messreihen/tavg_raw1_stim', tavg_raw)


#%% Plotten

plt.figure()
plt.plot(tavg_time, tavg_raw[:,:], 'k', alpha=0.1)
plt.plot(tavg_time, tavg_raw[:,:].mean(axis=1), 'r', alpha=1)
plt.grid(True)
plt.xlabel('Time (ms)')
plt.ylabel("Temporal average")
plt.show()

plt.figure()
plt.plot(tavg_time[:], tavg_raw[:, :] + np.r_[:76])
plt.title('Temporally Averaged Neuronal Activity', fontsize=12)
plt.xlabel('Time [ms]', fontsize=12)
plt.yticks(range(76), white_matter.region_labels[:], fontsize=8)
plt.yticks(range(76), white_matter.region_labels[:], fontsize=10)
plt.show()
