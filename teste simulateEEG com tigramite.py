# teste simulateEEG com tigramite

import numpy as np
import matplotlib.pyplot as plt
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import parcorr, gpdc, cmiknn

# Define the 18 EEG channels based on the 10-20 system
channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'O1', 'O2']

# Time specifications
Fs = 256  # Frequency (Hz)
dt = 1 / Fs  # Time step
total_time = 10  # Total time (s)
t = np.arange(0, total_time, dt)  # Time vector
EEG = np.zeros((len(channels), len(t))) # Initialize the EEG data matrix

# Generate non-stationary and non-linear signals for each channel
for i in range(len(channels)):
    EEG[i, :] = (0.5 * np.sin(2 * np.pi * 0.5 * t + np.random.rand() * 2 * np.pi) +
                0.3 * np.sin(2 * np.pi * 1.5 * t + np.random.rand() * 2 * np.pi) +
                0.2 * np.sin(2 * np.pi * 3 * t + np.random.rand() * 2 * np.pi) +
                0.1 * np.random.randn(len(t)))
    
# Introduce phase-lagged synchronization between specific pairs of channels
synchrony_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17)]  # Pares de canais para sincronizar (em Python, índices começam em 0)
phase_delays = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Atrasos de fase em segundos

for k, (ch1, ch2) in enumerate(synchrony_pairs):
    delay_samples = int(round(phase_delays[k] * Fs))
    
    # Apply phase lag to the second channel of the pair
    if delay_samples < len(t):
        EEG[ch2, delay_samples:] = EEG[ch1, :-delay_samples]


