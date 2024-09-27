# Simulate EEG channels

# This function simulates 18 EEG channels where ther synchrony for some pairs of channels
# This synchrony occurs with some phase delay since a given signal takes time to go from one cortical area to another
# Each channel is a non-stationary, non-linear time series

# Assumptions and limitations:
# the code assumes a sampling frequency of 256 Hz and a total duration of 10 seconds for the EEG signal


import numpy as np
import pandas as pd

EEG_DATA_FILE_LOCATION = "data/simulateEEG.csv"


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
synchrony_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]  
phase_delays = [0.1, 0.2, 0.3, 0.4, 0.5]

for k, (ch1, ch2) in enumerate(synchrony_pairs):
    delay_samples = int(round(phase_delays[k] * Fs))
    
    # Apply phase lag to the second channel of the pair
    if delay_samples < len(t):
        EEG[ch2, delay_samples:] = EEG[ch1, :-delay_samples]

# Create the source and destination with real EEG data
df = pd.DataFrame(EEG)
df.to_csv(EEG_DATA_FILE_LOCATION, index=False)