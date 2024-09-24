import idtxl.bivariate_te
import numpy as np
import matplotlib.pyplot as plt
import idtxl
import seaborn as sns

# Define the 18 EEG channels based on the 10/20 system
channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'O1', 'O2']

# Time specifications
Fs = 256  # Sampling frequency in Hz
dt = 1 / Fs  # Time step
T = 10  # Total time in seconds
t = np.arange(0, T, dt)  # Time vector

# Initialize the EEG data matrix
EEG = np.zeros((len(channels), len(t)))

# Generate non-stationary and non-linear signals for each channel
for i in range(len(channels)):
    # Create a non-stationary and non-linear signal
    EEG[i, :] = (0.5 * np.sin(2 * np.pi * 0.5 * t + np.random.rand() * 2 * np.pi) +
                 0.3 * np.sin(2 * np.pi * 1.5 * t + np.random.rand() * 2 * np.pi) +
                 0.2 * np.sin(2 * np.pi * 3 * t + np.random.rand() * 2 * np.pi) +
                 0.1 * np.random.randn(len(t)))

# Introduce synchrony with phase delay between specific pairs of channels
synchrony_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]  # Pairs of channels to synchronize (in Python, indices start at 0)
phase_delays = [0.1, 0.2, 0.3, 0.4, 0.5]  # Phase delays in seconds

for k, (ch1, ch2) in enumerate(synchrony_pairs):
    delay_samples = int(round(phase_delays[k] * Fs))
    
    # Apply phase delay to the second channel of the pair
    if delay_samples < len(t):
        EEG[ch2, delay_samples:] = EEG[ch1, :-delay_samples]

# Calculate the transfer entropy between pairs of variables
numChannels = len(channels)
te_matrix = np.zeros((numChannels, numChannels))
for i in range(numChannels):
    for j in range(numChannels):
        if i != j:
            data_i = EEG[i, :]
            data_j = EEG[j, :]
            # settings = idtxl.IdtxlSettings()
            # settings.default_settings()
            # settings.verbose = False
            te = idtxl.bivariate_te(data_i, data_j, mildest=1)
            te_matrix[i, j] = te
            print(f'Transfer entropy from Channel {i+1} to Channel {j+1}: {te}')

# Visualize the transfer entropy matrix
plt.figure(figsize=(10, 8))
sns.heatmap(te_matrix, annot=True, cmap='viridis', xticklabels=[f'C{i+1}' for i in range(numChannels)], yticklabels=[f'C{i+1}' for i in range(numChannels)])
plt.title('Transfer Entropy Matrix (TE) between EEG channels')

# Plot the generated EEG signals
plt.figure(figsize=(10, 15))
for i in range(len(channels)):
    plt.subplot(len(channels), 1, i + 1)
    plt.plot(t, EEG[i, :])
    plt.title(f'Channel {channels[i]}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()