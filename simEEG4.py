# Description: This code simulates a multi_channel EEG recording with high and low frquency components including artifacts and correlated channels. It also adds an event channel to simulate "right" and "wrong" trials and saves the simulated EEG data and parameters into a .csv file.


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Defining simulation parameters.
Fs = 1000  # Frequency (Hz)
dt = 1 / Fs  # Time step
total_time = 30  # Total time (s)
t = np.arange(0, total_time, dt)  # Time vector
numChannels = 16  # Number of EEG channels to simulate

# Define correlated channels
correlatedChannels = [1, 2, 3]  # Channels to be correlated
correlationStrength = 0.7  # Correlation coefficient (0 to 1)

# Defining frequencies and amplitudes for high-frequency components.
frequencies = [5, 10, 20, 40, 60, 70]
amplitudes = [1, 0.5, 0.25, 0.125, 1, 0.5]

# Defining artifact parameters
eyeBlinkFrequency = 1           # Eye blink artifact frequency
eyeBlinkAmplitude = 10          # Eye blink artifact amplitude
motionArtifactFrequency = 0.5   # Motion artifact frequency
motionArtifactAmplitude = 5     # Motion artifact amplitude
glossokineticFrequency = 0.1    # Glossokinetic artifact frequency
glossokineticAmplitude = 4      # Glossokinetic artifact amplitude"

# Initialize the EEG data matrix
simulatedEEG = np.zeros((numChannels, len(t)))

# Generate correlated noise for the specified channels
correlatedNoise = np.random.randn(len(correlatedChannels), len(t))
R = np.eye(len(correlatedChannels)) * (1 - correlationStrength) + correlationStrength
L = np.linalg.cholesky(R)
correlatedNoise = L @ correlatedNoise

# Simulate EEG for each channel
for channel in range(numChannels):
    # Add high-frequency components
    for i in range(len(frequencies)):
        simulatedEEG[channel, :] += amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * t)
    
    # Add correlated or uncorrelated noise
    if channel + 1 in correlatedChannels:
        idx = correlatedChannels.index(channel + 1)
        simulatedEEG[channel, :] += correlatedNoise[idx, :]
    else:
        noiseAmplitude = 0.2
        simulatedEEG[channel, :] += noiseAmplitude * np.random.randn(len(t))
    

    # Add low-frequency artifacts at random moments
    onset = np.random.randint(1, len(t) - Fs)
    duration = 0.25
    # Eye blink artifact
    blinkSignal = eyeBlinkAmplitude * np.sin(2 * np.pi * eyeBlinkFrequency * t[onset:onset + round(duration * Fs)])
    simulatedEEG[channel, onset:onset + len(blinkSignal)] += blinkSignal
    
    # Moviment artifact
    onset = np.random.randint(1, len(t) - Fs)
    duration = 0.5
    motionSignal = motionArtifactAmplitude * np.sin(2 * np.pi * motionArtifactFrequency * t[onset:onset + round(duration * Fs)])
    simulatedEEG[channel, onset:onset + len(motionSignal)] += motionSignal
    
    # Glossokinetic artifact
    onset = np.random.randint(1, len(t) - Fs)
    duration = 1
    glossokineticSignal = glossokineticAmplitude * np.sin(2 * np.pi * glossokineticFrequency * t[onset:onset + round(duration * Fs)])
    simulatedEEG[channel, onset:onset + len(glossokineticSignal)] += glossokineticSignal

# Add event channel for "correct" e "incorrect" tests
eventChannel = np.zeros(len(t))
numTrials = 20  # Total number of trial
trialDuration = 1  # Duration of each trial in seconds
trialSamples = int(trialDuration * Fs)

for _ in range(numTrials):
    trialStart = np.random.randint(1, len(t) - trialSamples + 1)
    if np.random.rand() > 0.5:
        eventChannel[trialStart] = 1  # Trial "correct"
    else:
        eventChannel[trialStart] = 2  # Trial "incorrect"

# Combine EEG data and event data
simulatedEEG = np.vstack([simulatedEEG, eventChannel])

# # Plot the simulated EEG signal for all channels
# plt.figure(figsize=(15, 10))
# for i in range(numChannels):
#     plt.subplot(4, 4, i + 1)
#     plt.plot(t, simulatedEEG[i, :])
#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude')
#     plt.title(f'Channel {i + 1}')
# plt.tight_layout()

# # Plot the event channels
# plt.figure(figsize=(10, 5))
# plt.stem(t, simulatedEEG[-1, :], basefmt=" ")
# plt.xlabel('Time (s)')
# plt.ylabel('Type of Event')
# plt.title('Channel of Event')
# plt.yticks([0, 1, 2], ['No Event', 'Correct', 'Incorrect'])

# plt.show()
