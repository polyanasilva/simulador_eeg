# nesse código tentarei traduzir o arquivo sim_EEG_4m. para o formato python

# Description: This code simulates a multi_channel EEG recording with high and low frquency components including artifacts and correlated channels. It also adds an event channel to simulate "right" and "wrong" trials and saves the simulated EEG data and parameters into a .mat file.

import numpy as np
from math import sin, pi


# Defining simulation parameters
Fs = 1000                           # Sampling frequency in Hz
T = 30                              # Length of the signal in seconds
t = np.arange(0, T-1/Fs, 1/Fs)      # Time vector
numChannels = 16                    # Number of EEG channels to simulate

# Denine correlated channels
correlatedChannels = [1, 2, 3]      # Channels to be correlated
correlationStrength = 0.7           # Corretation fcoefficient (0 to 1)

# Define frequencies and amplitudes for high-frequency components
frequencies = [5 ,10, 20, 40, 60, 70] #Frequencies in Hz
amplitudes = [1, 0.5, 0.25, 0.125, 1, 0.5]

# Define artifact parameters
eyeBlinkFrequency = 1          # Frequency of eye blink artifact
eyeBlinkAmplitude = 10         # Amplitude of eye blink artifact
motionArtifactFrequency = 0.5  # Frequency of motion artifact
motionArtifactAmplitude = 5    # Amplitude of motion artifact
glossokineticFrequency = 0.1   # Frequency of glossokinetic artifact
glossokineticAmplitude = 4     # Amplitude of glossokinetic artifact

# Initialize the EEF data matrix
simulatedEEG = np.zeros(numChannels, len(t))

# Generate correated noise for specified channels
correlatedNoise = np.random.randn(len(correlatedChannels), len(t))
R = np.eye(len(correlatedChannels)) * (1-correlationStrength) + correlationStrength
L = np.linalg.cholesky(R)
correlatedNoise *= L

# Simulate EEG for each channel
for channel in numChannels:
    # Add high-frequency components
    for i in len(frequencies):
        simulatedEEG[channel, :] = simulatedEEG[channel, :] + amplitudes(i) * sin(2 * pi * frequencies(i) * t)
    
    # Add correlated or uncorrelated noise
    if channel in correlatedChannels:
        idx = np.where(correlatedChannels == channel)[0]
        simulatedEEG[channel, :] += correlatedNoise[idx, :]
    else:
        noiseAmplitude = 0.2
        simulatedEEG[channel, :] += noiseAmplitude * np.random.randn(simulatedEEG.shape[1])

    # Add low-frquency artifacts at random times
    # Eye blink artifact
    onset = np.random.randint(1, len(t) - Fs + 1) # Random onset time for the artifact
    duration = 0.25                               # Duration of the eye blink in seconds
    blinkSignal = eyeBlinkAmplitude * sin(2 * pi * eyeBlinkFrequency * t[onset:onset+round(duration*Fs)])
    simulatedEEG[channel, onset:onset+len(blinkSignal)] += blinkSignal

    # Motion artifact
    onset = np.random.randint(1, len(t)- Fs + 1) # Random onset time for the artifact
    duration = 0.5                               # Duration of the motion artifact in seconds
    motionSignal = motionArtifactAmplitude * sin(2 * pi * motionArtifactFrequency * t[onset:onset+round(duration*Fs)])
    simulatedEEG[channel, onset:onset+len(motionSignal)] += motionSignal

    # Flossokinetic artifact
    onset = np.random.randint(1, len(t) - Fs + 1) # Random onset time for the artifact
    duration = 1
    glossokineticSignal = glossokineticAmplitude * sin(2 * pi * glossokineticFrequency * t[onset:onset+round(duration*Fs)])
    simulatedEEG[channel, onset:onset+len(glossokineticSignal)] += glossokineticSignal

# Add event channel for "right" and "wrong" trials
eventChannels = np.zeros(1, len(t))
numTrials = 20
trialDuration = 1
trialSamples = trialDuration * Fs

# for i in numTrials
#     trialStart = np.random.randint(1, len(t) - trialSamples)
#     if