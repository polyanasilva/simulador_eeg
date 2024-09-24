# nesse código traduzi o arquivo sim_EEG_4m. para o formato python

# Description: This code simulates a multi_channel EEG recording with high and low frquency components including artifacts and correlated channels. It also adds an event channel to simulate "right" and "wrong" trials and saves the simulated EEG data and parameters into a .mat file.

# facilitado

import numpy as np
import matplotlib.pyplot as plt

# Definindo parâmetros de simulação
Fs = 1000  # Frequência de amostragem em Hz
T = 30     # Duração do sinal em segundos
t = np.arange(0, T, 1/Fs)  # Vetor de tempo
numChannels = 16  # Número de canais de EEG a simular

# Definindo canais correlacionados
correlatedChannels = [1, 2, 3]  # Canais a serem correlacionados
correlationStrength = 0.7  # Coeficiente de correlação (0 a 1)

# Definindo frequências e amplitudes para componentes de alta frequência
frequencies = [5, 10, 20, 40, 60, 70]  # Frequências em Hz
amplitudes = [1, 0.5, 0.25, 0.125, 1, 0.5]

# Definindo parâmetros de artefatos
eyeBlinkFrequency = 1  # Frequência do artefato de piscar dos olhos
eyeBlinkAmplitude = 10  # Amplitude do artefato de piscar dos olhos
motionArtifactFrequency = 0.5  # Frequência do artefato de movimento
motionArtifactAmplitude = 5  # Amplitude do artefato de movimento
glossokineticFrequency = 0.1  # Frequência do artefato glossocinético
glossokineticAmplitude = 4  # Amplitude do artefato glossocinético

# Inicializar a matriz de dados EEG
simulatedEEG = np.zeros((numChannels, len(t)))

# Gerar ruído correlacionado para os canais especificados
correlatedNoise = np.random.randn(len(correlatedChannels), len(t))
R = np.eye(len(correlatedChannels)) * (1 - correlationStrength) + correlationStrength
L = np.linalg.cholesky(R)
correlatedNoise = L @ correlatedNoise

# Simular EEG para cada canal
for channel in range(numChannels):
    # Adicionar componentes de alta frequência
    for i in range(len(frequencies)):
        simulatedEEG[channel, :] += amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * t)
    
    # Adicionar ruído correlacionado ou não correlacionado
    if channel + 1 in correlatedChannels:
        idx = correlatedChannels.index(channel + 1)
        simulatedEEG[channel, :] += correlatedNoise[idx, :]
    else:
        noiseAmplitude = 0.2
        simulatedEEG[channel, :] += noiseAmplitude * np.random.randn(len(t))
    
    # Adicionar artefatos de baixa frequência em momentos aleatórios
    # Artefato de piscar dos olhos
    onset = np.random.randint(1, len(t) - Fs)
    duration = 0.25
    blinkSignal = eyeBlinkAmplitude * np.sin(2 * np.pi * eyeBlinkFrequency * t[onset:onset + round(duration * Fs)])
    simulatedEEG[channel, onset:onset + len(blinkSignal)] += blinkSignal
    
    # Artefato de movimento
    onset = np.random.randint(1, len(t) - Fs)
    duration = 0.5
    motionSignal = motionArtifactAmplitude * np.sin(2 * np.pi * motionArtifactFrequency * t[onset:onset + round(duration * Fs)])
    simulatedEEG[channel, onset:onset + len(motionSignal)] += motionSignal
    
    # Artefato glossocinético
    onset = np.random.randint(1, len(t) - Fs)
    duration = 1
    glossokineticSignal = glossokineticAmplitude * np.sin(2 * np.pi * glossokineticFrequency * t[onset:onset + round(duration * Fs)])
    simulatedEEG[channel, onset:onset + len(glossokineticSignal)] += glossokineticSignal

# Adicionar canal de eventos para testes "correto" e "incorreto"
eventChannel = np.zeros(len(t))
numTrials = 20  # Número total de ensaios
trialDuration = 1  # Duração de cada ensaio em segundos
trialSamples = int(trialDuration * Fs)

for _ in range(numTrials):
    trialStart = np.random.randint(1, len(t) - trialSamples + 1)
    if np.random.rand() > 0.5:
        eventChannel[trialStart] = 1  # Ensaio "correto"
    else:
        eventChannel[trialStart] = 2  # Ensaio "incorreto"

# Combinar dados EEG e dados de eventos
simulatedEEG = np.vstack([simulatedEEG, eventChannel])

# Plotar o sinal EEG simulado para todos os canais
plt.figure(figsize=(15, 10))
for i in range(numChannels):
    plt.subplot(4, 4, i + 1)
    plt.plot(t, simulatedEEG[i, :])
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Canal {i + 1}')
plt.tight_layout()

# Plotar o canal de eventos
plt.figure(figsize=(10, 5))
plt.stem(t, simulatedEEG[-1, :], basefmt=" ")
plt.xlabel('Tempo (s)')
plt.ylabel('Tipo de Evento')
plt.title('Canal de Eventos')
plt.yticks([0, 1, 2], ['Sem Evento', 'Correto', 'Incorreto'])

plt.show()
