# Tentativa de aplicação no tigramite no simEEG4

import numpy as np
import matplotlib.pyplot as plt
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import parcorr

# Definindo parâmetros de simulação
Fs = 100
T = 30
t = np.arange(0, T, 1/Fs)
numChannels = 16

# Definindo canais correlacionados
correlatedChannels = [1, 2, 3]
correlationStrength = 0.7

# Definindo frequencias e ampitudes para componentes de alta frequencia
frequencies = [5, 10, 20, 40, 60, 70]
amplitudes = [1, 0.5, 0.25, 0.125, 1, 0.5]

# Definindo parametros de artefatos
eyeBlinkFrequency = 1
eyeBlinkAmplitude = 10
motionArtifactFrequency = 0.5
motionArtifactAmplitude = 5
glossokineticFrequency = 0.1  
glossokineticAmplitude = 4  

# Inicializar a matriz de dados de EEG
simulatedEEG = np.zeros((numChannels, len(t)))

# Gerar ruido correlacionado para os canais especificos
correlatedNoise = np.random.randn(len(correlatedChannels), len(t))
R = np.eye(len(correlatedChannels)) * (1 - correlationStrength) + correlationStrength
L = np.linalg.cholesky(R)
correlatedNoise = L @ correlatedNoise

# Simular EEG para cada canal
for channel in range(numChannels):
    # adcionar componentes de alta frequencia
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

# Inclusao do calculo de causualidade sobre esses dados simulados
# conversão para que possa ser analisado pelo tigramite
var_names = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8', 'Ch9', 'Ch10', 'Ch11', 'Ch12', 'Ch13', 'Ch14','Ch15', 'Ch16']
dataframe = pp.DataFrame(simulatedEEG,
                         datatime=np.arange(len(simulatedEEG)),
                         var_names=var_names)

tp.plot_timeseries(dataframe, label_fontsize=20, tick_label_size=10)

Parcorr = parcorr.ParCorr(significance='analytic')
pcmci = PCMCI(
    dataframe = dataframe
    cond_ind_test= Parcorr
    verbosity= 1
)
pcmci.verbosity = 1
tau_max = 2
results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=None)
q_matriz = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], tau_max=tau_max, fdr_method='fdr_bh')

tp.plot_graph(graph = results['graph'],
              val_matrix= results['val_matriz'])



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
