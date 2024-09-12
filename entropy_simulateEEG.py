# tradução do simulate_eeg_channels

# Description:
# This function simulates 18 EEG channels where ther synchrony for some pairs of channels
# This synchrony occurs with some phase delay since a given signal takes time to go from one cortical area to another
# Each channel is a non-stationary, non-linear time series
# Posteriormente foi aplicado o cálculo da entropia de transferência sobre os dados simulados

# Assumptions and limitations:
# the code assumes a sampling frequency of 256 Hz and a total duration of 10 seconds for the EEG signal

import numpy as np
import matplotlib.pyplot as plt
import pyinform 
import seaborn as sns

# Definir os 18 canais de EEG com base no sistema 10/20
channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'O1', 'O2']

# Especificações de tempo
Fs = 256  # Frequência de amostragem em Hz
dt = 1 / Fs  # Passo de tempo
T = 10  # Tempo total em segundos
t = np.arange(0, T, dt)  # Vetor de tempo

# Inicializar a matriz de dados de EEG
EEG = np.zeros((len(channels), len(t)))

# Gerar sinais não estacionários e não lineares para cada canal
for i in range(len(channels)):
    # Criar um sinal não estacionário e não linear
    EEG[i, :] = (0.5 * np.sin(2 * np.pi * 0.5 * t + np.random.rand() * 2 * np.pi) +
                 0.3 * np.sin(2 * np.pi * 1.5 * t + np.random.rand() * 2 * np.pi) +
                 0.2 * np.sin(2 * np.pi * 3 * t + np.random.rand() * 2 * np.pi) +
                 0.1 * np.random.randn(len(t)))

# Introduzir sincronia com atraso de fase entre pares específicos de canais
synchrony_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]  # Pares de canais para sincronizar (em Python, índices começam em 0)
phase_delays = [1, 1, 1, 1, 1]  # Atrasos de fase em segundos

for k, (ch1, ch2) in enumerate(synchrony_pairs):
    delay_samples = int(round(phase_delays[k] * Fs))
    
    # Aplicar atraso de fase ao segundo canal do par
    if delay_samples < len(t):
        EEG[ch2, delay_samples:] = EEG[ch1, :-delay_samples]

# quantização dos dados
def quantize_data(data, num_bins=5):
    quantize_data = np.digitize(data, np.linspace(np.min(data), np.max(data), num_bins))
    return quantize_data

quantized_data = quantize_data(EEG)

# calculando a entropia de transferencia entre pares de variaveis
# exemplo: transferencia x1 para x2, x2 para x3, etc
numChannels = len(channels)
te_matrix = np.zeros((numChannels, numChannels))
for i in range(numChannels):
    for j in range(numChannels):
        if i != j: 
            te = pyinform.transferentropy.transfer_entropy(quantized_data[i, :], quantized_data[j, :], k=1)
            te_matrix[i,j] = te
            print(f'Entropia de transferência de Canal {i+1} para Canal {j+1}: {te}')

# visualizando a matriz de entropia de transferência
plt.figure(figsize=(10,8))
sns.heatmap(te_matrix, annot=True, cmap='viridis', xticklabels=[f'C{i+1}' for i in range(numChannels)], yticklabels=[f'C{i+1}' for i in range(numChannels)])
plt.title('Matriz de Entropia de Transferência (TE) entre canais de EEG')

# Plotar os sinais de EEG gerados
plt.figure(figsize=(10, 15))
for i in range(len(channels)):
    plt.subplot(len(channels), 1, i + 1)
    plt.plot(t, EEG[i, :])
    plt.title(f'Canal {channels[i]}')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()
