import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import idtxl
import jpype as jp

from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
from idtxl.visualise_graph import plot_network

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
    EEG[i, :] = (0.5 * np.sin(2 * np.pi * 0.5 * t + np.random.rand() * 2 * np.pi) +
                 0.3 * np.sin(2 * np.pi * 1.5 * t + np.random.rand() * 2 * np.pi) +
                 0.2 * np.sin(2 * np.pi * 3 * t + np.random.rand() * 2 * np.pi) +
                 0.1 * np.random.randn(len(t)))

# Introduzir sincronia com atraso de fase entre pares específicos de canais
synchrony_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
phase_delays = [0.1, 0.2, 0.3, 0.4, 0.5]

for k, (ch1, ch2) in enumerate(synchrony_pairs):
    delay_samples = int(round(phase_delays[k] * Fs))
    
    if delay_samples < len(t):
        EEG[ch2, delay_samples:] = EEG[ch1, :-delay_samples]

# Usando o IDTxl para calcular a entropia de transferência
settings = {
    'cmi_estimator': 'JidtGaussianCMI',  # Estimador para dados contínuos
    'max_lag_sources': 1,  # Lag máximo
    'min_lag_sources': 1,  # Lag mínimo
    'alpha_max': 0.05,  # Nível de significância
}

# Preparar os dados para o IDTxl
# A entrada deve estar na forma (samples, variables, trials)
data = Data(EEG.T, dim_order='ps')

# Inicializar o cálculo de TE
network_analysis = MultivariateTE()

# Executar a análise TE
results = network_analysis.analyse_network(settings=settings, data=data)

# Extrair a matriz de TE
te_matrix = np.zeros((len(channels), len(channels)))
for link in results.results['weights']:
    source = link[0]
    target = link[1]
    te_matrix[source, target] = results.results['weights'][(source, target)]

# Visualizando a matriz de entropia de transferência
plt.figure(figsize=(10, 8))
sns.heatmap(te_matrix, annot=True, cmap='viridis', xticklabels=channels, yticklabels=channels)
plt.title('Matriz de Entropia de Transferência (TE) entre canais de EEG')
plt.show()

# Visualizando os sinais de EEG gerados
plt.figure(figsize=(10, 15))
for i in range(len(channels)):
    plt.subplot(len(channels), 1, i + 1)
    plt.plot(t, EEG[i, :])
    plt.title(f'Canal {channels[i]}')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()
