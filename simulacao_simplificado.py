# # dados simplificados 
# import numpy as np

# # Parâmetros do tempo e canais
# fs = 256  # Frequência de amostragem (Hz)
# t = np.arange(0, 10, 1/fs)  # Tempo total de 10 segundos

# # Canal 1: Sinal base
# canal_1 = np.sin(2 * np.pi * 5 * t)  # Sinal senoidal de 5 Hz

# # Canal 2: Mesma forma de onda, mas com 1 segundo de atraso
# delay = int(fs * 1)  # 1 segundo de atraso
# canal_2 = np.roll(canal_1, delay)

# # Verificar os sinais
# import matplotlib.pyplot as plt
# plt.plot(t[:1024], canal_1[:1024], label="Canal 1")
# plt.plot(t[:1024], canal_2[:1024], label="Canal 2 (1 segundo de atraso)")
# plt.legend()
# plt.show()

# k_history = 256

# # Dados simulados
# EEG = np.vstack((canal_1, canal_2)).T  # Linhas = tempo, colunas = canais

# Importando a biblioteca JIDT
from jpype import *
import numpy as np

# Caminho para o JAR da JIDT
jarLocation = "lib/jidt/infodynamics.jar"

if not isJVMStarted():
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation, convertStrings=True)

# # Configurando os dados para o cálculo da TE
# source = JArray(JDouble, 1)(EEG[:, 0].tolist())  # Canal 1
# destination = JArray(JDouble, 1)(EEG[:, 1].tolist())  # Canal 2 (1 segundo de atraso)

# # Inicializando o calculador de Transferência de Entropia
# calcClass = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
# calc = calcClass()
# calc.initialise(k_history)  # Usando k_history = 256 para 1 segundo de atraso

# # Fornecendo os dados para o cálculo
# calc.setObservations(source, destination)

# # Computando a TE
# result = calc.computeAverageLocalOfObservations()
# print("TE_Kernel(Canal 1 -> Canal 2) = %.4f bits" % result)
# DEU ERRADO

# import numpy as np
# import matplotlib.pyplot as plt

# # Parâmetros de simulação
# fs = 256  # Frequência de amostragem (256 Hz)
# t = np.arange(0, 10, 1/fs)  # Tempo total de 10 segundos (2560 amostras)

# # Canal 1: Sinal base (senoidal simples com ruído)
# canal_1 = np.sin(2 * np.pi * 5 * t) + 0.2 * np.random.randn(len(t))  # Sinal senoidal de 5 Hz com ruído

# # Canal 2: Mesma forma de onda, mas com atraso de 1 segundo e mais ruído não linear
# canal_2 = np.roll(canal_1, 256) + 0.1 * np.random.randn(len(t))**3  # Atraso de 1 segundo e distorção não linear

# # Plotando para verificar o atraso
# plt.plot(t[:1024], canal_1[:1024], label="Canal 1")
# plt.plot(t[:1024], canal_2[:1024], label="Canal 2 (atrasado 1 segundo com não linearidade)")
# plt.legend()
# plt.show()

# # Organizar os dados no formato correto para JIDT
# EEG = np.vstack((canal_1, canal_2)).T  # Linhas = tempo, Colunas = canais

# # Converter os dados para o formato JIDT
# source = JArray(JDouble, 1)(EEG[:, 0].tolist())  # Canal 1
# destination = JArray(JDouble, 1)(EEG[:, 1].tolist())  # Canal 2

# # Inicializar o calculador de Transferência de Entropia
# calcClass = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
# calc = calcClass()
# calc.initialise()  # Ajustar k_history para 256 (1 segundo de atraso)

# # Passar os dados para o calculador
# calc.setObservations(source, destination)

# # Calcular a Transferência de Entropia
# result = calc.computeAverageLocalOfObservations()
# print("Transferência de Entropia (Canal 1 -> Canal 2) = %.4f bits" % result)


# from scipy.signal import correlate

# # Calcular a correlação cruzada entre os sinais
# corr = correlate(canal_1, canal_2, mode='full', method='auto')
# lags = np.arange(-len(canal_1) + 1, len(canal_1))
# lag_at_max_corr = lags[np.argmax(corr)]

# print(f"Máxima correlação cruzada ocorre com atraso de {lag_at_max_corr/fs:.2f} segundos")


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from jpype import JArray, JPackage

# Parâmetros de simulação
fs = 256  # Frequência de amostragem (256 Hz)
t = np.arange(0, 10, 1/fs)  # Tempo total de 10 segundos (2560 amostras)

# Canal 2: Sinal base (senoidal simples com ruído)
canal_2 = np.sin(2 * np.pi * 5 * t) + 0.2 * np.random.randn(len(t))  # Sinal senoidal de 5 Hz com ruído

# Canal 1: Atrasado em 1 segundo (256 amostras)
# Criando o canal 1 como cópia do canal 2 com atraso
canal_1 = np.concatenate((np.zeros(256), canal_2[:-256])) + 0.1 * np.random.randn(len(t))  # Canal 1 (atrasado)

# Plotando para verificar o atraso
plt.figure(figsize=(10, 5))
plt.plot(t[:1024], canal_1[:1024], label="Canal 1 (atrasado 1 segundo)")
plt.plot(t[:1024], canal_2[:1024], label="Canal 2")
plt.legend()
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.title("Sinais com Atraso de 1 Segundo")
plt.show()

# Calcular a correlação cruzada para verificar o atraso
corr = correlate(canal_2, canal_1, mode='full', method='auto')
lags = np.arange(-len(canal_2) + 1, len(canal_2))
lag_at_max_corr = lags[np.argmax(corr)]

print(f"Máxima correlação cruzada ocorre com atraso de {lag_at_max_corr/fs:.2f} segundos")

# Organizar os dados no formato correto para JIDT (linhas = tempo, colunas = canais)
EEG = np.vstack((canal_2, canal_1)).T  # Linhas = tempo, Colunas = canais

# Converter os dados para o formato JIDT
source = JArray(JDouble, 1)(EEG[:, 0].tolist())  # Canal 2
destination = JArray(JDouble, 1)(EEG[:, 1].tolist())  # Canal 1

# Inicializar o calculador de Transferência de Entropia
calcClass = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
calc = calcClass()
calc.initialise(256)  # Ajustar k_history para 256 (1 segundo de atraso)

# Passar os dados para o calculador
calc.setObservations(source, destination)

# Calcular a Transferência de Entropia
result = calc.computeAverageLocalOfObservations()
print("Transferência de Entropia (Canal 2 -> Canal 1) = %.4f bits" % result)




