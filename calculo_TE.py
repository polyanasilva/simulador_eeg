from jpype import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import example4_kernel as jiEEG

# Localização do JAR e inicialização da JVM
jarLocation = "lib/jidt/infodynamics.jar"
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

# data = pd.read_csv('simulateEEG.csv')
data = jiEEG.get_eeg_data()

# # Criação do objeto de cálculo de TE e configuração
teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
teCalc = teCalcClass()
teCalc.setProperty("NORMALISE", "true")  # Normaliza os dados (opcional)
teCalc.initialise(1)  # Use história de comprimento 1
teCalc.setProperty("k", "4")  # Define o parâmetro K para 4 pontos mais próximos

# Configuração das observações com dados de EEG
# Aqui, assumo que sourceArray e destArray são listas numpy
teCalc.setObservations(JArray(JDouble, 1)(data['EEG'][0].tolist()), JArray(JDouble, 1)(data['EEG'][1].tolist()))

# Cálculo da entropia de transferência média
result = teCalc.computeAverageLocalOfObservations()
print(f"Resultado TE: {result:.4f} nats")

# Cálculo dos valores locais de TE
localTE = teCalc.computeLocalOfPreviousObservations()
print(f"TE local (média): {np.mean(localTE):.4f} nats")

# Encerramento da JVM
shutdownJVM()

numChannels = 2

# Plotar os sinais de EEG gerados
plt.figure(figsize=(10, 15))
for i in range(len(data['channels'])):
    plt.subplot(len(data['channels']), 1, i + 1)
    plt.plot(data['t'], data['EEG'][i, :])
    plt.title(f'Canal {data['channels'][i]}')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# testei com e sem normalizacao e nesse caso deu o mesmo resultado