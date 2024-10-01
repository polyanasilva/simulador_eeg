from jpype import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from EEGdata import *

# Localização do JAR e inicialização da JVM
jarLocation = "lib/jidt/infodynamics.jar"
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

jieeg = EEGdata(file_name='simulateEEG')
data = jieeg.get_data()
tempo = jieeg.create_data()
channels = jieeg.get_channels()
parameters = jieeg.get_parameters()

# # Criação do objeto de cálculo de TE e configuração
teCalcClass = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
teCalc = teCalcClass()
# teCalc.setProperty("NORMALISE", "true")  # Normaliza os dados (opcional)
teCalc.initialise()  # Use história de comprimento 1
# teCalc.setProperty("k", "10")  # Define o parâmetro K para 4 pontos mais próximos
# teCalc.setProperty("LOG_BASE", "2")  # Base de logaritmo em 2 (bits)
# teCalc.setProperty("AUTO_EMBED_METHOD", "MAX_CORR_AIS")  # Embedding otimizado para maximizar a correlação
# teCalc.setProperty("L", "256")  # Definir o atraso em amostras (1 segundo se Fs = 1000 Hz)



# Configuração das observações com dados de EEG
# Aqui, assumo que sourceArray e destArray são listas numpy
teCalc.setObservations(JArray(JDouble, 1)(data[0].tolist()), JArray(JDouble, 1)(data[1].tolist()))

# Cálculo da entropia de transferência média
result = teCalc.computeAverageLocalOfObservations()
print(f"Resultado TE: {result:.4f} bits")

# Cálculo dos valores locais de TE
localTE = teCalc.computeLocalOfPreviousObservations()
print(f"TE local (média): {np.mean(localTE):.4f} bits")

# Encerramento da JVM
shutdownJVM()

# Plotar os sinais de EEG gerados
plt.figure(figsize=(10, 15))
for i in range(len(channels)):
    plt.subplot(len(channels), 1, i + 1)
    plt.plot(tempo, data[i, :])
    plt.title(f'Canal {channels[i]}')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# testei com e sem normalizacao e nesse caso deu o mesmo resultado