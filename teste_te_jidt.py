from jpype import *
import numpy as np
import matplotlib.pyplot as plt

# Localização do JAR e inicialização da JVM
jarLocation = "lib/jidt/infodynamics.jar"
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

# Exemplo de carregamento de dados de EEG
# Aqui, estou assumindo que os dados estão em um formato numpy. Adapte conforme necessário.
# Por exemplo, se usar MNE:
# import mne
# raw = mne.io.read_raw_edf('seu_arquivo.edf')
# eeg_data = raw.get_data()  # Isso retorna um array numpy (canais x tempo)

# Simulação de dados de EEG para exemplo:
# Substitua 'eeg_data' pelo seu carregamento de dados
eeg_data = np.random.randn(19, 750)  # 19 canais, 750 amostras (750 s a 1 Hz)

# Selecione dois canais para o cálculo de TE (modifique conforme necessário)
source_channel = eeg_data[0, :]  # Canal de origem (ex: canal 1)
dest_channel = eeg_data[1, :]    # Canal de destino (ex: canal 2)

# Criação da variável de origem e destino com dados de EEG reais
sourceArray = source_channel  # Substitua com o canal de origem desejado
destArray = dest_channel      # Substitua com o canal de destino desejado

# Criação do objeto de cálculo de TE e configuração
teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
teCalc = teCalcClass()
teCalc.setProperty("NORMALISE", "true")  # Normaliza os dados (opcional)
teCalc.initialise(1)  # Use história de comprimento 1
teCalc.setProperty("k", "4")  # Define o parâmetro K para 4 pontos mais próximos

# Configuração das observações com dados de EEG
# Aqui, assumo que sourceArray e destArray são listas numpy
teCalc.setObservations(JArray(JDouble, 1)(sourceArray.tolist()), JArray(JDouble, 1)(destArray.tolist()))

# Cálculo da entropia de transferência média
result = teCalc.computeAverageLocalOfObservations()
print(f"Resultado TE: {result:.4f} nats")

# Cálculo dos valores locais de TE
localTE = teCalc.computeLocalOfPreviousObservations()
print(f"TE local (média): {np.mean(localTE):.4f} nats")

# Encerramento da JVM
shutdownJVM()

# Plotando os sinais
plt.figure(figsize=(12, 6))

# Plot do Source Array
plt.subplot(2, 1, 1)
plt.plot(sourceArray, color='blue')
plt.title('Source Array')
plt.xlabel('Observações')
plt.ylabel('Amplitude')

# Plot do Dest Array
plt.subplot(2, 1, 2)
plt.plot(destArray, color='green')
plt.title('Dest Array')
plt.xlabel('Observações')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()