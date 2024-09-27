from jpype import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os



def get_eeg_data():
    LOCAL_ARQUIVO_DADOS_EEG = "data/dados_EEG.csv"
    EEG = []
    
    # Definir os 18 canais de EEG com base no sistema 10-20
    channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'O1', 'O2']
    # channels = ['Fp1', 'Fp2']

    # Especificações de tempo
    Fs = 256  # Frequência de amostragem em Hz
    dt = 1 / Fs  # Passo de tempo
    T = 10  # Tempo total em segundos
    t = np.arange(0, T, dt)  # Vetor de tempo

    # Verifica se o arquivo existe antes de carregar
    if os.path.exists(LOCAL_ARQUIVO_DADOS_EEG):
        # Se o arquivo existe, carrega os dados
        df = pd.read_csv(LOCAL_ARQUIVO_DADOS_EEG)
        EEG = df.to_numpy()

    else:
        # Localização do JAR e inicialização da JVM
        jarLocation = "lib/jidt/infodynamics.jar"
        startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

        

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
        # synchrony_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]  # Pares de canais para sincronizar (em Python, índices começam em 0)
        # phase_delays = [0.1, 0.2, 0.3, 0.4, 0.5]  # Atrasos de fase em segundos
        synchrony_pairs = [(0, 1)]
        phase_delays = [0.1] 

        for k, (ch1, ch2) in enumerate(synchrony_pairs):
            delay_samples = int(round(phase_delays[k] * Fs))
            
            # Aplicar atraso de fase ao segundo canal do par
            if delay_samples < len(t):
                EEG[ch2, delay_samples:] = EEG[ch1, :-delay_samples]

        # Criação da variável de origem e destino com dados de EEG reais
        # sourceArray = EEG[0] # colocar dps no calculo
        # destArray = EEG[1]
        df = pd.DataFrame(EEG)
        df.to_csv(LOCAL_ARQUIVO_DADOS_EEG, index=False)     
    
    return {
        'EEG':EEG, 
        'channels':channels, 
        'Fs':Fs, 
        'dt':dt, 
        'T':T, 
        't':t }


# # # Criação do objeto de cálculo de TE e configuração
# teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
# teCalc = teCalcClass()
# teCalc.setProperty("NORMALISE", "true")  # Normaliza os dados (opcional)
# teCalc.initialise(1)  # Use história de comprimento 1
# teCalc.setProperty("k", "4")  # Define o parâmetro K para 4 pontos mais próximos

# # Configuração das observações com dados de EEG
# # Aqui, assumo que sourceArray e destArray são listas numpy
# teCalc.setObservations(JArray(JDouble, 1)(sourceArray.tolist()), JArray(JDouble, 1)(destArray.tolist()))

# # Cálculo da entropia de transferência média
# result = teCalc.computeAverageLocalOfObservations()
# print(f"Resultado TE: {result:.4f} nats")

# # Cálculo dos valores locais de TE
# localTE = teCalc.computeLocalOfPreviousObservations()
# print(f"TE local (média): {np.mean(localTE):.4f} nats")

# # Encerramento da JVM
# shutdownJVM()



# # Plotar os sinais de EEG gerados
# plt.figure(figsize=(10, 15))
# for i in range(len(channels)):
#     plt.subplot(len(channels), 1, i + 1)
#     plt.plot(t, EEG[i, :])
#     plt.title(f'Canal {channels[i]}')
#     plt.xlabel('Tempo (s)')
#     plt.ylabel('Amplitude')
# plt.tight_layout()
# plt.show()

criar_Dados = get_eeg_data()