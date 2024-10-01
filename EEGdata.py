from jpype import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from CSVFileManager import *


class EEGdata:

    PATH = "data/" # pasta para armazenas os dados
    EEG = [] # iniciando a matriz
   
    # Definir os 18 canais de EEG com base no sistema 10-20
    channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'O1', 'O2']

    # Fs = 256  # Frequência de amostragem em Hz
    # T = 10  # Tempo total em segundos

    def __init__(self, file_name, Fs = 256, T = 10):
        
        self.file_path = self.PATH + file_name + '.csv'
        self.fm = CSVFileManager(self.file_path)
        self.Fs = Fs
        self.T = T


    def get_data(self):

        # Verifica se o arquivo existe antes de carregar
        if self.fm.exists():
            # Se o arquivo existe, carrega os dados
            df = self.fm.read()
            self.EEG = df.to_numpy()

        else:
            data = self.create_data
            df = pd.DataFrame(data)
            self.fm.write(df)
            self.EEG = data
        
        return self.EEG
    

    def get_channels(self):

        return self.channels


    def get_parameters(self):

        return {
            'Fs': self.Fs,
            'T': self.T
        }


    def create_data(self):    

        # Especificações de tempo
        dt = 1 / self.Fs  # Passo de tempo
        t = np.arange(0, self.T, dt)  # Vetor de tempo

        # Inicializar a matriz de dados de EEG
        data = np.zeros((len(self.channels), len(t)))

        # Gerar sinais não estacionários e não lineares para cada canal
        for i in range(len(self.channels)):
            # Criar um sinal não estacionário e não linear
            data[i, :] = (0.5 * np.sin(2 * np.pi * 0.5 * t + np.random.rand() * 2 * np.pi) +
                        0.3 * np.sin(2 * np.pi * 1.5 * t + np.random.rand() * 2 * np.pi) +
                        0.2 * np.sin(2 * np.pi * 3 * t + np.random.rand() * 2 * np.pi) +
                        0.1 * np.random.randn(len(t)))
            
        # Introduzir sincronia com atraso de fase entre pares específicos de canais
        synchrony_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17)]  # Pares de canais para sincronizar (em Python, índices começam em 0)
        phase_delays = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Atrasos de fase em segundos
        # phase_delays = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # Atrasos de fase em segundos
        # synchrony_pairs = [(0, 1)]
        # phase_delays = [0.1] 

        for k, (ch1, ch2) in enumerate(synchrony_pairs):
            delay_samples = int(round(phase_delays[k] * self.Fs))
            
            # Aplicar atraso de fase ao segundo canal do par
            if delay_samples < len(t):
                data[ch2, delay_samples:] = data[ch1, :-delay_samples]
    
        return t


    def remove_data(self):
        self.fm.delete()
        



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