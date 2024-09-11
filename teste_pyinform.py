# pyinform

import numpy as np
import pyinform
import matplotlib.pyplot as plt
import seaborn as sns

# Função dinâmica (mantendo a mesma estrutura para gerar dados)
def dyn(x1, x2, x3, x4):
    x1 = -0.287 * x2 + np.random.normal()
    x2 = 0.4 * x2 + 0.287 * x1 + np.random.normal()
    x3 = 0.9 * x3 + np.random.normal()
    x4 = 0.9 * x2 + np.random.normal()
    return x1, x2, x3, x4

# Geração inicial dos dados (inicializa x1, x2, x3, x4)
x1_ini, x2_ini, x3_ini, x4_ini = np.random.rand(4, 1) 
timestep = 200
data = np.expand_dims(np.concatenate((x1_ini, x2_ini, x3_ini, x4_ini), axis=0), axis=0) # shape inicial [1, 4]

# Simulação do sistema dinâmico por 200 timesteps
for step in range(timestep):
    x1_ini, x2_ini, x3_ini, x4_ini = dyn(x1_ini, x2_ini, x3_ini, x4_ini)
    temp = np.expand_dims(np.concatenate((x1_ini, x2_ini, x3_ini, x4_ini), axis=0), axis=0)
    data = np.concatenate((data, temp), axis=0) 

# Verificando o shape do dado gerado (deveria ser [201, 4])
print(f"Shape dos dados: {data.shape}")

# Quantização dos dados (necessário para o PyInform, por exemplo usando 5 níveis)
def quantize_data(data, num_bins=5):
    quantized_data = np.digitize(data, np.linspace(np.min(data), np.max(data), num_bins))
    return quantized_data

quantized_data = quantize_data(data)

# Calculando a entropia de transferência entre pares de variáveis
# Exemplo: Transferência de x1 para x2, x2 para x3, etc.

te_matrix = np.zeros((4, 4))  # Para armazenar entropia de transferência entre todas as variáveis
for i in range(4):
    for j in range(4):
        if i != j:  # Evita calcular entropia de transferência da variável para si mesma
            te = pyinform.transferentropy.transfer_entropy(quantized_data[:, i], quantized_data[:, j], k=1)
            te_matrix[i, j] = te
            print(f"Entropia de transferência de x{i+1} para x{j+1}: {te}")

print("\nMatriz de Entropia de Transferência (TE):")
print(te_matrix)

# Visualização da matriz de entropia de transferência

plt.figure(figsize=(8, 6))
sns.heatmap(te_matrix, annot=True, cmap="viridis", xticklabels=['x1', 'x2', 'x3', 'x4'], yticklabels=['x1', 'x2', 'x3', 'x4'])
plt.title('Matriz de Entropia de Transferência (TE)')
plt.show()
