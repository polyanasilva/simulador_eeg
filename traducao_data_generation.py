# Tradução do Data_Generation para o Python
# Essa simulação gera dados artificiais de Neuro Mass Model

import numpy as np

# Número de ROIs
Npop = 3

# Definição do tempo
dt = 0.0001
f_eulero = 1 / dt
tend = 57  # 56 segundos, porque um segundo será excluído devido a efeitos transitórios
t = np.arange(0, tend + dt, dt)
N = len(t)

np.random.seed(11)  # Semente para o ruído

# Definição dos parâmetros
# Constantes de conectividade
C = np.zeros((Npop, 8))
C[:, 0] = 40 * np.ones(Npop)  # Cep
C[:, 1] = 40 * np.ones(Npop)  # Cpe
C[:, 2] = 40 * np.ones(Npop)  # Csp
C[:, 3] = 50 * np.ones(Npop)  # Cps
C[:, 4] = 20 * np.ones(Npop)  # Cfs
C[:, 5] = 40 * np.ones(Npop)  # Cfp
C[:, 6] = 60 * np.ones(Npop)  # Cpf
C[:, 7] = 20 * np.ones(Npop)  # Cff

# Definição das sinapses excitatórias e inibitórias
Wp = np.zeros((Npop, Npop))

# Sinapses excitatórias
Wp[1, 0] = 40

# Sinapses inibitórias
Wf = np.zeros((Npop, Npop))

e0 = 2.5  # Valor de saturação da sigmoide
r = 0.56  # Inclinação da sigmoide

D = 0.0166 * np.ones(Npop)  # Atraso entre as regiões (16ms)

a = np.array([75, 30, 300])  # Recíproco das constantes de tempo sinápticas (\omega)

G = np.array([5.17, 4.45, 57.1])  # Ganhos sinápticos

# Simulação
for trial in range(1, 11):
    print(trial)

    sigma = np.sqrt(9 / dt)  # Desvio padrão do ruído de entrada
    np_noise = np.random.randn(Npop, N) * sigma  # Ruído de entrada para neurônios excitatórios
    nf_noise = np.random.randn(Npop, N) * sigma  # Ruído de entrada para neurônios inibitórios

    # Definindo as equações de uma única ROI
    yp = np.zeros((Npop, N))
    xp = np.zeros((Npop, N))
    vp = np.zeros(Npop)
    zp = np.zeros((Npop, N))
    ye = np.zeros((Npop, N))
    xe = np.zeros((Npop, N))
    ve = np.zeros(Npop)
    ze = np.zeros((Npop, N))
    ys = np.zeros((Npop, N))
    xs = np.zeros((Npop, N))
    vs = np.zeros(Npop)
    zs = np.zeros((Npop, N))
    yf = np.zeros((Npop, N))
    xf = np.zeros((Npop, N))
    zf = np.zeros((Npop, N))
    vf = np.zeros(Npop)
    xl = np.zeros((Npop, N))
    yl = np.zeros((Npop, N))

    step_red = 100  # Redução de passo de 10000 para 100 Hz
    fs = f_eulero / step_red
    eeg = np.zeros((Npop, int((N - 1 - 10000) / step_red)))  # Exclusão do primeiro segundo devido a um possível transiente

    m = np.zeros(Npop)  # Valor médio do ruído de entrada

    kmax = int(np.round(np.max(D) / dt))

    for k in range(N - 1):
        up = np_noise[:, k] + m  # Entrada de contribuições exógenas para neurônios excitatórios
        uf = nf_noise[:, k]  # Entrada de contribuições exógenas para neurônios inibitórios

        if k > kmax:
            for i in range(Npop):
                up[i] += np.dot(Wp[i, :], zp[:, int(np.round(k - D[i] / dt))])
                uf[i] += np.dot(Wf[i, :], zp[:, int(np.round(k - D[i] / dt))])

        # Potenciais de membrana pós-sinápticos
        vp[:] = C[:, 1] * ye[:, k] - C[:, 3] * ys[:, k] - C[:, 6] * yf[:, k]
        ve[:] = C[:, 0] * yp[:, k]
        vs[:] = C[:, 2] * yp[:, k]
        vf[:] = C[:, 5] * yp[:, k] - C[:, 4] * ys[:, k] - C[:, 7] * yf[:, k] + yl[:, k]

        # Densidade média de spikes
        zp[:, k] = 2 * e0 / (1 + np.exp(-r * vp[:])) - e0
        ze[:, k] = 2 * e0 / (1 + np.exp(-r * ve[:])) - e0
        zs[:, k] = 2 * e0 / (1 + np.exp(-r * vs[:])) - e0
        zf[:, k] = 2 * e0 / (1 + np.exp(-r * vf[:])) - e0

        # Potencial pós-sináptico para neurônios piramidais
        xp[:, k + 1] = xp[:, k] + (G[0] * a[0] * zp[:, k] - 2 * a[0] * xp[:, k] - a[0] * a[0] * yp[:, k]) * dt
        yp[:, k + 1] = yp[:, k] + xp[:, k] * dt

        # Potencial pós-sináptico para interneurônios excitatórios
        xe[:, k + 1] = xe[:, k] + (G[0] * a[0] * (ze[:, k] + up[:] / C[:, 1]) - 2 * a[0] * xe[:, k] - a[0] * a[0] * ye[:, k]) * dt
        ye[:, k + 1] = ye[:, k] + xe[:, k] * dt

        # Potencial pós-sináptico para interneurônios inibitórios lentos
        xs[:, k + 1] = xs[:, k] + (G[1] * a[1] * zs[:, k] - 2 * a[1] * xs[:, k] - a[1] * a[1] * ys[:, k]) * dt
        ys[:, k + 1] = ys[:, k] + xs[:, k] * dt

        # Potencial pós-sináptico para interneurônios inibitórios rápidos
        xl[:, k + 1] = xl[:, k] + (G[0] * a[0] * uf[:] - 2 * a[0] * xl[:, k] - a[0] * a[0] * yl[:, k]) * dt
        yl[:, k + 1] = yl[:, k] + xl[:, k] * dt
        xf[:, k + 1] = xf[:, k] + (G[2] * a[2] * zf[:, k] - 2 * a[2] * xf[:, k] - a[2] * a[2] * yf[:, k]) * dt
        yf[:, k + 1] = yf[:, k] + xf[:, k] * dt

    # Geração de dados de 3 ROIs
    start = 10000  # Exclusão do primeiro segundo devido a um possível transiente
    eeg = np.dot(np.diag(C[:, 1]), ye[:, start::step_red]) - np.dot(np.diag(C[:, 3]), ys[:, start::step_red]) - np.dot(np.diag(C[:, 6]), yf[:, start::step_red])

    if trial == 1:
        eeg1 = eeg
    elif trial == 2:
        eeg2 = eeg
    elif trial == 3:
        eeg3 = eeg
    elif trial == 4:
        eeg4 = eeg
    elif trial == 5:
        eeg5 = eeg
    elif trial == 6:
        eeg6 = eeg
    elif trial == 7:
        eeg7 = eeg
    elif trial == 8:
        eeg8 = eeg
    elif trial == 9:
        eeg9 = eeg
    elif trial == 10:
        eeg10 = eeg

tt = t[start::step_red]  # Vetor de tempo
