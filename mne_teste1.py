import numpy as np
import mne
import matplotlib.pyplot as plot

#Create some dummy metadata
numChannels = 32
sampling_freq = 200 # in Hz
info = mne.create_info(numChannels, sfreq=sampling_freq)
# print(info)

ch_names = [f'MEG{n:03}' for n in range(1,10)] + ["EOG001"]
ch_types = ['mag','grad','grad'] * 3 + ['eog']
info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
# print(info)
# print(ch_names[8])

# no caso se o nome dos canais seguirem um dos esquemas de nomeação de montagens padrão. os locais espaciais podem ser adicionados automaticamente usando o set_montage

ch_names = ['Fp1', 'Fp2', 'Fz', 'Cz', 'Pz', 'O1', 'O2']
ch_types = ['eeg'] * 7
info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
info.set_montage('standard_1020')
info['description'] = 'My custom dataset'
info['bads'] = ['O1'] # names of bads channels
# print(info)

# Create Raw objects

times = np.linspace(0, 1, sampling_freq, endpoint=False)
sine = np.sin(20 * np.pi * times)
cosine = np.cos(10 * np.pi * times)
data = np.array([sine, cosine])

info = mne.create_info(ch_names=['10 Hz sine', '5 Hz cosine'], ch_types=['misc'] * 2, sfreq=sampling_freq)

simulated_raw = mne.io.RawArray(data, info)
simulated_raw.plot(show_scrollbars=False, show_scalebars=False)

# Creating Epochs objects
data = np.array(
    [
        [0.2 * sine, 1.0 * cosine],
        [0.4 * sine, 0.8 * cosine],
        [0.6 * sine, 0.6 * cosine],
        [0.8 * sine, 0.4 * cosine],
        [1.0 * sine, 0.2 * cosine]
    ]
)

simulated_epochs = mne.EpochsArray(data, info)
simulated_epochs.plot(picks='misc', show_scrollbars=False, events=True)
