import numpy as np
from pathlib import Path

# audio
import librosa
import librosa.display
import pyaudio
import wave

SAMPLE_RATE = 16000        #  couvre la fréquence de la voix humaine
DT = 0.02

params = {
    'max_audio_length': 10,    # T_MAX : Durée max d'un fichie audio
    'max_audio_length_CV_SW': 5,
    'max_audio_length_G_SW': 2,
    'alphabet': ' !"&\',-.01234:;\\abcdefghijklmnopqrstuvwxyz',
    'causal_convolutions': False,
    'stack_dilation_rates': [1, 3, 9, 27],
    'stacks': 6,
    'stack_kernel_size': 7,
    'stack_filters': 3*128,
    'sampling_rate': SAMPLE_RATE,
    #
    'n_fft': 160*8,
    'frame_step': 160*4,
    'lower_edge_hertz': 0,
    'upper_edge_hertz': 8000,
    'num_mel_bins': 160
}


# ######################################################################################################################
# Signal Analysis functions
# ######################################################################################################################
# @st.cache
def plot_audio(audio_data, fe):
    plt.plot(1/fe*np.arange(len(audio_data)),audio_data)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

def logMelSpectrogram_sw(audio, fe, dt):
    # Spectrogram
    stfts = np.abs(librosa.stft(audio,
                        n_fft = int(dt*fe),
                        hop_length = int(dt*fe),
                        center = True
                        )).T
    num_spectrogram_bins = stfts.shape[-1]
    # MEL filter
    linear_to_mel_weight_matrix = librosa.filters.mel(
                                sr=fe,
                                n_fft=int(dt*fe) + 1,
                                n_mels=num_spectrogram_bins,
                    ).T

    # Apply the filter to the spectrogram
    mel_spectrograms = np.tensordot(
                stfts,
                linear_to_mel_weight_matrix,
                1
            )
    return np.log(mel_spectrograms + 1e-6)

def logMelSpectrogram(audio, params, fe):
    stfts = librosa.stft(audio,
                        n_fft = int(params['n_fft']*fe/SAMPLE_RATE),
                        hop_length = int(params["frame_step"]*fe/SAMPLE_RATE),
                        center = False
                        ).T
    power_spectrograms = np.real(stfts * np.conj(stfts))
    num_spectrogram_bins = power_spectrograms.shape[-1]
    linear_to_mel_weight_matrix = librosa.filters.mel(
                                sr=fe,
                                n_fft=int(params['n_fft']*fe/SAMPLE_RATE) + 1,
                                n_mels=params['num_mel_bins'],
                                fmin=params['lower_edge_hertz'],
                                fmax=params['upper_edge_hertz']
                    ).T
    mel_spectrograms = np.tensordot(
                power_spectrograms,
                linear_to_mel_weight_matrix,
                1
            )
    return (np.log(mel_spectrograms + 1e-6).astype(np.float16))

def plot_logMelSpectrogram(audio, params, fe):
    sns.heatmap(np.rot90(logMelSpectrogram(audio, params, fe)), cmap='inferno', xticklabels=False, yticklabels=False)

def getAudio_stft(audio_path):
    audio_data = []  # basic infos
    audio_stft = []  # features       
    fname = Path(audio_path)
    if fname.exists() == True :
        X, sample_rate = librosa.load(audio_path, mono = True, sr = None)
        # Resampling at 16000 Hz
        SR = params['sampling_rate']
        X = librosa.resample(X, sample_rate, SR)
        # features                
        stft = np.abs(librosa.stft(X))
        audio_data.append(X)
        audio_stft.append(stft)
    return audio_data, audio_stft

def processRawData(raw_data, max_duration):
    T_max = max_duration
    fe = SAMPLE_RATE
    size_ = len(raw_data)
    k = 0
    X_audio=[]
    for i in range(size_):
        k += 1
        data = raw_data[i]
        if len(data) >= T_max*fe: 
            # Shape invalid: truncate pour le coup
            data = data[:int(T_max*fe)]
        # After this transformation add zeroes to have the right shape
        else :
            data = np.concatenate([data, np.zeros(int(T_max*fe - len(data)))])
        X_audio.append(data)
    return X_audio


def load_rawdata(X_path, dt=0.02, T_max=2):
    all_data = []
    all_path =[]
    SR = params['sampling_rate']
    size_ =len(X_path)
    for index in (range(size_)):
        audio_path = X_path[index]
        # Load the audio file
        fname = Path(audio_path)
        if fname.exists() == True :
            # print(audio)
            X, sample_rate = librosa.load(audio_path, mono = True, sr = None)
            # Resampling at 16000 Hz
            X = librosa.resample(X, sample_rate, SR)
            all_data.append(X)        
    return all_data


def convert2logMelSpectrogram(X_data):
    fe = params['sampling_rate']
    X_audio=[]
    size_ =len(X_data)
    for i in range(size_):
        # Apply the logMelSpectrogram function.    
        spectre_audio = logMelSpectrogram_sw(X_data[i], fe, DT)
        X_audio.append(spectre_audio)
    return np.array(X_audio)   