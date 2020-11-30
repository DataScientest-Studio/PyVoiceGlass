import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
import serial, time
import serial.tools.list_ports
from time import sleep
import tensorflow as tf
import speechnet as spnet   
import signalanalysis as sg

# audio
import librosa
import librosa.display
import pyaudio
import wave

SERIAL_PORT = '/dev/cu.usbmodem142201'
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

g_words = {
    'right':    0,                                             
    'five':     1,
    'zero':     2,
    'cat':      3,
    'yes':      4,
    'six':      5,
    'down':     6,
    'house':    7,
    'sheila':   8,
    'three':    9,
    'off':     10,
    'left':    11,
    'bed':     12,
    'happy':   13,
    'eight':   14,
    'bird':    15,
    'nine':    16,
    'tree':    17,
    'one':     18,
    'no':      19,
    'go':      20,
    'on':      21,
    'stop':    22,
    'seven':   23,
    'dog':     24,
    'four':    25,
    'wow':     26,
    'up':      27,
    'two':     28,
    'marvin':  29   
}

cv_words = {
    'neuf':    0,                                             
    'Hey':     1,
    'oui':     2,
    'Firefox': 3,
    'trois':   4,
    'sept':    5,
    'zéro':    6,
    'non':     7,
    'six':     8,
    'huit':    9,
    'quatre': 10,
    'cinq':   11,
    'un':     12,
    'deux':   13
}


# ######################################################################################################################
# General function
# ######################################################################################################################
def displaylabel(prediction, words):
    listOfKeys = [key  for (key, value) in words.items() if value == prediction]
    return listOfKeys[0]    
    

def sendtext_arduino(predicted_text):
    ard = serial.Serial(SERIAL_PORT, 9600)
    time.sleep(2) #give the connection a second to settle
    txt = predicted_text.strip()
    counter = 32 # Below 32 everything in ASCII is gibberish
    while True:
        counter +=1
        time.sleep(.5) # Delay for one tenth of a second
        if counter == 255:
            break;      
        ard.write(txt.encode())
        
def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))

@st.cache
def load_data(dataset, delimiter = None):
    #data = pd.read_csv(DATA_URL, nrows=nrows)
    #lowercase = lambda x: str(x).lower()
    #data.rename(lowercase, axis='columns', inplace=True)
    #data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    #return data
    if delimiter == None:
        delimiter = ','
    data = pd.read_csv(dataset, sep= delimiter)
    return data

    
# ######################################################################################################################
# Audio functions 
# ######################################################################################################################
@st.cache
def load_audio(audio_path):
    return librosa.load(audio_path, sr = SAMPLE_RATE)

   
def load_audio_default(audio_path):
    return librosa.load(audio_path, sr = None)


@st.cache    
def record_audio(duration):
    filename = "recorded.wav"
    chunk = 1024
    FORMAT = pyaudio.paInt16
    channels = 1
    sample_rate = 44100
    record_seconds = duration
    p = pyaudio.PyAudio()
    # open stream object as input & output
    stream = p.open(format=FORMAT,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)
    frames = []
    print("Recording...")
    for i in range(int(44100 / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)
    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(sample_rate)
    wf.writeframes(b"".join(frames))
    wf.close()                                                            
    audio="recorded.wav"
    return audio
     
    
# ######################################################################################################################
# Loading models
# ######################################################################################################################    
    
def loadModel_CV():    
    model = spnet.SpeechNet(params)
    number_exploited_data = params['max_audio_length']*params['sampling_rate']-params['n_fft']
    lengths = int(number_exploited_data/params['frame_step']+1)
    model(np.random.uniform(size=[1, lengths, params['num_mel_bins']]))
    model.load_weights('model/cv/cv_multiple.h5')
    return model

def loadModel_CV_SW():    
    # create the CNN model using Conv 1D
    model = tf.keras.models.load_model('model/cv-sw/cv_singleword.h5')
    model.load_weights('model/cv-sw/cv_singleword_weights')
    return model


def loadModel_G_SW():
    # create the CNN model using Conv 1D
    model = tf.keras.models.load_model('model/g-sw/g_singleword.h5')
    model.load_weights('model/g-sw/g_singleword_weights')
    return model


def predict_singleword_cv(audio_path):
    # text = "texte prédit sur le modèle simple entrainé sur le dataset Common Voice Single Word"
    text = audio_path
    num_classes = 14
    dt=0.02
    T_max = 5
    # modèle simple
    model = loadModel_CV_SW()
    # Prédiction
    a_path = []
    a_path.append(audio_path)    
    a_data = sg.load_rawdata(np.array(a_path))
    a_data = np.array(sg.processRawData(a_data, T_max))
    a_mel = sg.convert2logMelSpectrogram(a_data)
    spectre_audio = a_mel.reshape(1,251, 161)
    prediction = model.predict(spectre_audio).argmax()
    text = displaylabel(prediction, cv_words)
    return text

def predict_singleword_g(audio_path):
    # text = "texte prédit sur le modèle simple entrainé sur le dataset Google Single Word"
    text = audio_path
    num_classes = 30
    dt=0.02
    T_max = 2
    # modèle simple
    model = loadModel_G_SW()
    # Prédiction
    dataT, feT = load_audio(audio_path)  
    if len(dataT)>= T_max*feT:
        dataT = dataT[:int(T_max*feT)]
    else :
        dataT = np.concatenate([dataT, np.zeros(int(T_max*feT - len(dataT)))])
    spectre_audio = sg.logMelSpectrogram_sw(dataT, feT, dt)
    spectre_audio = spectre_audio.reshape(1,101,161)
    #
    prediction = model.predict(spectre_audio).argmax()
    text = displaylabel(prediction, g_words)
    #
    return text  
    

def predict_multiplewords_cv(audio_path):
    text = ""
    T_max = params['max_audio_length']
    # modèle avancée 
    audio_data, audio_stft = sg.getAudio_stft(audio_path)
    audio_data = sg.processRawData(audio_data, T_max)
    # Feature
    X_audio = np.array(audio_data)
    # Mise en forme des fichiers audios X_audio sous la forme d'un tableau array de log mel spectrogramme dans la variable X.
    fe = 16000
    X = np.array([sg.logMelSpectrogram(audio, params, fe) for audio in X_audio])
    # Matrice de probabilité
    model = loadModel_CV()
    y_logit = model(X)
    text = spnet.greedy_decoder(y_logit, params)
    return text

@st.cache
def predict(model_index, audio_path):
    # modèle simple CV
    if model_index == 0:
        text = predict_singleword_cv(audio_path)
    # modèle simple Google
    elif model_index == 1:
        text = predict_singleword_g(audio_path)     # OK
    # modèle avancée 
    elif model_index == 2:
        text = predict_multiplewords_cv(audio_path) # OK
    
    return text