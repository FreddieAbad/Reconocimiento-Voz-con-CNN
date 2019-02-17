import pyaudio
import wave
import sys
#MFCC
import librosa.display
import librosa.feature
import matplotlib.pyplot as plt
import numpy as np
#PREDICT
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
#GUI
import tkinter as Tk
top = Tk.Tk()
#Grabacion Audio
def recorderV(segundosT=1):
    chunk = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = segundosT
    RATE = 44100
    RECORD_SECONDS = 1  #5
    WAVE_OUTPUT_FILENAME = "vozTemporal.wav"
    p = pyaudio.PyAudio()
    stream = p.open(format = FORMAT,
                    channels = CHANNELS,
                    rate = RATE,
                    input = True,
                    frames_per_buffer = chunk)

    print ("* GRABANDO")
    all = []
    for i in range(0, int(RATE/chunk*RECORD_SECONDS)):
        data = stream.read(chunk)
        all.append(data)
    print ("* LISTO")

    stream.close()
    p.terminate()

    # write data to WAVE file
    data = b''.join(all)
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()
    predict('tmp/vozTemporal.png')
def extract_mfcc(file, fmax, nMel):
    y, sr = librosa.load(file)
    plt.figure(figsize=(3, 3), dpi=100)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=nMel, fmax=fmax)
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), fmax=fmax)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('tmp/vozTemporal.png', bbox_inches='tight', pad_inches=-0.1)
    plt.close()
    return
modelo = 'modelo1seg.h5'
pesos_modelo = 'pesos1seg.h5'
###
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        cnn = load_model(modelo)
###

longitud, altura = 150, 150
cnn.load_weights(pesos_modelo)
def predict(file):
    file=""
    x = load_img(file, target_size=(longitud, altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = cnn.predict(x)
    result = array[0]
    answer = np.argmax(result)
    resultadoPredecir=""
    if answer == 0:
        print("Prediccion: Alba")
        resultadoPredecir="Bryan Alba"
    elif answer == 1:
        print("Prediccion: Aguilar")
        resultadoPredecir="Bryan Aguilar"
    elif answer == 2:
        print("Prediccion: Abad")
        resultadoPredecir="Freddy Abad"
    elif answer == 3:
        print("Prediccion: Pesantez")
        resultadoPredecir="Mauricio Pesantez"
    else:
        print("No se ha reconocido")
        resultadoPredecir="No se pudo reconocer"
    #res = np.argmax(str(np.argmax(array)))
    s1.set('Prediccion Hablante: '+ resultadoPredecir)

    print("-------------------- \n")
"""
recorderV(1)
extract_mfcc("vozTemporal.wav",8000, 256)
print("Real: Abad")
predict('tmp/vozTemporal.png')
"""


s1 = Tk.StringVar()
s1.set('Prediccion Hablante: ')
L0 = Tk.Label(top, text = 'Reconocimiento de Voz (CNN)', font = (None, 30))
B1 = Tk.Button(top, text = 'Grabar', command = recorderV, font = (None, 20))
L1 = Tk.Label(top, textvariable = s1, font = (None, 25))
B3 = Tk.Button(top, text = 'Quit', command = quit, font = (None, 20))  

L0.pack()
B1.pack()
B3.pack()
L1.pack(fill = Tk.X)

top.mainloop()
