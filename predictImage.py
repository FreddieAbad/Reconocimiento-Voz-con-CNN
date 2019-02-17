import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
###
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

modelo = 'modelo.h5'
pesos_modelo = 'pesos.h5'
###
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        cnn = load_model(modelo)
###

longitud, altura = 150, 150
cnn.load_weights(pesos_modelo)

def predict(file):
    x = load_img(file, target_size=(longitud, altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = cnn.predict(x)
    result = array[0]
    answer = np.argmax(result)
    if answer == 0:
        print("Prediccion: Alba")
    elif answer == 1:
        print("Prediccion: Aguilar")
    elif answer == 2:
        print("Prediccion: Abad")
    elif answer == 3:
        print("Prediccion: Pesantez")
    else:
        print("No se ha reconocido")
    print("-------------------- \n")
    return answer

print("Real: Abad")
predict('ab.png')

print("Real: Pesantez")
predict('pe.png')

print("Real: Aguilar")
predict('ag.png')

print("Real: Alba")
predict('al.png')
