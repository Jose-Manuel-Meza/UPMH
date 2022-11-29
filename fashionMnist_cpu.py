# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 11:15:39 2022

@author: Jose Manuel Meza Gonzalez
"""
import multiprocessing
import time
import matplotlib.pyplot as plt   #Libreria para gráficar
import numpy as np                #Libreria para calculos con arrelos

#Componentes de keras con tensorflow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import optimizers, backend
import tensorflow.keras.utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.datasets import fashion_mnist #Descargar la base de datos

#--------------------Descarga de DataSet---------------------------------------

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#Verificamos el tamaño de cada porcion de datos---------------

#print('El numero de imágenes en el conjunto x_train es:', x_train.shape[0])
#print('El numero de etiquetas en el conjunto y_train es:', y_train.shape[0])
#print('El numero de imágenes en el conjunto x_test es:', x_test.shape[0])
#print('El numero de etiquetas en el conjunto y_test es:', y_test.shape[0])

#El numero de imágenes en el conjunto x_train es: 60000
#El numero de etiquetas en el conjunto y_train es: 60000
#El numero de imágenes en el conjunto x_test es: 10000
#El numero de etiquetas en el conjunto y_test es: 10000

#---------------------Preprocesamiento de Datos--------------------------------
# Aplanamiento
# x_train.reshape(cantidad_de_imágenes, 784)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# Escalamiento
# Escala los valores de los pixeles al rango 0 -> 1
x_train = x_train/255
x_test = x_test/255

# Codificación 1-hot
# Aplica codificación 1-hot a las etiquetas y almacena las nuevas etiquetas en variables alternativas y_train1 y y_test1
y_train1 = np_utils.to_categorical(y_train)
y_test1 = np_utils.to_categorical(y_test)



#--------------------Creacion del Modelo---------------------------------------

# Escribe la instrucción que elimina los modelos creados con keras 
# Recuerda que es necesario eliminar el modelo cada que se desee volver a entrenar
backend.clear_session()

# Define aqui tu modelo convolucional colocando las capas convolucionales y pooling, la capa flatten y las capas dense. 
# Tip 1: comienza con modelos pequeños y luego aumentas su tamaño. 
# RECUERDA QUE ÚNICAMENTE LA PRIMER CAPA DEBE LLEVAR EL PARÁMETRO: input_shape = (alto, ancho, número_de_canales)
# RECUERDA QUE LA ULTIMA CAPA REQUIERE UNA NEURONA POR CLASE. 

modeloConv = Sequential()
modeloConv.add(Conv2D(64, kernel_size=(2,2), activation='relu',
                      input_shape = (28,28,1)))
modeloConv.add(MaxPooling2D(pool_size=(2,2)))
modeloConv.add(Conv2D(128, kernel_size=(2,2), activation='relu'))
modeloConv.add(MaxPooling2D(pool_size=(2,2)))
modeloConv.add(Flatten())
modeloConv.add(Dense(30, activation='relu'))
modeloConv.add(Dense(20, activation='relu'))
modeloConv.add(Dense(10, activation='softmax'))


# Fin de la definición del modelo

# La siguiente línea de código es para mostrar la tabla con las características del modelo
modeloConv.summary()

#modelo_1.get_weights()
# Define el optimizador Adam
adam = optimizers.Adam(learning_rate = 0.001)

# Compila el modelo definiendo el optimizador "adam", la función de costo (loss) como la "categorical_crossentropy"
# y la metrica como el "accuracy"
modeloConv.compile(optimizer=adam, 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])


#-------Implementacion de entrenamiento ---------------------------------------
#vars_entrenables0 = modelo_1.trainable_variables[-3]
#print(vars_entrenables0.shape)
# Entrena el modelo eligiendo un número de epocas y los datos de entrenamiento y validación
# Recuerda que guardamos el entrenamiento en una variable para poder graficar el error
# RECUERDA ENTRENAR CON LOS "Y" CODIFICADOS CON 1-HOT

inicio_time = time.perf_counter()

m = modeloConv.fit(x_train, y_train1, 
                 validation_data = (x_test, y_test1),
                 epochs = 20, batch_size = 70,
                 verbose = True)

fin_time = time.perf_counter()

print('Tiempo de ejecución = ',fin_time-inicio_time)

#----------------------División del dataset-----------------------------------

x_train_0 = x_train[:20000]
x_train_1 = x_train[20001:40000]
x_train_2 = x_train[40000:]

y_train1_0 = y_train1[:20000]
y_train1_1 = y_train1[20001:40000]
y_train1_2 = y_train1[40000:]

x_test_0 = x_test[:3333]
x_test_1 = x_test[3334:6666]
x_test_2 = x_test[6667:]

y_test1_0 = y_test1[:3333]
y_test1_1 = y_test1[3334:6666]
y_test1_2 = y_test1[6667:]

#-------------------Creación copia 0 del modelo------------------------------

backend.clear_session()

modeloConv0  = keras.models.clone_model(modeloConv)

modeloConv0.summary()

modeloConv0.compile(optimizer=adam, 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])



inicio_time = time.perf_counter()

m0 = modeloConv0.fit(x_train_0, y_train1_0, 
                 validation_data = (x_test_0, y_test1_0),
                 epochs = 20, batch_size = 70,
                 verbose = True)

fin_time = time.perf_counter()

print('Tiempo de ejecución = ',fin_time-inicio_time)

#-------------------Creación copia 1 del modelo------------------------------

backend.clear_session()

modeloConv1  = keras.models.clone_model(modeloConv)

modeloConv1.summary()

modeloConv1.compile(optimizer=adam, 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])



inicio_time = time.perf_counter()

m0 = modeloConv1.fit(x_train_1, y_train1_1, 
                 validation_data = (x_test_1, y_test1_1),
                 epochs = 20, batch_size = 70,
                 verbose = True)

fin_time = time.perf_counter()

print('Tiempo de ejecución = ',fin_time-inicio_time)

#-------------------Creación copia 2 del modelo------------------------------

backend.clear_session()

modeloConv2  = keras.models.clone_model(modeloConv)

modeloConv2.summary()

modeloConv2.compile(optimizer=adam, 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])



inicio_time = time.perf_counter()

m0 = modeloConv2.fit(x_train_2, y_train1_2, 
                 validation_data = (x_test_2, y_test1_2),
                 epochs = 20, batch_size = 70,
                 verbose = True)

fin_time = time.perf_counter()

print('Tiempo de ejecución = ',fin_time-inicio_time)