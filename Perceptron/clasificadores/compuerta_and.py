'''
Creado inicialmente el 20-09-2019

@author: Isaac Silva Luna
'''
import random

from sklearn.model_selection import train_test_split

import PerceptronSimple
import matplotlib.pyplot as grafica
import numpy as np
import pandas as pd
import sklearn.metrics as metricas


if __name__ == '__main__':
    # Lectura de datos
    datos = pd.read_csv("../and.csv")
    X = datos.drop('salida_and', axis=1)
    Y = datos['salida_and']
    # Separación datos de entrenamiento y de prueba
    X_entrenamiento, X_prueba, Y_entrenamiento, Y_prueba = train_test_split(
        X, Y, test_size=0.2, random_state=0)
    # Entrenamiento. Ajuste de pesos, según ciclos y tasa de aprendizaje
    X_entrenamiento = X_entrenamiento.values
    X_prueba = X_prueba.values
    perceptron = PerceptronSimple.PerceptronSimple(
        np.random.rand(X.shape[1]), random.random())  # Pesos iniciales y Bias
    tasas_error = perceptron.ajuste(X_entrenamiento, Y_entrenamiento, 10, 0.1)
    grafica.plot(list(tasas_error.values()))
    grafica.xlabel("# Ciclo")
    grafica.ylabel("Tasa Error")
    grafica.show()
    # Predicción con pesos ajustados sobre datos de prueba
    Y_pred_prueba = perceptron.prediccion(X_prueba)
    # Nivel de error del aprendizaje
    print("Nivel de error del aprendizaje: " +
          str(1.0 - metricas.accuracy_score(Y_pred_prueba, Y_prueba)) + " %")
