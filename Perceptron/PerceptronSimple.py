'''
Creado inicialmente el 20-09-2019

@author: Isaac Silva Luna
'''
import numpy as np


class PerceptronSimple(object):
    def __init__(self, pesos, bias, umbral=0.5):
        self.pesos = pesos
        self.bias = bias
        self.umbral = umbral

    def activacion(self, x):
        # Activación tipo Hardlim
        return 1 if (np.dot(self.pesos, x) + self.bias >= 0) else 0

    def prediccion(self, X):
        Y = []
        for x in X:
            resultado = self.activacion(x)
            Y.append(resultado)
        return np.array(Y)

    def ajuste(self, X, Y, ciclos=10, tasa_aprendizaje=0.1):

        tasas_error = {}

        for i in range(ciclos):
            print("Ciclo #" + str(i + 1) + " de " + str(ciclos))
            error_num = 0
            for x, y in zip(X, Y):
                y_prediccion = self.activacion(x)
                error = y - y_prediccion
                if error != 0:
                    error_num += 1
                    self.pesos += tasa_aprendizaje * error * x
                    self.bias += error
            tasas_error[i] = error_num / len(Y)
        return tasas_error
