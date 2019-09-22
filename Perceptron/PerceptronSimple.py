'''
Creado inicialmente el 20-09-2019

@author: Isaac Silva Luna
'''
import numpy as np
import sklearn.metrics.accuracy_score as accuracy_score

class PerceptronSimple(object):
    def __init__ (self):
        self.w = None
        self.b = None
    
    def modelo(self, x):
        #Formula base del Perceptron (LaTeX):
        #y=\left\{\begin{array}{lr}\text{1, si }\sum_{n}^{i=0}w_{i}x_{i}\geq0\\
        #\text{0, caso contrario}\\\end{array}\right.
        return 1 if (np.dot(self.w, x) >= self.b) else 0
    
    def prediccion(self, X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
            return np.array(Y)
    
    def ajuste(self, X, Y, epochs = 1, lr = 1):
    
        self.w = np.ones(X.shape[1])
        self.b = 0

        precision = {}
        maxima_precision = 0

        wt_matrix = []

        for i in range(epochs):
            for x, y in zip(X, Y):
                y_pred = self.model(x)
                if y == 1 and y_pred == 0:
                    self.w = self.w + lr * x
                    self.b = self.b - lr * 1
                elif y == 0 and y_pred == 1:
                    self.w = self.w - lr * x
                    self.b = self.b + lr * 1

            wt_matrix.append(self.w)    

            precision[i] = accuracy_score(self.predict(X), Y)
            if (precision[i] > maxima_precision):
                maxima_precision = precision[i]
                chkptw = self.w
                chkptb = self.b

        self.w = chkptw
        self.b = chkptb

        return np.array(wt_matrix)