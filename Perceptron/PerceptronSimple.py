'''
Creado inicialmente el 20-09-2019

@author: Isaac Silva Luna
'''
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics.accuracy_score as accuracy_score

class PerceptronSimple(object):
    def __init__ (self):
        self.w = None
        self.b = None
    
    def modelo(self, x):
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

        accuracy = {}
        max_accuracy = 0

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

            accuracy[i] = accuracy_score(self.predict(X), Y)
            if (accuracy[i] > max_accuracy):
                max_accuracy = accuracy[i]
                j = i
                chkptw = self.w
                chkptb = self.b

        self.w = chkptw
        self.b = chkptb

        print(max_accuracy,j)

        plt.plot(accuracy.values())
        plt.xlabel("Ciclo #")
        plt.ylabel("Precision")
        plt.ylim([0, 1])
        plt.show()

        return np.array(wt_matrix)        