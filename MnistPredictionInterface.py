import pickle 
import cv2
import numpy as np
import Neural_network_code.neuroNetwork as nn
import Neural_network_code.Optimizers as op

#NeuroNetwork and optimmizer must exist or else the pickle file will not be able to understand the class

with open('MNIST_training\\SavedModel\\NeuroNetworkModel.pickle', 'rb') as file:
    model = pickle.load(file)

def convertImageData(X):
    X = cv2.imread(X, cv2.IMREAD_GRAYSCALE)
    X = cv2.resize(X, (28,28))
    X = 255 - np.array([X]) 
    #The training set is inverted so you have to invert the colors 
    
    X = (X.reshape(X.shape[0], -1).astype(np.float32)- 127.5)/127.5
    
    return X

def predict(X):
    X = convertImageData(X)
    return model.predict(X)

#imageFile = 'MNIST_training\\FashionMNISTImage\\test\\1\\0001.png'
imageFile = 'MNIST_training\\inputImage.png'
#* Enter your image here

predictedClass = predict(imageFile)

classifications = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(classifications[predictedClass[0]])
