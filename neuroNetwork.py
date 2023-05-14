
import numpy as np 
import nnfs
from nnfs.datasets import spiral_data
import Optimizers as op

nnfs.init()

class activateRELU:
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0,inputs)
    def backwards(self, gradients):
        self.dinputs = gradients.copy()
        self.dinputs[self.inputs <= 0] = 0

        #apply the activateRelU function to the derivatives - because values that
        #went 0 should not have an impact after this point 

class softMaxFunction:
    def forward(self, inputs):
        newInputs = np.exp(inputs - inputs.max(axis = 1, keepdims = True))
        #The subtrction exist to prevent an overflow
        self.outputs = newInputs/np.sum(newInputs, axis = 1, keepdims = True) 
    def backwards(self, dvalues):
        for index, (outputs, dvalue) in enumerate(self.outputs, dvalues):
            jacobMatrix = np.diagflat(outputs.reshape(-1,1)) - np.dot(outputs, outputs.T)
            self.dinputs[index] = np.dot(jacobMatrix, dvalue)

#For binary regression
class sigmoidFunction:
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = 1/(1 + np.exp(-self.inputs))

    def backwards(self, dvalues):
        self.dinputs = dvalues * self.outputs * (1 - self.outputs)

class FunctionLoss:
    def calculate(self, outputs, targetedClass):
        sampleLoss = self.forward(outputs, targetedClass)
        return np.mean(sampleLoss)

    def regulizationForward(self, layer):
        lossRegulizationloss = 0 

        #L1 Regulization
        lossRegulizationloss += layer.weightsRegulizationL1 * np.sum(np.abs(layer.weights))
        lossRegulizationloss += layer.biasRegulizationL1 * np.sum(np.abs(layer.bias))
        
        #L2 Regulization
        lossRegulizationloss += layer.weightsRegulizationL2 * np.sum(np.abs(layer.weights**2))
        lossRegulizationloss += layer.biasRegulizationL2 * np.sum(np.abs(layer.bias**2))

        return lossRegulizationloss

class LossCategoricalCrossEntropy(FunctionLoss):
    def forward(self, outputs, targetedClass):
        #There are two types - either they return a single dimension list specifying which index they are 
        #targetting for each dimension or a multi-dimensional list specificying the value they want 
        #IF MULTIDIMENSIONAL IT MUST BE AN NPARRAY
        '''
        ex 1: [1,2,0]
        ex 2, [[0,1,0], [0,0,1], [1,0,0]]
        '''
        try: 
            targetedClass.shape
        except AttributeError: 
            targetedClass = np.array(targetedClass)
        
        if len(targetedClass.shape) == 1: #check to see if the list is single 
            selectedResults = outputs[range(len(outputs)), targetedClass]
        
        if len(targetedClass.shape) == 2: #check to see if the list is multi-dimensional 
            selectedResults = np.sum(outputs * targetedClass, axis = 1, keepdims = True)
        
        selectedResults =np.clip(selectedResults, 1e-7, 1-1e-7)
        
        return -np.log(selectedResults)
    
    def backwards(self, outputs, realY):
        dimensions = outputs.shape
        if len(dimensions) == 1:
            realY = np.eye(dimensions[1])[realY]
        self.dinputs = -realY / outputs
        self.dinputs = self.dinputs/dimensions[0]

class LossBinaryCrossEntropy(FunctionLoss):
        def forward(self, inputs, realY):
            clippedY =  np.clip(inputs, 1e-7, 1-1e-7)

            perSampleLoss = -(realY * np.log(clippedY)) + (1 - realY) * np.log(1 - clippedY)
            #? Left side calculates how wrong the variable is from being 1 and the right side calculate
            #? how wrong it is from being 0 

            return np.mean(perSampleLoss, axis = -1)
        
        def backwards(self, dvalues, realY):
            clippedDvalues = np.clip(dvalues, 1e-7, 1-1e-7)
            dimensions = dvalues.shape
            self.dinputs  = -((realY/clippedDvalues) - (1-realY)/(1-clippedDvalues))/dimensions[1]/dimensions[0]

class CategoricalCrossEntropy:
    def __init__(self):
        self.softmax = softMaxFunction()
        self.loss = LossBinaryCrossEntropy()

    def forward(self, inputs, y_true):
        self.softmax.forward(inputs)
        self.outputs = self.softmax.outputs
        self.loss.forward(self.outputs, y_true)
        
        return self.loss.outputs

    def backwards(self, dvalues, y_true):
        dimensions = y_true.shape

        if len(dimensions) == 2:
            y_true = np.argmax(y_true, axis = 1)
        
        self.dinputs = dvalues.copy()
        self.dinputs[range(dimensions[0]), y_true] -= 1
        #Derivative is equal to subtract the desired outputs by 1
        self.dinputs = self.dinputs/dimensions[0]
        #Normalize it

class dropoutLayer:
    def __init__(self, rate):
        self.rate = 1 - rate 
    
    def forward(self, inputs):
        self.inputs = inputs
        self.binomial = np.random.binomial(1, self.rate, size = inputs.shape)
        self.outputs = self.inputs * self.binomial
        
    def backwards(self, dvalues):
        self.dinputs = dvalues * self.binomial
   
class neuroNetworkLayer:
    def __init__(self, itemsInArray, Dimensions, weightsRegulizationL1 = 0, biasRegulizationL1 = 0, weightsRegulizationL2 = 0, biasRegulizationL2 = 0):
        self.weights = 0.01 * np.random.randn(itemsInArray, Dimensions)
        #Weights must be as close as possible to zero 
        #! self.weights = np.zeros((itemsInArray, Dimensions))
        self.bias = np.zeros((1,Dimensions))
        #! bias MUST START WITH ZEROS
        self.weightsRegulizationL1 = weightsRegulizationL1
        self.biasRegulizationL1 = biasRegulizationL1
        self.weightsRegulizationL2 = weightsRegulizationL2
        self.biasRegulizationL2 = biasRegulizationL2

        #! Only apply regulization at the second Layer
    
    def forward(self,inputs):
        self.input = inputs 
        self.outputs = np.dot(self.input, self.weights) + self.bias #softMaxFunction(pushForward(np.dot(self.input,self.weights)) + self.bias)
       # print(self.outputs)

    def backwards(self, dvalues):
        self.dweights = np.dot(self.input.T, dvalues)

        self.dbias = np.sum(dvalues, axis = 0, keepdims = True)
        

        #Backward pass of the regulization
        if self.weightsRegulizationL1 > 0:
            dL1weights = np.ones_like(self.weights)
            dL1weights[self.weights < 0] = -1
            
            self.dweights += self.weightsRegulizationL1 * dL1weights
        
        if self.biasRegulizationL1 > 0:
            dL1weights = np.ones_like(self.bias)
            dL1weights[self.bias < 0] = -1

            self.dbias += self.biasRegulizationL1 * dL1weights
        
        if self.weightsRegulizationL2 > 0:
            self.dweights += self.weightsRegulizationL2 * 2 * self.weights
        
        if self.biasRegulizationL2 > 0:
            self.dbias += self.biasRegulizationL2 * 2 * self.bias

        self.dinputs = np.dot(dvalues, self.weights.T)


X,y = spiral_data(samples = 100, classes = 2)
Xtest, Ytest = spiral_data(samples = 100, classes = 2)

y = y.reshape(-1, 1)
Ytest = Ytest.reshape(-1,1)

Optimize = op.Optimizer_Adam(rate = 0.02, decay = 5e-7)
#Optimize = op.Optimizer_Adam(rate = 0.02)

#Layer1 = neuroNetworkLayer(2,256)
Layer1 = neuroNetworkLayer(2, 64, weightsRegulizationL2 = 5e-4, biasRegulizationL2=5e-4)
#! Only apply the regulization on the hidden layer - basically every layer up to the outputs layer
#! which in our case is Layer2
dropout = dropoutLayer(0.1)
Layer2 = neuroNetworkLayer(64,1)
Activation = activateRELU()
#loss = CategoricalCrossEntropy()  
Activation2 = sigmoidFunction()
loss = LossBinaryCrossEntropy()

bestAccuracy = 0 

for epoch in range(10001):

    Layer1.forward(X)   
    Activation.forward(Layer1.outputs)
    Layer2.forward(Activation.outputs)
    Activation2.forward(Layer2.outputs)
    
    NormalLoss = loss.calculate(Layer2.outputs, y)
    RegulizationLoss = loss.regulizationForward(Layer1) + loss.regulizationForward(Layer2) 
    ActualLoss = NormalLoss + RegulizationLoss

    #predictions = np.argmax(loss.outputs, axis = 1)
    #predictions2 = np.argmax(y, axis = 1) if len(y.shape) == 2 else y

    predictions = (Activation2.outputs > 0.5) * 1
    accuracy = np.mean(predictions == y)

    
    if epoch % 100 == 0:
        print('e1', epoch)
        print('p', accuracy)
        print('l', ActualLoss)
        print('nl', NormalLoss)
        print('rl', RegulizationLoss)
    
    loss.backwards(Activation2.outputs, y)
    Activation2.backwards(loss.dinputs)
    Layer2.backwards(Activation2.dinputs)
    Activation.backwards(Layer2.dinputs)
    Layer1.backwards(Activation.dinputs)
    
    Optimize.adjustRate()
    Optimize.updateWeightsBias(Layer1)
    Optimize.updateWeightsBias(Layer2)
    Optimize.updateStep()


print('')
print('test run')

for epoch1 in range(1):
    Layer1.forward(Xtest)   
    Activation.forward(Layer1.outputs)
    Layer2.forward(Activation.outputs)
    
    NormalLoss = loss.calculate(Layer2.outputs, Ytest)
    RegulizationLoss = loss.regulizationForward(Layer1) + loss.regulizationForward(Layer2) 
    ActualLoss = NormalLoss + RegulizationLoss

    
    predictions = (Activation2.outputs > 0.5) * 1
    accuracy = np.mean(predictions == Ytest)
    #predictions = np.argmax(Activation2.outputs, axis = 1)
    #predictions2 = np.argmax(Ytest, axis = 1) if len(Ytest.shape) == 2 else Ytest
    #accuracy = np.mean(predictions == predictions2)

    if not epoch % 100:
        print('e', epoch1)
        print('p', accuracy)
        print('l', ActualLoss)
        print('nl', NormalLoss)
        print('rl', RegulizationLoss)

    loss.backwards(Activation2.outputs, Ytest)
    Activation2.backwards(loss.dinputs)
    Layer2.backwards(Activation2.dinputs)
    Activation.backwards(Layer2.dinputs)
    Layer1.backwards(Activation.dinputs)


print(f'accuracy: {accuracy} loss: {ActualLoss} epoch: {epoch}')







   # y * -log(predictedy) - (1-y) * -log(1 - predictedy)     

# (realY / np.log(Y)) - (1-realY)/np.log(1-Y) / dimensionp[1] / dimension[0]

