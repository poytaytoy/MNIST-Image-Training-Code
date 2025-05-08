import numpy as np 

#SGD utilizes a momentum movement to slow down movement going backwards while using the gradient to calculate descent 
#!Actual definition indicates picking some random samples, but in this textbook it could mean anything as long as it is a batch of samples
class Optimizer_VanillaGD:    
    def __init__(self, rate = 1, decay = 0.001, momentum = 0.9):
        #! Remember to calibrate (x) the decay by increments of 1e-(x) and the momentum by 1 - 0.1*(x) 
        #! For momentum, if it is 1, then it will be 33% prediction forever due to 1's property. 
        #! Learningrate calibration is unnecessary due to decay 
        self.learningRate = rate
        self.currentLearningRate = rate
        self.steps = 0 
        self.learningDecay = decay 
        self.momentum = momentum
    
    def updateWeightsBias(self, layer):

        if self.momentum:
            if not hasattr(layer, 'momentumWeights'):
                layer.momentumWeights = np.zeros_like(layer.weights)
                layer.momentumBias = np.zeros_like(layer.bias)
        
            weightUpdates = self.momentum * layer.momentumWeights - layer.dweights * self.currentLearningRate
            biasUpdates = self.momentum * layer.momentumBias - layer.dbias * self.currentLearningRate

            layer.momentumWeights = weightUpdates
            layer.momentumBias = biasUpdates
        else:
            weightUpdates = - layer.dweights * self.currentLearningRate
            biasUpdates = - layer.dbias * self.currentLearningRate
        
        layer.weights += weightUpdates
        layer.bias += biasUpdates


    def adjustRate(self):
        if self.learningDecay:    
            self.currentLearningRate = self.learningRate/(1 + self.learningDecay * self.steps) 

    def updateStep(self):
        self.steps += 1
    #Momemtum makes sure that they don't have the ability to go back and thus bouncing back and forth


#Unlike SGD, who updates values depending on the gradient, Ada focuses on a normalized increment so that less-touched neurons would get too
#They do by using a cache to collect the sum of all the updates - the bigger it is, the slower the updates become 
class Optimizer_Ada:    
    def __init__(self, rate = 1, decay = 1e-5, epsilon = 1e-7):
        #! Remember to calibrate (x) the decay by increments of 1e-(x) and the momentum by 1 - 0.1*(x) 
        #! For momentum, if it is 1, then it will be 33% prediction forever due to 1's property. 
        #! Learningrate calibration is unnecessary due to decay 
        self.learningRate = rate
        self.currentLearningRate = rate
        self.steps = 0 
        self.learningDecay = decay 
        self.epsilon = epsilon
    
    def updateWeightsBias(self, layer):

        if not hasattr(layer, 'cacheWeight'):
            layer.cacheWeights = np.zeros_like(layer.weights)
            layer.cacheBias = np.zeros_like(layer.bias)

        layer.cacheWeights += layer.dweights ** 2
        layer.cacheBias = layer.dbias ** 2

        layer.weights += self.currentLearningRate * layer.dweights/(np.sqrt(layer.cacheWeights) + self.epsilon)
        layer.bias += self.currentLearningRate * layer.dbias/(np.sqrt(layer.cacheBias) + self.epsilon)

    def adjustRate(self):
        if self.learningDecay:    
            self.currentLearningRate = self.learningRate/(1 + self.learningDecay * self.steps) 

    def updateStep(self):
        self.steps += 1
    #Momemtum makes sure that they don't have the ability to go back and thus bouncing back and forth

#Basically Ada, but with a decay, so that the optimizer can focus on more recent gradient products
#The theory is that this will it to decelerate much quickly when it reaches the global minimum (because it will detect the less fluctuation in changes from the more recent gradients)
class Optimizer_RMSProp:    
    def __init__(self, rate = 1, decay = 1e-5, epsilon = 1e-7, rho = 0.999):
        #Be sure to calibrate the rho and decay 
        #! Remember to calibrate (x) the decay by increments of 1e-(x) and the momentum by 1 - 0.1*(x) 
        #! For momentum, if it is 1, then it will be 33% prediction forever due to 1's property. 
        #! Learningrate calibration is unnecessary due to decay 
        self.learningRate = rate
        self.currentLearningRate = rate
        self.steps = 0 
        self.learningDecay = decay 
        self.epsilon = epsilon
        self.rho = rho 
    
    def updateWeightsBias(self, layer):

        if not hasattr(layer, 'cacheWeight'):
            layer.cacheWeights = np.zeros_like(layer.weights)
            layer.cacheBias = np.zeros_like(layer.bias)
        

        layer.cacheWeights = layer.cacheWeights * self.rho + (1 - self.rho) * layer.dweights ** 2
        layer.cacheBias = layer.cacheBias * self.rho + (1 - self.rho) * layer.dbias ** 2

        layer.weights += -self.currentLearningRate * layer.dweights/(np.sqrt(layer.cacheWeights) + self.epsilon)
        layer.bias += -self.currentLearningRate * layer.dbias /(np.sqrt(layer.cacheBias) + self.epsilon)

    def adjustRate(self):
        if self.learningDecay:    
            self.currentLearningRate = self.learningRate/(1 + self.learningDecay * self.steps) 

    def updateStep(self):
        self.steps += 1
    #Momemtum makes sure that they don't have the ability to go back and thus bouncing back and forth


#*This is the most popular optimizer -- True Optimizer
#It combines momentum and RMSprop - this allow it to have the best of both worlds
#However's it is really bad at generalizing - switch to SGD (actual SGD) for these cases instead 
#Momentum and cache that decays 


class Optimizer_Adam:    
    def __init__(self, rate = 0.05, decay = 5e-7, epsilon = 1e-7, beta1 = 0.9,  beta2 = 0.999):
        #Be sure to calibrate the rho and decay 
        #! Remember to calibrate (x) the decay by increments of 1e-(x) and the momentum by 1 - 0.1*(x) 
        #! For momentum, if it is 1, then it will be 33% prediction forever due to 1's property. 
        #! Learningrate calibration is unnecessary due to decay 
        self.learningRate = rate
        self.currentLearningRate = rate
        self.steps = 0 
        self.learningDecay = decay 
        self.epsilon = epsilon
        self.beta1 = beta1 #momentum decay 
        self.beta2 = beta2 #cache decay 
    
    def updateWeightsBias(self, layer):

        if not (hasattr(layer, 'cacheWeights')):
            layer.cacheWeights, layer.momentumWeights = np.zeros_like(layer.weights), np.zeros_like(layer.weights)
            layer.cacheBias, layer.momentumBias = np.zeros_like(layer.bias), np.zeros_like(layer.bias)
            
        #Momentum
        layer.momentumWeights = self.beta1 * layer.momentumWeights + (1 - self.beta1) * layer.dweights 
        layer.momentumBias = self.beta1 * layer.momentumBias + (1 - self.beta1) * layer.dbias
        
        #? Correcitons - stored in another variable because it is meant to boost learning speed at the early stages 
        momentumWeightsCorrected = layer.momentumWeights / (1 - self.beta1 ** (self.steps + 1))
        momentumBiasCorrected = layer.momentumBias / (1 - self.beta1 ** (self.steps + 1))

        #Cache
        layer.cacheWeights = self.beta2 * layer.cacheWeights + (1 - self.beta2) * layer.dweights ** 2
        #layer.cacheBias = self.beta2 * layer.cacheBias + (1 - self.beta2) * layer.dbias ** 2
        layer.cacheBias = self.beta2 * layer.cacheBias + (1 - self.beta2) * layer.dbias**2
        
        #?Corrections 
        cacheWeightsCorrected = layer.cacheWeights / (1 - self.beta2 ** (self.steps + 1))
        cacheBiasCorrected = layer.cacheBias / (1 - self.beta2 ** (self.steps + 1))

        
        layer.weights += -self.currentLearningRate * momentumWeightsCorrected / (np.sqrt(cacheWeightsCorrected) + self.epsilon)
        layer.bias += - self.currentLearningRate * momentumBiasCorrected / (np.sqrt(cacheBiasCorrected) + self.epsilon)
    
    def adjustRate(self):
        if self.learningDecay:    
            self.currentLearningRate = self.learningRate/(1 + self.learningDecay * self.steps) 
    
    def updateStep(self):
        self.steps += 1
    #Momemtum makes sure that they don't have the ability to go back and thus bouncing back and forth
    


