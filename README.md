My code on making my first neural network from scratch using numpy that is trained with the MNIST image dataset.

The prediction interface for the MNIST dataset is in the MnistPrediction.py in the MNIST_training folder. 

Here's the neural network layers configured: 

Input:
- Greyscale and compress image to 28 by 28 pixel with CV2

Layers:
- Layer 1: Dense (28 -> 128), Activation: ReLU
- Layer 2: Dense (128 -> 128), Activation: ReLU
- Output Layer: Dense (128 -> 10), Activation: Softmax

Output: 
- Classified as one of: 'T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'

Training:
- Loss Function: Categorical Cross-Entropy
- Optimizer: Adam (Learning Rate: 0.001, Decay: 1e-3)

Basically, 

If you input an image like this: 

<img width="523" height="457" alt="image" src="https://github.com/user-attachments/assets/e1901f32-b8c5-484a-925c-93879090c161" />

It'll output that it is a T-shirt/Top :D
