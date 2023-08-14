from micronets.autograd_engine import Value
from micronets.graph import draw
import matplotlib.pyplot as plt
import random

class Neuron:
    
    def __init__(self, nin, activation):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
        self.activation = activation
        
    def __call__(self, x):
        # w * x + b
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        out = self.activation(act)
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
    
class Layer:
    
    def __init__(self, nin, nout, activation):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]
        
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs
    
    def parameters(self):
        params=[]
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params
    
    
""" Dense Neural Network """

class DNN:  
    
    def __init__(self, input_features, layers, activations):  # nin is the number of inputs in each neuron and nouts is a list of neurons we want in each layer
        sz = [input_features] + layers
        activation_funcs = [ACTIVATION_FUNCTIONS[name] for name in activations]
        self.layers = [Layer(sz[i], sz[i+1], activation_funcs[i]) for i in range(len(layers))]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        params=[]
        for layer in self.layers:
            ps = layer.parameters()
            params.extend(ps)
        return params
    
    def train(self, X, Y, iterations, loss_function, learning_rate=0.01):
        history = []
        
        for i in range(iterations):
            
            #forward pass
            ypred = [self(x) for x in X]
            error = loss_function(Y, ypred)

            #backward pass
            for p in self.parameters(): 
                p.grad = 0.0
            error.backward()

            #gradient descent
            for p in self.parameters():
                p.data += -learning_rate * p.grad
                
            history.append([i+1, error.data])
        
        graph = draw(error)
            
        return history, graph
    
    def predict(self, X):
        ypred = [self(x).data for x in X]
        return ypred
    
    def plot(self, history):
        plt.xlabel("Iterations")
        plt.ylabel("Cost Function")
        plt.title("Learning Curve")
        plt.plot([row[0] for row in history], [row[1] for row in history], color='red', label='loss')
        plt.legend()
        
        return None

     
ACTIVATION_FUNCTIONS = {
        'tanh': Value.tanh,
        'relu': Value.relu,
        'sigmoid': Value.sigmoid,
        'linear': Value.linear
}