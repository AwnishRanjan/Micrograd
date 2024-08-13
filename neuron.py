from engine import value
import random

class Neuron():
    def __init__(self, nin):
        self.weight = [value(random.uniform(-1,1)) for _ in range(nin)]
        self.bias = value(random.uniform(-1,1))

    def __call__(self,x):
        act = sum(((xi * wi) for xi , wi in zip(self.weight ,x)), self.bias)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.weight +[self.bias]
    

class Layer:
    def __init__(self , nin , non):
        self.neurons  = [Neuron(nin) for _ in range(non)]
    
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        # params = []
        # for neuron in self.neurons:
        #     ps = neuron.parameters()
        #     params.extend(ps)
        #     return ps 

class MLP:
    def __init__(self , nin , nouts):
        sz = [nin]+ nouts
        self.layers = [Layer(sz[i] , sz[i+1]) for i in range(len(nouts))] 

    def __call__(self ,x):
        for layer in self.layers:
          x = layer(x)
        return x 
    
    def parameters(self):
       return [p for layer in self.layers for p in layer.parameters()]
    
