import math
import numpy as np

class value:
    def __init__(self, data , children= () ,operation='',lable = ''):
        self.data = data 
        self.previous = set(children)
        self.operation = operation
        self.label = lable 
        self.grad = 0.0
        self._backward  = lambda : None 
    def __repr__(self):
        return f'value(data = {self.data}) , Grad = {self.grad})'
    
    def __add__(self, other):
        other = other if isinstance(other , value) else value(other)
        out = value(self.data+other.data , children=(self, other), operation='+')
        
        def _backward():
            self.grad +=1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward = _backward 
        return out 
    
    def __mul__(self,other):
        other = other if isinstance(other , value) else value(other)
        out = value(self.data*other.data , children=(self, other), operation='*')
        
        def _backward():
            self.grad +=other.data*out.grad
            other.grad+=self.data *out.grad
        out._backword = _backward 
        return out 
    
    def __pow__(self, other):
        assert[isinstance(other , (int , float)) , "only support int/float"]
        out = value(self.data**other, children=(self, ), operation=f'**{other}')
        def _backward():
            self.grad += other * (self.data ** (other-1)) * out.grad

        out._backward = _backward 
        return out 
    
    def tanh(self):
        x = self.data
        t = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        out = value(t , (self, ) , 'tanh')  
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out 
    
    def __exp__(self):
        x = self.data 
        out = value(math.exp(x), (self ,) , 'exp')
        def _backward():
            self.grad = out.data * out.grad 
        out._backward = _backward
        return out 
    
    ## reverse functions ::::::
    def __truediv__(self , other):
       return (self * other**-1)
    
    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rmul__(self, other):
        return self * other 
    
    def __sub__(self , other):
        return self +(-other)
    
    def __rsub__(self, other): # other - self
        return other + (-self)
    
    def __neg__(self):
        return self * -1
    
    def __hash__(self):
        return hash(id(self))  # Ensure hashability

    def __eq__(self, other):
        return id(self) == id(other)

    ## backpropogation :: 

    def backward(self):
        self.grad= 1.0
        topo = []
        visited = set()
        def build_top(v):
            if v not in visited:
                visited.add(v)
                for child in v.previous:
                    build_top(child)
                topo.append(v)
        build_top(self)
        self.grad = 1
        # in reverse manner we have to go in network 
        for node in reversed(topo):
            node._backward()