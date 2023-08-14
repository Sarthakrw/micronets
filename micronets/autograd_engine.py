import math

class Value:
    
    """ Initialization of Value Object """
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        
        # variables for graphical representation
        self._backward = lambda : None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    """ Mathematical Operations """
    
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
            
        return out
    
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __neg__(self):
        return self * -1    
    
    def __mul__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out

    def exp(self):
        x = self.data
        out = Value(math.e**x, (self,), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        
        return out
    
    def log(self):
        x = self.data
        epsilon = 1e-15
        out = Value(math.log(x+epsilon), (self,), 'log')
        
        def _backward():
            self.grad += (1/(x+epsilon)) * out.grad
        out._backward = _backward
        
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int,float)) # fixes the value of 'other' to a float or an int
        out = Value(self.data**other, (self,), f'**{other}')
        
        def _backward():
            self.grad += other*(self.data**(other-1)) * out.grad
        out._backward = _backward
            
        return out
    
    """ Activation Functions """
    
    def tanh(self):
        x=self.data
        tanh = ((math.e)**(2*x) - 1) / ((math.e)**(2*x) + 1)
        out = Value(tanh, (self,), 'tanh')
        
        def _backward():
            self.grad += (1-(tanh)**2) * out.grad
        out._backward = _backward
        
        return out
    
    def relu(self):
        x = self.data
        out = Value(0 if x<0 else x, (self,), 'ReLU')
            
        def _backward():
            self.grad += (x >= 0) * out.grad
        out._backward = _backward
        
        return out
    
    def sigmoid(self):
        x = self.data
        sig = 1/(1 + math.e**(-x))
        out = Value(sig, (self,), 'sigmoid')
        
        def _backward():
            self.grad += (sig*(1-sig)) * out.grad
        out._backward = _backward
        
        return out
    
    def linear(self):
        out = Value(self.data, (self,), 'linear')
        
        def _backward():
            self.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out
    
    """ Reverse Dunder Methods"""
   
    def __rmul__(self, other):
        return self * other
    
    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __rtruediv__(self, other): # other / self
        return other * self**-1
    

    """Object Representation (developer use)"""
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    """ Topological Sorting Implementation 
        (determines the order in which gradients should be computed during backpropagation) """

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo. append(v)
        build_topo(self)
        
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()