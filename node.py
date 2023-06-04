from uuid import uuid4
from graphviz import Digraph

class Node:
    def __init__(self, val, children = [], op='ass'):
        self.identity = uuid4()
        self.val = val
        self.children = children
        self.op = op
        self.grad = 0
        self.backward_prop = lambda : None

    def __to_comp_node(self, obj):
        if not isinstance(obj,Node):
            return Node(val=obj)
        else:
            return obj
        
    def __sub__(self, other):
        other= self.__to_comp_node(other)
        output = Node(val=self.val-other.val,children=[self,other],op='sub')
        def _backward_prop():
            self.grad += output.grad * 1
            other.grad += output.grad * (-1)
        self.backward_prop = _backward_prop
        return output
    
    def __rsub__(self,other):
        other= self.__to_comp_node(other)
        output = other-self
        return output

    def __add__(self, other):
        other= self.__to_comp_node(other)
        output = Node(val=self.val+other.val,children=[self,other],op='add')
        def _backward_prop():
            self.grad += output.grad * 1
            other.grad += output.grad * 1
        self.backward_prop = _backward_prop
        return output
    
    def __radd__(self,other):
        other= self.__to_comp_node(other)
        output = other + self
        return output
    
    def __mul__(self, other):
        other= self.__to_comp_node(other)
        output = Node(val=self.val*other.val,children=[self,other],op='mul')
        def _backward_prop():
            self.grad += output.grad * other.val
            other.grad += output.grad * self.val
        self.backward_prop = _backward_prop
        return output
    
    def __rmul__(self,other):
        other= self.__to_comp_node(other)
        output = other * self
        return output
    
    def __pow__(self, exponent):
        if not isinstance(exponent,(int,float)):
            raise ValueError("Unsupported Types")
        output = Node(val=self.val**exponent,children=[self],op=f'pow {exponent}')
        def _backward_prop():
            self.grad += (output.grad * (exponent*(self.val)**(exponent-1)))
        self.backward_prop= _backward_prop
        return output
    
    def __eq__(self,other):
        other= self.__to_comp_node(other)
        return self.val == other.val
    
    def __repr__(self):
        return f"op: {self.op} | val: {self.val:.4f} | children: {len(self.children)} | grad: {self.grad}"
    
    def __hash__(self):
        return int(self.identity)
    
    def topo_sort(self, collect_edges = False):
        res = []
        visited = set()
        if collect_edges : edges = []
        def visit(node):
            if node not in visited:
                visited.add(node)
                for child in node.children:
                    if collect_edges:
                        edges.append((child,node))
                res.append(node)
                for child in node.children:
                    visit(child)
        visit(self)
        if collect_edges:
            return res, edges
        return res
    
    def backward(self):
        nodes = self.topo_sort()
        self.grad = 1
        for node in nodes:
            node.backward_prop()

    def draw_graph(self):
        nodes, edges = self.topo_sort(collect_edges = True)
        dot = Digraph(format = 'svg', graph_attr = {'rankdir' : 'TB'})
        for n in nodes:
            dot.node(name = str(hash(n)), label = f"{n.op} | {n.val:.2f} | grad {n.grad:.2f}")
        for n1, n2 in edges:
            dot.edge(str(hash(n1)), str(hash(n2)))
        return dot
    
assert Node(val=5) == 5, "Assignment Failure"
assert (Node(val=5)-Node(val=2)) == 3 , "Subtraction Failure"
assert (Node(val=5)-2) == 3 , "Subtraction Failure"
assert (5 - Node(val=2)) == 3 , "Subtraction Failure"
assert (Node(val=5)+Node(val=2)).val == 7 , "Addition Failure"
assert (Node(val=5)+2) == 7 , "Addition Failure"
assert (5 + Node(val=2)) == 7 , "Addition Failure"
assert (Node(val=5)*Node(val=2)).val == 10 , "Multiplication Failure"
assert (Node(val=5)*2) == 10 , "Multiplication Failure"
assert (5 * Node(val=2)) == 10 , "Multiplication Failure"
assert (Node(val=3)**2 == 9), "Power Failure"