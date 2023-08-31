"""
This script contains the code for the smallest component for the small-scale extremely simple autograd engine.
This code is greatly inspired by the gracious instructions of Andrej Karpathy:
https://www.youtube.com/watch?v=VMj-3S1tku0
"""

from typing import List, Union
import src.NN_from_scratch.computation as comp


# let's create the building unit of the engine, The 'variable'
class Variable:
    # create class variables needed for easier manipulation
    # class variable to save the number of 'Variable' instances
    _num_vars = 0
    _epsilon = 10 ** -8
    _gradient_step = 10 ** -5

    @classmethod
    def get_other_value(cls, other: Union[float, int, 'Variable']) -> float:
        return other.value if isinstance(other, Variable) else other

    @classmethod
    def get_other(cls, other: Union[float, int, 'Variable']) -> 'Variable':
        return other if isinstance(other, Variable) else Variable(other)

    def __new__(cls, *args, **kwargs):
        # the main idea here is to increase the number of variables
        cls._num_vars += 1

    def __init__(self,
                 value: float,
                 grad: float = 1,
                 _children: List[Union[float, int, 'Variable']] = None,
                 _operation: str = None,
                 label: str = None):
        # value is simply the numerical value saved in the variable
        self.value = value

        # the grad variable is a bit tricky to define: It satisfies 2 properties
        # it saves the gradient of the last 'Variable' on which the backward method was called
        # with respect to this variable: Assuming L = L(a) (a is self), then L.backward() will set
        # self.grad to dL / da
        # the default is the gradient of the variable with respect to itself: slightly counter-intuitive
        # but find a way to wrap ur head about it
        self.grad = grad

        # since each variable can be written as a function of other variables: need to save the
        # other variables: self._children and the relation between: self.operation

        self._children = [] if _children is None else _children
        self._operation = _operation

        # label needed for readability
        self.label = f'v_{self._num_vars}' if label is None else label
        # add the current label to the cls variable '_vars'

    def __repr__(self) -> str:
        return f'variable {1}'

    def __del__(self):
        # make sure to decrement the cls.variable _num_vars
        # and remove the associated label from the class dictionary
        self._num_vars -= 1

    # let's add the operators
    def __add__(self, other: Union['Variable', float, int]) -> 'Variable':
        # the idea is quite simple
        return Variable(value=(self.value + self.get_other_value(other)),
                        _operation=comp.ADD,
                        _children=[self, self.get_other(other)])

    def __sub__(self, other: Union['Variable', float, int]) -> 'Variable':
        # the idea is quite simple
        return Variable(value=self.value - self.get_other_value(other),
                        _operation=comp.SUB,
                        _children=[self, self.get_other(other)])

    def __mul__(self, other: Union['Variable', float, int]) -> 'Variable':
        return Variable(value=self.value * self.get_other_value(other),
                        _operation=comp.MUL,
                        _children=[self, self.get_other(other)])

    def __truediv__(self, other: Union['Variable', float, int]) -> 'Variable':
        return Variable(value=self.value / (self.get_other_value(other) + self._epsilon * (other.value != 0)),
                        _operation=comp.DIV,
                        _children=[self, self.get_other(other)])

    def __pow__(self, other: Union['Variable', float, int], modulo=None):
        return Variable(value=pow(self.value, (self.get_other_value(other))),
                        _operation=comp.EXP,
                        _children=[self, self.get_other(other)])

    def backward(self, chain_gradient: float = 1):
        # set the gradient regardless
        # (it is the base case also)
        self.grad = chain_gradient

        # the first non-base case is having children variables
        if len(self._children) > 0:
            # let's define The current variable in terms of the children
            children_function = lambda c1, c2: comp.compute(c1, c2, self._operation)
            # we use the children_function to compute
            v1_grad = comp.compute_gradient(children_function, self._gradient_step, 0,
                                            self._children[0], self._children[1])

            v2_grad = comp.compute_gradient(children_function, self._gradient_step, 1,
                                            self._children[0], self._children[1])

            # call the backward operation on both children nodes
            self._children[0].backward(chain_gradient=v1_grad)

            self._children[1].backward(chain_gradient=v2_grad)

