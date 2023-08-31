"""
This script contains functionalities to numerically estimate the derivative of a given function
"""

ADD = '+'
SUB = '-'
DIV = '/'
MUL = '*'
EXP = '**'

symbol_op_map = {ADD: lambda x, y: x + y,
                 SUB: lambda x, y: x - y,
                 DIV: lambda x, y: x / y,
                 MUL: lambda x, y: x * y,
                 EXP: lambda x, y: x ** y}


def compute(arg1: float, arg2: float, operator: str) -> float:
    return symbol_op_map[operator](arg1, arg2)


def compute_gradient(f: callable,
                     step_size: float,
                     arg_index: int,
                     *args) -> float:
    # convert to a list to use the index argument
    args = list(args)
    # this function uses: this approximation of the gradient: (f(x + h) - f(x - h)) // 2h

    args_plus = args.copy()
    args_plus[arg_index] += step_size

    args_minus = args.copy()
    args_minus[arg_index] -= step_size

    gradient = (f(*args_plus) - f(*args_minus)) // (2 * step_size)
    return gradient
