import random
import numpy as np


MAX_SEED = 2 ** 32

float32 = 'float32'


class Graph:
    def __init__(self):
        self.reset()

    def set_env(self, new_env):
        self.env = new_env

    def set_seed(self):
        self.seed = random.randrange(MAX_SEED)

    def add_variable(self, variable):
        self.variables.append(variable)

    def reset(self):
        self.env = {}
        self.set_seed()
        self.variables = []

    def as_default(self):
        return GraphContext(self)


class GraphContext:
    def __init__(self, graph):
        self.graph = graph

    def __enter__(self):
        global _default_graph
        self.prev_graph = get_default_graph()
        _default_graph = self.graph

    def __exit__(self, type, value, traceback):
        global _default_graph
        _default_graph = self.prev_graph


_default_graph = Graph()


def get_default_graph():
    return _default_graph


class Op:
    def __init__(self, name, shape, dtype):
        self.name = name
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(shape)
        self.dtype = dtype

    def __add__(self, other):
        if isinstance(other, Op):
            return AddOp(self, other)
        return AddOp(self, ConstantOp(other))

    def __repr__(self):
        return 'Tensor("{}", shape={}, dtype={})'.format(
                    self.name, self.shape, self.dtype)


class ConstantOp(Op):
    def __init__(self, val, shape=None, dtype=None):
        self.val = np.array(val)
        super().__init__('Const', self.val.shape, self.val.dtype)

    def evaluate(self):
        return self.val


def constant(val, shape=(), dtype=float32):
    return ConstantOp(val, shape, dtype)


class RandomUniformOp(Op):
    def __init__(self, shape=None, dtype=None):
        super().__init__('random_uniform', shape, dtype)

    def evaluate(self):
        np.random.seed(get_default_graph().seed)
        return np.random.uniform(0, 1, self.shape)


def random_uniform(shape=(), dtype=float32):
    return RandomUniformOp(shape, dtype)


class AddOp(Op):
    def __init__(self, val1, val2):
        super().__init__('add', val1.shape, val1.dtype)
        self.val1 = val1
        self.val2 = val2

    def evaluate(self):
        return self.val1.evaluate() + self.val2.evaluate()


class PlaceholderOp(Op):
    def __init__(self, shape=None, dtype=None):
        super().__init__('Placeholder', shape, dtype)

    def evaluate(self):
        return np.array(get_default_graph().env[self], dtype=self.dtype)


def placeholder(dtype, shape=()):
    return PlaceholderOp(shape, dtype)


class Initializer():
    def __init__(self, init_fn):
        self.init_fn = init_fn

    def set_var(self, variable):
        self.variable = variable

    def evaluate(self):
        self.variable.assign(self.init_fn(self.variable))


def random_uniform_initializer():
    return Initializer(
        lambda v: RandomUniformOp(shape=v.shape).evaluate())


class VariableOp(Op):
    def __init__(self, name, shape=None, dtype=None, initializer=None):
        super().__init__(name, shape, dtype)
        get_default_graph().add_variable(self)
        self.initializer = initializer
        if callable(self.initializer):
            self.initializer = self.initializer()
        self.initializer.set_var(self)

    def assign(self, value):
        self.value = value

    def evaluate(self):
        return self.value


def get_variable(
        name, shape=(), dtype=float32,
        initializer=random_uniform_initializer):
    return VariableOp(name, shape, dtype, initializer)


def global_variables_initializer():
    return [v.initializer for v in get_default_graph().variables]


class layers:
    class Layer(Op):
        def __init__(self, name, shape=None, dtype=None):
            super().__init__(name, shape, dtype)

    class Dense(Layer):
        def __init__(self, units=1):
            self.units = units
            self.shape = (self.units,)
            super().__init__('dense', shape=self.shape)

        def __call__(self, inputs):
            self.inputs = inputs
            self.dtype = inputs.dtype
            self.weights = get_variable(
                    'dense_weights', shape=(inputs.shape[-1], self.units))
            self.biases = get_variable('dense_biases', shape=(self.units,))
            return self

        def evaluate(self):
            W = self.weights.evaluate()
            x = self.inputs.evaluate()
            b = self.biases.evaluate()
            return np.dot(x, W) + b


def evaluate(graph):
    if isinstance(graph, tuple):
        return tuple(evaluate(g) for g in graph)
    if isinstance(graph, list):
        return [evaluate(g) for g in graph]
    return graph.evaluate()


class Session:
    def __init__(self):
        self.graph = get_default_graph()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def run(self, graph, feed_dict={}):
        self.graph.reset()
        self.graph.set_env(feed_dict)
        return evaluate(graph)

    def close(self):
        self.graph.reset()
