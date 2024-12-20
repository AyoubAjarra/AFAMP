import itertools
import random
import numpy as np
import math

def init_seed(seed=42):
    global rng
    rng = np.random.RandomState(seed=seed)
       
## Single Fourier coefficient estimate

def random_input_generator(n):
    """
    returns a random {-1,1}-vector of length `n`.
    """
    return np.random.choice((-1, +1), n)

def random_inputs(n, num):
    """
    returns an iterator for a random sample of {-1,1}-vectors of length `n` (with replacement).
    """
    for i in range(num):
        yield random_input_generator(n)

def estimate_fourier_coefficient(h, S, sample_size):
    """
    Estimates the Fourier coefficient of `f` on subset `S` by evaluating the function on `sample_size`
    random inputs.
    """
    return np.mean([h.eval(x) * uniform_basis(S, x) for x in random_inputs(h.input_length,sample_size)])

'''
def sample_inputs(n, num):
    """
    returns an iterator for either random samples of {-1,1}-vectors of length `n` if `num` < 2^n,
    and an iterator for all {-1,1}-vectors of length `n` otherwise.
    Note that we return only 2^n vectors even with `num` > 2^n.
    In other words, the output of this function is deterministic if and only if num >= 2^n.
    """
    return random_inputs(n, num) if num < 2**n else itertools.product((-1, +1), repeat=n)
'''

## Goldreich Levin setup

def uniform_basis(s, x):
    assert len(s) == len(x), "s and x must have the same length"
    s = np.array(s,dtype =int)
    result = np.prod([x[i] if s[i] == 1 else 1 for i in range(len(s))])
    
    return result

def projection(v1, v2):
    return (np.dot(v1, v2) / np.dot(v2, v2)) * v2

def Gram_Schmidt_basis(S, X):
    orthogonal_basis = []
    S_list = [S] if isinstance(S, np.ndarray) and S.ndim == 1 else S
    for s in S_list:
        v = uniform_basis(s, X)
        v = np.array([v], dtype=float)
        for u in orthogonal_basis:
            v -= projection(v, u)
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-10:  
            v = v / v_norm  
        orthogonal_basis.append(v)
    return orthogonal_basis

class GoldreichLevin:
    def __init__(self, model, tau, delta,budget):
        self.model = model
        self.tau = tau
        epsilon = tau ** 2 / 4
        self.denom = tau ** 2 / (8 * self.model.input_length * (1 - delta))
        self.sample_size = budget * int(math.ceil(12 * math.log(2.0 / self.denom) / (epsilon ** 2)))
        self.history = {}

    def find_heavy_monomials(self):
        initial_bucket = (0, tuple(np.zeros(self.model.input_length)))
        return self.dynamic_programming_find(initial_bucket)

    def dynamic_programming_find(self, initial_bucket):
        queue = [initial_bucket]
        result = []

        while queue:
            current_bucket = queue.pop(0)

            if current_bucket in self.history:
                result.extend(self.history[current_bucket])
                continue

            k, S = current_bucket
            if k == self.model.input_length:
                result.append((S, self.sample_weight(current_bucket)))
            else:
                extended_S = np.copy(S)
                extended_S[k] = 1
                next_buckets = [(k + 1, S), (k + 1, tuple(extended_S))]

                for new_bucket in next_buckets:
                    current_weight = self.sample_weight(new_bucket)
                    if current_weight > self.tau ** 2 / 2:
                        queue.append(new_bucket)

        return result


    def sample_weight(self, bucket):
        k = bucket[0] 
        S = bucket[1] 
        J = np.where(np.arange(self.model.input_length) < k, 1, 0) # Initialize T
        estimate = 0.0
        i = 0
        while i < self.sample_size:
            z = random_input_generator(self.model.input_length - k) 
            x = np.append(random_input_generator(k), z) 
            y = np.append(random_input_generator(k), z) 
            estimate += self.model.eval(x) * uniform_basis(np.multiply(S, J), x) * self.model.eval(y) * uniform_basis(np.multiply(S, J), y)
            i +=1
        estimate /= self.sample_size
        return estimate