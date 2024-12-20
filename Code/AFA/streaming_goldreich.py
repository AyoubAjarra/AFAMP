from utils import *

class GoldreichLevin:
    def __init__(self, model, tau, budget):
        self.model = model
        self.tau = tau
        self.sample_size = budget 
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
                    if current_weight > self.tau ** 2 / 4:
                        queue.append(new_bucket)
        return result
    def sample_weight(self, bucket):
        k = bucket[0]
        S = bucket[1]
        J = np.where(np.arange(self.model.input_length) < k, 1, 0)  # Binary vector
    
        estimate = 0.0
        i = 0
        while i < self.sample_size:
            z = random_input_generator(self.model.input_length - k)
            x = np.append(random_input_generator(k), z)
            y = np.append(random_input_generator(k), z)
        
            # Make sure the multiplication of S and J is handled correctly
            S_mult = np.multiply(S, J)  # This is a binary vector
            estimate += (
                self.model.eval(x)
                * np.sum(Gram_Schmidt_basis(S_mult, x))  # Adjusting how Gram_Schmidt_basis is called
                * self.model.eval(y)
                * np.sum(Gram_Schmidt_basis(S_mult, y))
            )
            i += 1
    
        estimate /= self.sample_size
        return estimate


    def sample_weight(self, bucket):
        k = bucket[0] 
        S = bucket[1] 
        J = np.where(np.arange(self.model.input_length) < k, 1, 0) 
        estimate = 0.0
        i = 0
        while i < self.sample_size:
            z = random_input_generator(self.model.input_length - k) 
            x = np.append(random_input_generator(k), z) 
            y = np.append(random_input_generator(k), z) 
            estimate += np.sum(self.model.eval(x) * Gram_Schmidt_basis(np.multiply(S, J), x) * self.model.eval(y) * Gram_Schmidt_basis(np.multiply(S, J), y))
            i +=1
        estimate = estimate/self.sample_size
        return estimate
    
class Efficient_GoldreichLevin:
    def __init__(self, model, tau, budget):
        self.model = model
        self.tau = tau
        self.sample_size = budget 
        self.history = {}

    def find_heavy_monomials(self):
        initial_bucket = np.zeros(self.model.input_length)
        initial_bucket[0] = 1
        initial_bucket = (1, tuple(initial_bucket))
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
        J = np.where(np.arange(self.model.input_length) < k, 1, 0) 
        estimate = 0.0
        i = 0
        while i < self.sample_size:
            z = random_input_generator(self.model.input_length - k) 
            x = np.append(random_input_generator(k), z) 
            y = np.append(random_input_generator(k), z) 
            estimate += np.sum(self.model.eval(x) * uniform_basis(np.multiply(S, J), x) * self.model.eval(y) * uniform_basis(np.multiply(S, J), y))
            i +=1
        estimate = estimate/self.sample_size
        return estimate