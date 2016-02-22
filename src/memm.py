import numpy as np
import random

class MEMM(object):

    def __init__(self, n_iter, lmbd):
        self.n_iter = n_iter
        self.lmbd = lmbd

    def init_weights(self):
        # have a weight vector for every class
        self.weights = np.zeros((self.n_classes, self.n_feats))
        for j in range(self.n_classes):
            for i in range(self.n_feats):
                self.weights[j][i] = random.uniform(0.0001, 0.001)
        self.bias = [0.0] * self.n_classes

    # keeping it a separate function
    # in case we have to deal with sparsity here
    def dot_product(self, x_vect, i):
        dp = np.dot(x_vect, self.weights[i].T)
        return np.add(dp, self.bias[i])

    def softmax(self, x_vect, i):
        num = np.exp(self.dot_product(x_vect, i))
        denom = np.sum([np.exp(self.dot_product(x_vect, j))
                        for j in range(self.n_classes)])
        return num/denom

    def argmax(self, x_vect):
        y_probs = np.array([self.softmax(x_vect, i)
                            for i in range(self.n_classes)])
        return y_probs.argmax()

    def update_weights(self, x_vect, y):
        # update includes regularization parameter
        probs = [self.softmax(x_vect, i) for i in range(self.n_classes)]
        for j in range(self.n_classes):
            for i in range(self.n_feats):
                if j == y:
                    self.weights[j][i] += \
                        (x_vect[i] - probs[j] * x_vect[i]) \
                        + self.lmbd * weights[j][i]
                    self.bias[j] += 1 - probs[j]
                else:
                    self.weights[j][i] += \
                         - probs[j] * x_vect[i] \
                         + self.lmbd * weights[j][i]
                    self.bias[j] += - probs[j]

    def fit(self, samples, labels):
        self.dict_labels = dict(zip(set(labels), range(0, len(set(labels)))))
        labels = [self.dict_labels[i] for i in labels]

        self.n_classes = len(set(labels))
        self.n_feats = samples.shape[1]

        self.init_weights()

        for i in range(self.n_iter):
            for x, y in zip(samples, labels):
                self.update_weights(x, y)

    def predict(self, samples):
        y_pred = [self.argmax(x) for x in samples]
        return np.array([self.dict_labels[y] for y in y_pred])

    def viterbi(self, obs):
        V = [{}]
        path = {}

        # initialize base cases
        for y in range(self.n_classes):
            V[0][y] = self.softmax(obs[0])
            path[y] = [y]

        # run viterbi for t > 0
        for t in range(1, len(obs)):
            V.append({})
            newpath = {}

            for y in self.states:
                (prob, state) = max((V[t-1][y0] * self.softmax(obs[t], y), y0)
                                for y0 in self.states)
                V[t][y] = prob
                newpath[y] = path[state] + [y]

            # don't remember old paths
            path = newpath

        # return the most likely sequence over the given time frame
        n = len(obs) - 1
        (prob, state) = max((V[n][y], y) for y in self._states)
        return (prob, path[state])
