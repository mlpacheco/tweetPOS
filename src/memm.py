import numpy as np
import random
import sys
import math

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
    def dot_product(self, x_vect, y):
        dp = 0.0
        for (i, x) in x_vect:
            dp += x * self.weights[y][i]
        dp += self.bias[y]
        return dp

    def softmax(self, x_vect, i, a):
        num = np.exp(self.dot_product(x_vect, i) - a)
        denom = np.sum([np.exp(self.dot_product(x_vect, j) - a)
                        for j in range(self.n_classes)])
        return num/denom

    def argmax(self, x_vect):
        y_probs = np.array([self.softmax(x_vect, i)
                            for i in range(self.n_classes)])
        return y_probs.argmax()

    def update_weights(self, x_vect, y):
        # update includes regularization parameter
        # use constant a to deal with overflow
        a = max([self.dot_product(x_vect, i)
                 for i in range(self.n_classes)])
        probs = [self.softmax(x_vect, i, a)
                 for i in range(self.n_classes)]
        #print probs
        #print "weights", self.weights
        for j in range(self.n_classes):
            for i, x in x_vect:
                if j == y:
                    self.weights[j][i] += \
                        (x - probs[j] * x) \
                        + self.lmbd * self.weights[j][i]
                    self.bias[j] += 1 - probs[j]
                else:
                    self.weights[j][i] += \
                         - probs[j] * x \
                         + self.lmbd * self.weights[j][i]
                    self.bias[j] += - probs[j]

    def fit(self, samples, labels, n_feats):
        print "fitting a multinomial logistic regression model..."
        self.dict_labels = dict(zip(range(0, len(set(labels))), set(labels)))
        inverse_dict = dict(map(reversed, self.dict_labels.items()))
        labels = [inverse_dict[i] for i in labels]

        self.n_classes = len(set(labels))
        self.n_feats = n_feats

        self.init_weights()

        for i in range(self.n_iter):
            print "iteration: ", i
            for x, y in zip(samples, labels):
                self.update_weights(x, y)
        print "done."

    def predict_sequences(self, observation_list):
        print "decoding sequences..."
        y_seq = []
        for i, obs in enumerate(observation_list):
            y_seq.append(self.viterbi(obs))
            sys.stdout.write("\r\textracted {0}% of sequences".\
                    format((i * 100) / len(observation_list)))
        sys.stdout.write('\n')

        ret_tok = []
        ret_seq = []

        for (prob, path) in y_seq:
            path = [self.dict_labels[x] for x in path]
            ret_tok += path
            ret_seq.append(" ".join(path))

        print "done."
        return np.array(ret_tok), np.array(ret_seq)

    def viterbi(self, obs):
        V = [{}]
        path = {}

        # initialize base cases
        for y in range(self.n_classes):
            a = max([self.dot_product(obs[0], i)
                 for i in range(self.n_classes)])
            V[0][y] = self.softmax(obs[0], y, a)
            path[y] = [y]

        # run viterbi for t > 0
        for t in range(1, len(obs)):
            V.append({})
            newpath = {}

            a = max([self.dot_product(obs[t], i)
                 for i in range(self.n_classes)])

            for y in range(self.n_classes):
                (prob, state) = max((V[t-1][y0] * self.softmax(obs[t], y, a), y0)
                                for y0 in range(self.n_classes))
                V[t][y] = prob
                newpath[y] = path[state] + [y]

            # don't remember old paths
            path = newpath

        # return the most likely sequence over the given time frame
        n = len(obs) - 1
        (prob, state) = max((V[n][y], y) for y in range(self.n_classes))
        return (prob, path[state])
