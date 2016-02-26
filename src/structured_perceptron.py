import numpy as np
import random
import sys
from sentence import Sentence

class StructuredPerceptron(object):

    def __init__(self, n_iter, fe, l_rate):
        self.n_iter = n_iter
        self.fe = fe
        self.l_rate = l_rate

    def init_weights(self):
        # have a weight vector for every class
        self.weights = np.zeros((self.n_classes, self.n_feats))
        for j in range(self.n_classes):
            for i in range(self.n_feats):
                self.weights[j][i] = random.uniform(0.0001, 0.001)
        self.bias = [0.0] * self.n_classes

    def sparsify(self, feature_matrix):
        sparse_matrix = []
        for elem in feature_matrix:
            sparse_row = []
            for i, feat in enumerate(elem):
                if feat != 0:
                    sparse_row.append((i, feat))
            sparse_matrix.append(sparse_row)
        return sparse_matrix


    def create_features(self, x_vect, y_vect):
        sequence = Sentence(x_vect, y_vect)
        feat = self.fe.extract(sequence)
        return self.sparsify(feat)

    def update_weights(self, feat_prime, y_prime, feat_hat, y_hat):
        for elem_index in range(len(feat_prime)):
            y_p = y_prime[elem_index]
            y_h = y_hat[elem_index]

            for (i, x) in feat_prime[elem_index]:
                self.weights[y_p][i] += self.l_rate * x
            for (i, x) in feat_hat[elem_index]:
                self.weights[y_h][i] -= self.l_rate * x

            self.bias[y_p] += self.l_rate
            self.bias[y_h] -= self.l_rate

    def dot_product(self, x_vect, y):
        dp = 0.0
        for (i, x) in x_vect:
            dp += x * self.weights[y][i]
        dp += self.bias[y]
        return dp

    def different(self, y_hat, y_prime):
        for i in range(len(y_hat)):
            if y_hat[i] != y_prime[i]:
                return True
        return False

    def fit(self, sequences):
        print "fitting a structured perceptron model..."
        labels = [s.labels for s in sequences]
        labels = [item for sublist in labels for item in sublist]
        self.dict_labels = dict(zip(range(0, len(set(labels))), set(labels)))
        inverse_dict = dict(map(reversed, self.dict_labels.items()))
        labels = [inverse_dict[i] for i in labels]

        self.n_classes = len(set(labels))
        self.n_feats = self.fe.num_feats

        self.init_weights()

        for i in range(self.n_iter):
            print "iteration: ", i
            for s in sequences:
                x = s.tokens
                y_prime = [inverse_dict[lb] for lb in s.labels]
                prob, y_hat = self.viterbi(s)
                phi_prime = self.create_features(x, y_prime)
                phi_hat = self.create_features(x, y_hat)

                if self.different(y_hat, y_prime):
                    self.update_weights(phi_prime, y_prime, phi_hat, y_hat)

    def viterbi(self, obs):
        obs = self.sparsify(self.fe.extract([obs]))
        V = [{}]
        path = {}

        # initialize base cases
        for y in range(self.n_classes):
            V[0][y] = self.dot_product(obs[0], y)
            path[y] = [y]

        # run viterbi for t > 0
        for t in range(1, len(obs)):
            V.append({})
            newpath = {}

            for y in range(self.n_classes):
                (prob, state) = max((V[t-1][y0] + self.dot_product(obs[t], y), y0)
                                for y0 in range(self.n_classes))
                V[t][y] = prob
                newpath[y] = path[state] + [y]

            # don't remember old paths
            path = newpath

        # return the most likely sequence over the given time frame
        n = len(obs) - 1
        (prob, state) = max((V[n][y], y) for y in range(self.n_classes))
        return (prob, path[state])

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

        print "doone."
        return np.array(ret_tok), np.array(ret_seq)


