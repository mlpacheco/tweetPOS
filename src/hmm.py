from collections import Counter
import numpy as np

class HMM(object):

    def __init__(self):
        self.state_counts = {}
        self.states = {}
        self.priors = {}
        self.transitions = {}
        self.emissions = {}

    def get_counts(self, sequences):
        self.state_counts = Counter()
        pair_counts = Counter()
        state_seq_counts = Counter()
        for sentence in sequences:
            self.state_counts.update(Counter(sentence.labels))
            pairs = [(token, pos) for (token, pos)\
                     in zip(sentence.preproc_tokens, sentence.labels)]
            pair_counts.update(Counter(pairs))
            state_pairs = zip(*[sentence.labels[x:] for x in range(2)])
            state_seq_counts.update(Counter(state_pairs))
        return pair_counts, state_seq_counts


    def init_probabilities(self, sequences):
        pair_counts, state_seq_counts = \
            self.get_counts(sequences)
        #self.most_common = self.state_counts.most_common(1)[0][0]
        num_obs = sum(self.state_counts.values())

        self.states = self.state_counts.keys()

        # init priors
        for key, value in self.state_counts.items():
            self.priors[key] = (value*1.0 + 1.0)/(num_obs + 1.0)

        # init emissions
        for key, value in pair_counts.items():
            if key[1] not in self.emissions:
                self.emissions[key[1]] = {}
            else:
                self.emissions[key[1]][key[0]] =\
                    (value*1.0 + 1.0)/(self.state_counts[key[1]] + 1.0)

        # init transitions
        for key, value in state_seq_counts.items():
            if key[1] not in self.transitions:
                self.transitions[key[1]] = {}
            else:
                self.transitions[key[1]][key[0]] =\
                    (value*1.0)/(self.state_counts[key[1]] + 1.0)


    def get_emissions(self, state, obs):
        if state in self.emissions and \
           obs in self.emissions[state]:
            return self.emissions[state][obs]
        else:
            return 1.0/(self.state_counts[state] + 1.0)

    def get_transitions(self, prev_state, state):
        if state in self.transitions and \
           prev_state in self.transitions[state]:
            return self.transitions[state][prev_state]
        else:
            return 1.0/(self.state_counts[state] + 1.0)

    def viterbi(self, obs):
        V = [{}]
        path = {}

        # initialize base cases
        for y in self.states:
            V[0][y] = self.priors[y] * self.get_emissions(y, obs[0])
            path[y] = [y]

        # run viterbi for t > 0
        for t in range(1, len(obs)):
            V.append({})
            newpath = {}

            for y in self.states:
                (prob, state) = max((V[t-1][y0] *
                                     self.get_transitions(y0, y) *
                                     self.get_emissions(y, obs[t]), y0)
                                for y0 in self.states)
                V[t][y] = prob
                newpath[y] = path[state] + [y]

            # don't remember old paths
            path = newpath

        # return the most likely sequence over the given time frame
        n = len(obs) - 1
        (prob, state) = max((V[n][y], y) for y in self.states)
        return (prob, path[state])

    def fit(self, sequences):
        self.init_probabilities(sequences)

    def predict(self, sentence):
        print self.viterbi(sentence)


