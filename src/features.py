import re

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class FeatureExtractor(object):

    def extract(self, s):
        if type(s) == list:
            return self.extract_multiple_sequences(s)
        else:
            return self.extract_sequence(s)

    def extract_multiple_sequences(self, sequences):
        ret = self.extract_sequence(sequences[0])
        for i in range(1, len(sequences)):
            s = sequences[i]
            ret = np.vstack((ret, self.extract_sequence(s)))
        return ret

'''merge al extracted features'''
class MergeFE(FeatureExtractor):

    def __init__(self, basis_fe):
        self._underlying_fe = basis_fe

    def train(self, sequences):
        self.num_feats = 0
        for fe in self._underlying_fe:
            fe.train(sequences)
            self.num_feats += fe.num_feats

    def extract_sequence(self, sequence):
        ret = self._underlying_fe[0].extract_sequence(sequence)
        for i in range(1, len(self._underlying_fe)):
            fe = self._underlying_fe[i]
            ret = np.concatenate((ret, fe.extract_sequence(sequence)), axis=1)
        return np.asarray(ret)

'''contains ngram: binary'''
class UnigramsFE(FeatureExtractor):

    def train(self, sequences):
        unigrams = []
        for s in sequences:
            unigrams += s.preproc_tokens
        self.feats = dict(zip(set(unigrams), range(0, len(set(unigrams)))))
        self.num_feats = len(self.feats.keys())

    def extract_sequence(self, sequence):
        ret = []
        for i, g in enumerate(sequence.preproc_tokens):
            f = [0.0] * self.num_feats
            try:
                f[self.feats[g]] = 1.0
            except:
                pass
            ret.append(f)
        return np.asarray(ret)

'''contains digits: binary'''
class DigitsFE(FeatureExtractor):

    def train(self, sequences):
        self.num_feats = 1

    def extract_sequence(self, sequence):
        ret = []
        for token in sequence.tokens:
            if re.match(r'.*\d.*', token):
                ret.append([1.0])
            else:
                ret.append([0.0])
        return np.asarray(ret)

'''contains hyphens: binary'''
class HyphensFE(FeatureExtractor):

    def train(self, sequences):
        self.num_feats = 1

    def extract_sequence(self, sequence):
        ret = []
        for token in sequence.tokens:
            if re.match(r'.*-.*', token):
                ret.append([1.0])
            else:
                ret.append([0.0])
        return np.asarray(ret)

'''contains suffix: binary'''
class SuffixFE(FeatureExtractor):

    def __init__(self, length):
        self.length = length

    def train(self, sequences):
        suffixes = []
        for s in sequences:
            suffixes += [t[-self.length:] for t in s.preproc_tokens]
        self.feats = dict(zip(set(suffixes), range(0, len(set(suffixes)))))
        self.num_feats = len(self.feats.keys())

    def extract_sequence(self, sequence):
        suffixes = [t[-self.length:] for t in sequence.preproc_tokens]
        ret = []
        for i, s in enumerate(suffixes):
            f = [0.0] * self.num_feats
            try:
                f[self.feats[s]] = 1.0
            except:
                pass
            ret.append(f)
        return np.asarray(ret)

'''is capitalized: binary'''
class CapitalizedFE(FeatureExtractor):

    def train(self, sequences):
        self.num_feats = 1

    def extract_sequence(self, sequence):
        ret = []
        for token in sequence.tokens:
            if re.match(r'^[A-Z][a-z].+$', token):
                ret.append([1.0])
            else:
                ret.append([0.0])
        return np.asarray(ret)

'''is in caps lock: binary'''
class CapsLockFE(FeatureExtractor):

    def train(self, sequences):
        self.num_feats = 1

    def extract_sequence(self, sequence):
        ret = []
        for token in sequence.tokens:
            if re.match(r'^[A-Z]+$', token):
                ret.append([1.0])
            else:
                ret.append([0.0])
        return np.asarray(ret)

'''previous tags: binary'''
class PreviousTagsFE(FeatureExtractor):

    def __init__(self, n):
        self.n = n

    def train(self, sequences):
        tags = []
        for s in sequences:
            for i in range(0, self.n):
                tags.append("START")
            for i in range(self.n, len(s.labels)):
                tags.append(s.labels[i-self.n])
        self.feats = dict(zip(set(tags),
                          range(0, len(set(tags)))))
        self.num_feats = len(self.feats.keys())

    def extract_sequence(self, sequence):
        tags = []
        for i in range(0, min(self.n, len(sequence.labels))):
            tags.append("START")
        for i in range(self.n, len(sequence.labels)):
            tags.append(sequence.labels[i-self.n])

        ret = []
        for i, t in enumerate(tags):
            f = [0.0]*len(self.feats)
            try:
                f[self.feats[t]] = 1.0
            except:
                pass
            ret.append(f)

        return np.asarray(ret)

'''rt pattern: binary'''
class RetweetFE(FeatureExtractor):

    def train(self, sequences):
        self.num_feats = 1

    def extract_sequence(self, sequence):
        ret = []
        for token in sequence.tokens:
            if re.match(r'RT', token):
                ret.append([1.0])
            else:
                ret.append([0.0])
        return np.asarray(ret)

'''username pattern: binary'''
class UsernameFE(FeatureExtractor):

    def train(self, sequences):
        self.num_feats = 1

    def extract_sequence(self, sequence):
        ret = []
        for token in sequence.tokens:
            if re.match(r'@\w+', token):
                ret.append([1.0])
            else:
                ret.append([0.0])
        return np.asarray(ret)

'''hashtag pattern: binary'''
class HashtagFE(FeatureExtractor):

    def train(self, sequences):
        self.num_feats = 1

    def extract_sequence(self, sequence):
        ret = []
        for token in sequence.tokens:
            if re.match(r'#\w+', token):
                ret.append([1.0])
            else:
                ret.append([0.0])
        return np.asarray(ret)

'''url pattern: binary'''
class UrlFE(FeatureExtractor):

    def train(self, sequences):
        self.num_feats = 1

    def extract_sequence(self, sequence):
        ret = []
        for token in sequence.tokens:
            if re.match(r'http://.*', token):
                ret.append([1.0])
            else:
                ret.append([0.0])
        return np.asarray(ret)
