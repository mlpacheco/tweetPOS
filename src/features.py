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
        for fe in self._underlying_fe:
            fe.train(sequences)

    def extract_sequence(self, sequence):
        ret = self._underlying_fe[0].extract_sequence(sequence)
        for i in range(1, len(self._underlying_fe)):
            fe = self._underlying_fe[i]
            ret = np.concatenate((ret, fe.extract_sequence(sequence)), axis=1)
        return np.asarray(ret)

'''contains ngram: binary'''
class NgramFE(FeatureExtractor):

    def __init__(self, m, n):
        self.m = m
        self.n = n

    def train(self, sequences):
        documents = []
        for s in sequences:
            documents += s.preproc_tokens
        self.ngram_vectorizer = \
            CountVectorizer(ngram_range=(self.m,self.n), binary=True)
        self.ngram_vectorizer.fit(documents)

    def extract_sequence(self, sequence):
        ret = self._ngram_vectorizer.transform(sequence.preproc_tokens)
        return ret.toarray()

'''contains digits: binary'''
class DigitsFE(FeatureExtractor):

    def train(self, sequences):
        pass

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
        pass

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
        documents = []
        for s in sequences:
            documents += [t[-self.length:] for t in s.preproc_tokens]
        self.suffix_vectorizer = \
            CountVectorizer(ngram_range=(1,1), binary=True)
        self.suffix_vectorizer.fit(documents)

    def extract_sequence(self, sequence):
        ret = self.suffix_vectorizer.transform(sequence.preproc_tokens)
        return ret.toarray()

'''is capitalized: binary'''
class CapitalizedFE(FeatureExtractor):

    def train(self, sequences):
        pass

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
        pass

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

    def extract_sequence(self, sequence):
        tags = []
        for i in range(0, min(self.n, len(sequence.labels))):
            tags.append("START")
        for i in range(self.n, len(sequence.labels)):
            tags.append(sequence.labels[i-self.n])

        ret = []
        for i, t in enumerate(tags):
            f = [0.0]*len(self.feats)
            f[self.feats[t]] = 1.0
            ret.append(f)

        return np.asarray(ret)
