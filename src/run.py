import optparse

from hmm import HMM
from memm import MEMM
from structured_perceptron import StructuredPerceptron

from sentence import Sentence
from features import *

from sklearn import metrics

def parse_dataset(filepath):
    sequences = []; tokens = []; labels = []
    with open(filepath) as f:
        for line in f:
            args = line.split()
            if len(args) == 2:
                tokens.append(args[0])
                labels.append(args[1])
            else:
                sentence = Sentence(tokens, labels)
                sequences.append(sentence)
                tokens = []; labels = []
    return sequences

def get_observations(feats, sequences):
    i = 0; j = 0; observation_list = []
    for s in sequences:
        j += len(s.tokens)
        observation_list.append(feats[i:j])
        i = j
    return observation_list

def sparsify(feature_matrix):
    sparse_matrix = []
    for elem in feature_matrix:
        sparse_row = []
        for i, feat in enumerate(elem):
            if feat != 0:
                sparse_row.append((i, feat))
        sparse_matrix.append(sparse_row)
    return sparse_matrix


parser = optparse.OptionParser()
parser.add_option('-i', help='input training file',
                  dest='input_train', type='string')
parser.add_option('-o', help='output testing file',
                  dest='out_test', type='string')
(opts, args) = parser.parse_args()

train_sequences = parse_dataset(opts.input_train)
test_sequences = parse_dataset(opts.out_test)

fe = MergeFE([
              UnigramsFE()
              ,DigitsFE()
              ,HyphensFE()
              ,SuffixFE(1)
              ,SuffixFE(2)
              ,SuffixFE(3)
              ,CapitalizedFE()
              ,CapsLockFE()
              ,HashtagFE()
              ,UsernameFE()
              ,RetweetFE()
              ,PreviousTagsFE(1)
              ,PreviousTagsFE(2)
              ,UrlFE()
              ])

print "training features..."
fe.train(train_sequences)


print "train tweets", len(train_sequences)
print "test tweets", len(test_sequences)

# Hidden Markov Models
'''
hmm = HMM()
hmm.fit(train_sequences)

for seq in test_sequences:
    print seq.preproc_tokens
    hmm.predict(seq.preproc_tokens)
'''

# MaxEnt Markov Models

print "extracting features..."
train_feat = fe.extract(train_sequences)
print "training set feats:", train_feat.shape

test_feat = fe.extract(test_sequences)
print "testing set feats:", test_feat.shape

print "moving to a sparse representation..."
train_feat = sparsify(train_feat)
test_feat = sparsify(test_feat)

train_labl = [s.labels for s in train_sequences]
train_labl = [item for sublist in train_labl for item in sublist]

obsr_labl = [s.labels for s in test_sequences]
test_seq = [" ".join(x) for x in obsr_labl]
test_tok = [item for sublist in obsr_labl for item in sublist]

obsr_list = get_observations(test_feat, test_sequences)

memm = MEMM(10, 0.0001)
memm.fit(train_feat, train_labl, fe.num_feats)
pred_tok, pred_seq = memm.predict_sequences(obsr_list)


# Structured Perceptron using viterbi in the inference step

'''percep = StructuredPerceptron(10, fe, 0.1)
percep.fit(train_sequences)
pred_tok, pred_seq = percep.predict_sequences(test_sequences)

obsr_labl = [s.labels for s in test_sequences]
test_seq = [" ".join(x) for x in obsr_labl]
test_tok = [item for sublist in obsr_labl for item in sublist]'''


# print results
print metrics.classification_report(test_tok, pred_tok)
print metrics.accuracy_score(test_tok, pred_tok)
print metrics.accuracy_score(test_seq, pred_seq)

