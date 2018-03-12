import os
import json
import string
import wikiwords
import unicodedata
import numpy as np

from collections import Counter
from nltk.corpus import stopwords

words = frozenset(stopwords.words('english'))
punc = frozenset(string.punctuation)
def is_stopword(w):
    return w.lower() in words

def is_punc(c):
    return c in punc

baseline = wikiwords.freq('the')
def get_idf(w):
    return np.log(baseline / (wikiwords.freq(w.lower()) + 1e-10))

def load_data(path):
    from doc import Example
    data = []
    for line in open(path, 'r', encoding='utf-8'):
        if path.find('race') < 0 or np.random.random() < 0.6:
            data.append(Example(json.loads(line)))
    print('Load %d examples from %s...' % (len(data), path))
    return data

class Dictionary(object):
    NULL = '<NULL>'
    UNK = '<UNK>'
    START = 2

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)

    def __init__(self):
        self.tok2ind = {self.NULL: 0, self.UNK: 1}
        self.ind2tok = {0: self.NULL, 1: self.UNK}

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return self.normalize(key) in self.tok2ind

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, self.UNK)
        if type(key) == str:
            return self.tok2ind.get(self.normalize(key),
                                    self.tok2ind.get(self.UNK))

    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.ind2tok[key] = item
        elif type(key) == str and type(item) == int:
            self.tok2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, token):
        token = self.normalize(token)
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def tokens(self):
        """Get dictionary tokens.

        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        tokens = [k for k in self.tok2ind.keys()
                  if k not in {'<NULL>', '<UNK>'}]
        return tokens

vocab, pos_vocab, ner_vocab, rel_vocab = Dictionary(), Dictionary(), Dictionary(), Dictionary()
def gen_race_vocab(data):
    race_vocab = Dictionary()
    build_vocab()
    cnt = Counter()
    for ex in data:
        cnt += Counter(ex.passage.split())
        cnt += Counter(ex.question.split())
        cnt += Counter(ex.choice.split())
    for key, val in cnt.most_common(30000):
        if key not in vocab:
            race_vocab.add(key)
    print('Vocabulary size: %d' % len(race_vocab))
    writer = open('./data/race_vocab', 'w', encoding='utf-8')
    writer.write('\n'.join(race_vocab.tokens()))
    writer.close()

def build_vocab(data=None):
    global vocab, pos_vocab, ner_vocab, rel_vocab
    # build word vocabulary
    if os.path.exists('./data/vocab'):
        print('Load vocabulary from ./data/vocab...')
        for w in open('./data/vocab', encoding='utf-8'):
            vocab.add(w.strip())
        print('Vocabulary size: %d' % len(vocab))
    else:
        cnt = Counter()
        for ex in data:
            cnt += Counter(ex.passage.split())
            cnt += Counter(ex.question.split())
            cnt += Counter(ex.choice.split())
        for key, val in cnt.most_common():
            vocab.add(key)
        print('Vocabulary size: %d' % len(vocab))
        writer = open('./data/vocab', 'w', encoding='utf-8')
        writer.write('\n'.join(vocab.tokens()))
        writer.close()
    # build part-of-speech vocabulary
    if os.path.exists('./data/pos_vocab'):
        print('Load pos vocabulary from ./data/pos_vocab...')
        for w in open('./data/pos_vocab', encoding='utf-8'):
            pos_vocab.add(w.strip())
        print('POS vocabulary size: %d' % len(pos_vocab))
    else:
        cnt = Counter()
        for ex in data:
            cnt += Counter(ex.d_pos)
            cnt += Counter(ex.q_pos)
        for key, val in cnt.most_common():
            if key: pos_vocab.add(key)
        print('POS vocabulary size: %d' % len(pos_vocab))
        writer = open('./data/pos_vocab', 'w', encoding='utf-8')
        writer.write('\n'.join(pos_vocab.tokens()))
        writer.close()
    # build named entity vocabulary
    if os.path.exists('./data/ner_vocab'):
        print('Load ner vocabulary from ./data/ner_vocab...')
        for w in open('./data/ner_vocab', encoding='utf-8'):
            ner_vocab.add(w.strip())
        print('NER vocabulary size: %d' % len(ner_vocab))
    else:
        cnt = Counter()
        for ex in data:
            cnt += Counter(ex.d_ner)
        for key, val in cnt.most_common():
            if key: ner_vocab.add(key)
        print('NER vocabulary size: %d' % len(ner_vocab))
        writer = open('./data/ner_vocab', 'w', encoding='utf-8')
        writer.write('\n'.join(ner_vocab.tokens()))
        writer.close()
    # Load conceptnet relation vocabulary
    assert os.path.exists('./data/rel_vocab')
    print('Load relation vocabulary from ./data/rel_vocab...')
    for w in open('./data/rel_vocab', encoding='utf-8'):
        rel_vocab.add(w.strip())
    print('Rel vocabulary size: %d' % len(rel_vocab))

def gen_submission(data, prediction):
    assert len(data) == len(prediction)
    writer = open('out-%d.txt' % np.random.randint(10**18), 'w', encoding='utf-8')
    for p, ex in zip(prediction, data):
        p_id, q_id, c_id = ex.id.split('_')[-3:]
        writer.write('%s,%s,%s,%f\n' % (p_id, q_id, c_id, p))
    writer.close()

def gen_debug_file(data, prediction):
    writer = open('./data/output.log', 'w', encoding='utf-8')
    cur_pred, cur_choices = [], []
    for i, ex in enumerate(data):
        if i + 1 == len(data):
            cur_pred.append(prediction[i])
            cur_choices.append(ex.choice)
        if (i > 0 and ex.id[:-1] != data[i - 1].id[:-1]) or (i + 1 == len(data)):
            writer.write('Passage: %s\n' % data[i - 1].passage)
            writer.write('Question: %s\n' % data[i - 1].question)
            for idx, choice in enumerate(cur_choices):
                writer.write('%s  %f\n' % (choice, cur_pred[idx]))
            writer.write('\n')
            cur_pred, cur_choices = [], []
        cur_pred.append(prediction[i])
        cur_choices.append(ex.choice)

    writer.close()

def gen_final_submission(data):
    import glob
    proba_list = []
    for f in glob.glob('./out-*.txt'):
        print('Process %s...' % f)
        lines = open(f, 'r', encoding='utf-8').readlines()
        lines = map(lambda s: s.strip(), lines)
        lines = list(filter(lambda s: len(s) > 0, lines))
        assert len(lines) == len(data)
        proba_list.append(lines)
    avg_proba, p_q_id = [], []
    for i in range(len(data)):
        cur_avg_p = np.average([float(p[i].split(',')[-1]) for p in proba_list])
        cur_p_q_id = ','.join(data[i].id.split('_')[-3:-1])
        if i == 0 or cur_p_q_id != p_q_id[-1]:
            avg_proba.append([cur_avg_p])
            p_q_id.append(cur_p_q_id)
        else:
            avg_proba[-1].append(cur_avg_p)
    gen_debug_file(data, [p for sublist in avg_proba for p in sublist])
    writer = open('answer.txt', 'w', encoding='utf-8')
    assert len(avg_proba) == len(p_q_id)
    cnt = 0
    for probas, cur_p_q_id in zip(avg_proba, p_q_id):
        cnt += 1
        assert len(probas) > 1
        pred_ans = np.argmax(probas)
        writer.write('%s,%d' % (cur_p_q_id, pred_ans))
        if cnt < len(p_q_id):
            writer.write('\n')
    writer.close()
    os.system('zip final_output.zip answer.txt')
    print('Please submit final_output.zip to codalab.')

def eval_based_on_outputs(path):
    dev_data = load_data('./data/dev-data-processed.json')
    label = [int(ex.label) for ex in dev_data]
    gold, cur_gold = [], []
    for i, ex in enumerate(dev_data):
        if i + 1 == len(dev_data):
            cur_gold.append(label[i])
        if (i > 0 and ex.id[:-1] != dev_data[i - 1].id[:-1]) or (i + 1 == len(dev_data)):
            gy = np.argmax(cur_gold)
            gold.append(gy)
            cur_gold = []
        cur_gold.append(label[i])
    prediction = [s.strip() for s in open(path, 'r', encoding='utf-8').readlines() if len(s.strip()) > 0]
    prediction = [int(s.split(',')[-1]) for s in prediction]
    assert len(prediction) == len(gold)
    acc = sum([int(p == g) for p, g in zip(prediction, gold)]) / len(gold)
    print('Accuracy on dev_data: %f' % acc)

if __name__ == '__main__':
    # build_vocab()
    trial_data = load_data('./data/trial-data-processed.json')
    train_data = load_data('./data/train-data-processed.json')
    dev_data = load_data('./data/dev-data-processed.json')
    test_data = load_data('./data/test-data-processed.json')
    build_vocab(trial_data + train_data + dev_data + test_data)
