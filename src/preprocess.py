import os
import sys
import spacy
import copy
import json
import math
import wikiwords

from collections import Counter

class Tokens(object):
    """A class to represent a list of tokenized text."""
    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i: j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token

        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if 'pos' not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if 'lemma' not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if 'ner' not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.

        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """
        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [(s, e + 1)
                  for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not _skip(words[s:e + 1])]

        # Concatenate into strings
        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get('non_ent', 'O')
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while (idx < len(entities) and entities[idx] == ner_tag):
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups

class SpacyTokenizer():

    def __init__(self, **kwargs):
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
            model: spaCy model to use (either path, or keyword like 'en').
        """
        model = kwargs.get('model', 'en')
        self.annotators = copy.deepcopy(kwargs.get('annotators', set()))
        nlp_kwargs = {'parser': False}
        if not {'lemma', 'pos', 'ner'} & self.annotators:
            nlp_kwargs['tagger'] = False
        if not {'ner'} & self.annotators:
            nlp_kwargs['entity'] = False
        self.nlp = spacy.load(model, **nlp_kwargs)

    def tokenize(self, text):
        # We don't treat new lines as tokens.
        clean_text = text.replace('\n', ' ').replace('\t', ' ').replace('/', ' / ').strip()
        # remove consecutive spaces
        if clean_text.find('  ') >= 0:
            clean_text = ' '.join(clean_text.split())
        tokens = self.nlp.tokenizer(clean_text)
        if {'lemma', 'pos', 'ner'} & self.annotators:
            self.nlp.tagger(tokens)
        if {'ner'} & self.annotators:
            self.nlp.entity(tokens)

        data = []
        for i in range(len(tokens)):
            # Get whitespace
            start_ws = tokens[i].idx
            if i + 1 < len(tokens):
                end_ws = tokens[i + 1].idx
            else:
                end_ws = tokens[i].idx + len(tokens[i].text)

            data.append((
                tokens[i].text,
                text[start_ws: end_ws],
                (tokens[i].idx, tokens[i].idx + len(tokens[i].text)),
                tokens[i].tag_,
                tokens[i].lemma_,
                tokens[i].ent_type_,
            ))

        # Set special option for non-entity tag: '' vs 'O' in spaCy
        return Tokens(data, self.annotators, opts={'non_ent': ''})


TOK = None

def init_tokenizer():
    global TOK
    TOK = SpacyTokenizer(annotators={'pos', 'lemma', 'ner'})


digits2w = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three',
            '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}
def replace_digits(words):
    global digits2w
    return [digits2w[w] if w in digits2w else w for w in words]

def tokenize(text):
    """Call the global process tokenizer on the input text."""
    global TOK
    tokens = TOK.tokenize(text)
    output = {
        'words': replace_digits(tokens.words()),
        'offsets': tokens.offsets(),
        'pos': tokens.pos(),
        'lemma': tokens.lemmas(),
        'ner': tokens.entities(),
    }
    return output

from utils import is_stopword, is_punc
def compute_features(d_dict, q_dict, c_dict):
    # in_q, in_c, lemma_in_q, lemma_in_c, tf
    q_words_set = set([w.lower() for w in q_dict['words']])
    in_q = [int(w.lower() in q_words_set and not is_stopword(w) and not is_punc(w)) for w in d_dict['words']]
    c_words_set = set([w.lower() for w in c_dict['words']])
    in_c = [int(w.lower() in c_words_set and not is_stopword(w) and not is_punc(w)) for w in d_dict['words']]

    q_words_set = set([w.lower() for w in q_dict['lemma']])
    lemma_in_q = [int(w.lower() in q_words_set and not is_stopword(w) and not is_punc(w)) for w in d_dict['lemma']]
    c_words_set = set([w.lower() for w in c_dict['lemma']])
    lemma_in_c = [int(w.lower() in c_words_set and not is_stopword(w) and not is_punc(w)) for w in d_dict['lemma']]

    tf = [0.1 * math.log(wikiwords.N * wikiwords.freq(w.lower()) + 10) for w in d_dict['words']]
    tf = [float('%.2f' % v) for v in tf]
    d_words = Counter(filter(lambda w: not is_stopword(w) and not is_punc(w), d_dict['words']))
    from conceptnet import concept_net
    p_q_relation = concept_net.p_q_relation(d_dict['words'], q_dict['words'])
    p_c_relation = concept_net.p_q_relation(d_dict['words'], c_dict['words'])
    assert len(in_q) == len(in_c) and len(lemma_in_q) == len(in_q) and len(lemma_in_c) == len(in_q) and len(tf) == len(in_q)
    assert len(tf) == len(p_q_relation) and len(tf) == len(p_c_relation)
    return {
        'in_q': in_q,
        'in_c': in_c,
        'lemma_in_q': lemma_in_q,
        'lemma_in_c': lemma_in_c,
        'tf': tf,
        'p_q_relation': p_q_relation,
        'p_c_relation': p_c_relation
    }

def get_example(d_id, q_id, c_id, d_dict, q_dict, c_dict, label):
    return {
            'id': d_id + '_' + q_id + '_' + c_id,
            'd_words': ' '.join(d_dict['words']),
            'd_pos': d_dict['pos'],
            'd_ner': d_dict['ner'],
            'q_words': ' '.join(q_dict['words']),
            'q_pos': q_dict['pos'],
            'c_words': ' '.join(c_dict['words']),
            'label': label
        }

def preprocess_dataset(path, is_test_set=False):
    writer = open(path.replace('.json', '') + '-processed.json', 'w', encoding='utf-8')
    ex_cnt = 0
    for obj in json.load(open(path, 'r', encoding='utf-8'))['data']['instance']:
        if not obj['questions']:
            continue
        d_dict = tokenize(obj['text'])
        d_id = path + '_' + obj['@id']
        try:
            qs = [q for q in obj['questions']['question']]
            dummy = qs[0]['@text']
        except:
            # some passages have only one question
            qs = [obj['questions']['question']]
        for q in qs:
            q_dict = tokenize(q['@text'])
            q_id = q['@id']
            for ans in q['answer']:
                c_dict = tokenize(ans['@text'])
                label = int(ans['@correct'].lower() == 'true') if not is_test_set else -1
                c_id = ans['@id']
                example = get_example(d_id, q_id, c_id, d_dict, q_dict, c_dict, label)
                example.update(compute_features(d_dict, q_dict, c_dict))
                writer.write(json.dumps(example))
                writer.write('\n')
                ex_cnt += 1
    print('Found %d examples in %s...' % (ex_cnt, path))
    writer.close()


def _get_race_obj(d):
    for root_d, _, files in os.walk(d):
        for f in files:
            if f.endswith('txt'):
                obj = json.load(open(root_d + '/' + f, 'r', encoding='utf-8'))
                yield obj

def preprocess_race_dataset(d):
    import utils
    utils.build_vocab()
    def is_passage_ok(words):
        return len(words) >= 50 and len(words) <= 500 and sum([int(w in utils.vocab) for w in words]) >= 0.85 * len(words)
    def is_question_ok(words):
        return True
    def is_option_ok(words):
        s = ' '.join(words).lower()
        return s != 'all of the above' and s != 'none of the above'
    writer = open('./data/race-processed.json', 'w', encoding='utf-8')
    ex_cnt = 0
    for obj in _get_race_obj(d):
        d_dict = tokenize(obj['article'].replace('\n', ' ').replace('--', ' '))
        if not is_passage_ok(d_dict['words']):
            continue
        d_id = obj['id']
        assert len(obj['options']) == len(obj['answers']) and len(obj['answers']) == len(obj['questions'])
        q_cnt = 0
        for q, ans, choices in zip(obj['questions'], obj['answers'], obj['options']):
            q_id = str(q_cnt)
            q_cnt += 1
            ans = ord(ans) - ord('A')
            assert 0 <= ans < len(choices)
            q_dict = tokenize(q.replace('_', ' _ '))
            if not is_question_ok(q_dict['words']):
                continue
            for c_id, choice in enumerate(choices):
                c_dict = tokenize(choice)
                if not is_option_ok(c_dict['words']):
                    continue
                label = int(c_id == ans)
                c_id = str(c_id)
                example = get_example(d_id, q_id, c_id, d_dict, q_dict, c_dict, label)
                example.update(compute_features(d_dict, q_dict, c_dict))
                writer.write(json.dumps(example))
                writer.write('\n')
                ex_cnt += 1
    print('Found %d examples in %s...' % (ex_cnt, d))
    writer.close()

def preprocess_conceptnet(path):
    import utils
    utils.build_vocab()
    writer = open('concept.filter', 'w', encoding='utf-8')
    def _get_lan_and_w(arg):
        arg = arg.strip('/').split('/')
        return arg[1], arg[2]
    for line in open(path, 'r', encoding='utf-8'):
        fs = line.split()
        relation, arg1, arg2 = fs[1].split('/')[-1], fs[2], fs[3]
        lan1, w1 = _get_lan_and_w(arg1)
        if lan1 != 'en' or not all(w in utils.vocab for w in w1.split('_')):
            continue
        lan2, w2 = _get_lan_and_w(arg2)
        if lan2 != 'en' or not all(w in utils.vocab for w in w2.split('_')):
            continue
        obj = json.loads(fs[-1])
        if obj['weight'] < 1.0:
            continue
        writer.write('%s %s %s\n' % (relation, w1, w2))
    writer.close()

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'conceptnet':
        preprocess_conceptnet('conceptnet-assertions-5.5.5.csv')
        exit(0)
    init_tokenizer()
    preprocess_dataset('./data/trial-data.json')
    preprocess_dataset('./data/dev-data.json')
    preprocess_dataset('./data/train-data.json')
    preprocess_dataset('./data/test-data.json', is_test_set=True)
    # preprocess_race_dataset('./data/RACE/')
