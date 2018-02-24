from utils import is_stopword

class ConceptNet:

    def __init__(self, path='./data/concept.filter'):
        self.data = {}
        cnt = 0
        for triple in open(path, 'r', encoding='utf-8'):
            r, arg1, arg2 = triple.split()
            if not arg1 in self.data:
                self.data[arg1] = {}
            self.data[arg1][arg2] = r
            if not arg2 in self.data:
                self.data[arg2] = {}
            self.data[arg2][arg1] = r
            cnt += 1
        print('Load %d triples from %s' % (cnt, path))

    def get_relation(self, w1, w2):
        if is_stopword(w1) or is_stopword(w2):
            return '<NULL>'
        w1 = '_'.join(w1.lower().split())
        w2 = '_'.join(w2.lower().split())
        if not w1 in self.data:
            return '<NULL>'
        return self.data[w1].get(w2, '<NULL>')

    def p_q_relation(self, passage, query):
        passage = [w.lower() for w in passage]
        query = [w.lower() for w in query]
        query = set(query) | set([' '.join(query[i:(i+2)]) for i in range(len(query))])
        query = set([q for q in query if not is_stopword(q)])
        ret = ['<NULL>' for _ in passage]
        for i in range(len(passage)):
            for q in query:
                r = self.get_relation(passage[i], q)
                if r != '<NULL>':
                    ret[i] = r
                    break
                r = self.get_relation(' '.join(passage[i:(i+2)]), q)
                if r != '<NULL>':
                    ret[i] = r
                    break
        return ret

concept_net = ConceptNet()
if __name__ == '__main__':
    net = ConceptNet()
    print(net.get_relation('positive', 'negative'))
    print(net.get_relation('under water', 'above water'))
    while True:
        w1, w2 = input('Please input two words: ').strip().split()[:2]
        print(net.get_relation(w1, w2))