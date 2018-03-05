import logging
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

from utils import vocab
from doc import batchify
from trian import TriAN

logger = logging.getLogger()

class Model:

    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.finetune_topk = args.finetune_topk
        self.lr = args.lr
        self.use_cuda = (args.use_cuda == True) and torch.cuda.is_available()
        print('Use cuda:', self.use_cuda)
        if self.use_cuda:
            torch.cuda.set_device(int(args.gpu))
        self.network = TriAN(args)
        self.init_optimizer()
        if args.pretrained:
            print('Load pretrained model from %s...' % args.pretrained)
            self.load(args.pretrained)
        else:
            self.load_embeddings(vocab.tokens(), args.embedding_file)
        self.network.register_buffer('fixed_embedding', self.network.embedding.weight.data[self.finetune_topk:].clone())
        if self.use_cuda:
            self.network.cuda()
        print(self.network)
        self._report_num_trainable_parameters()

    def _report_num_trainable_parameters(self):
        num_parameters = 0
        for p in self.network.parameters():
            if p.requires_grad:
                sz = list(p.size())
                if sz[0] == len(vocab):
                    sz[0] = self.finetune_topk
                num_parameters += np.prod(sz)
        print('Number of parameters: ', num_parameters)

    def train(self, train_data):
        self.network.train()
        self.updates = 0
        iter_cnt, num_iter = 0, (len(train_data) + self.batch_size - 1) // self.batch_size
        for batch_input in self._iter_data(train_data):
            feed_input = [x for x in batch_input[:-1]]
            y = batch_input[-1]
            pred_proba = self.network(*feed_input)

            loss = F.binary_cross_entropy(pred_proba, y)
            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm(self.network.parameters(), self.args.grad_clipping)

            # Update parameters
            self.optimizer.step()
            self.network.embedding.weight.data[self.finetune_topk:] = self.network.fixed_embedding
            self.updates += 1
            iter_cnt += 1

            if self.updates % 20 == 0:
                print('Iter: %d/%d, Loss: %f' % (iter_cnt, num_iter, loss.data[0]))
        self.scheduler.step()
        print('LR:', self.scheduler.get_lr()[0])

    def evaluate(self, dev_data, debug=False, eval_train=False):
        if len(dev_data) == 0:
            return -1.0
        self.network.eval()
        correct, total, prediction, gold = 0, 0, [], []
        dev_data = sorted(dev_data, key=lambda ex: ex.id)
        for batch_input in self._iter_data(dev_data):
            feed_input = [x for x in batch_input[:-1]]
            y = batch_input[-1].data.cpu().numpy()
            pred_proba = self.network(*feed_input)
            pred_proba = pred_proba.data.cpu()
            prediction += list(pred_proba)
            gold += [int(label) for label in y]
            assert(len(prediction) == len(gold))

        if eval_train:
            prediction = [1 if p > 0.5 else 0 for p in prediction]
            acc = sum([1 if y1 == y2 else 0 for y1, y2 in zip(prediction, gold)]) / len(gold)
            return acc

        cur_pred, cur_gold, cur_choices = [], [], []
        if debug:
            writer = open('./data/output.log', 'w', encoding='utf-8')
        for i, ex in enumerate(dev_data):
            if i + 1 == len(dev_data):
                cur_pred.append(prediction[i])
                cur_gold.append(gold[i])
                cur_choices.append(ex.choice)
            if (i > 0 and ex.id[:-1] != dev_data[i - 1].id[:-1]) or (i + 1 == len(dev_data)):
                py, gy = np.argmax(cur_pred), np.argmax(cur_gold)
                if debug:
                    writer.write('Passage: %s\n' % dev_data[i - 1].passage)
                    writer.write('Question: %s\n' % dev_data[i - 1].question)
                    for idx, choice in enumerate(cur_choices):
                        writer.write('*' if idx == gy else ' ')
                        writer.write('%s  %f\n' % (choice, cur_pred[idx]))
                    writer.write('\n')
                if py == gy:
                    correct += 1
                total += 1
                cur_pred, cur_gold, cur_choices = [], [], []
            cur_pred.append(prediction[i])
            cur_gold.append(gold[i])
            cur_choices.append(ex.choice)

        acc = 1.0 * correct / total
        if debug:
            writer.write('Accuracy: %f\n' % acc)
            writer.close()
        return acc

    def predict(self, test_data):
        # DO NOT SHUFFLE test_data
        self.network.eval()
        prediction = []
        for batch_input in self._iter_data(test_data):
            feed_input = [x for x in batch_input[:-1]]
            pred_proba = self.network(*feed_input)
            pred_proba = pred_proba.data.cpu()
            prediction += list(pred_proba)
        return prediction

    def _iter_data(self, data):
        num_iter = (len(data) + self.batch_size - 1) // self.batch_size
        for i in range(num_iter):
            start_idx = i * self.batch_size
            batch_data = data[start_idx:(start_idx + self.batch_size)]
            batch_input = batchify(batch_data)

            # Transfer to GPU
            if self.use_cuda:
                batch_input = [Variable(x.cuda(async=True)) for x in batch_input]
            else:
                batch_input = [Variable(x) for x in batch_input]
            yield batch_input

    def load_embeddings(self, words, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.

        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        words = {w for w in words if w in vocab}
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))
        embedding = self.network.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(embedding_file) as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == embedding.size(1) + 1)
                w = vocab.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[vocab[w]].copy_(vec)
                    else:
                        logging.warning('WARN: Duplicate embedding found for %s' % w)
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[vocab[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[vocab[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def init_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.lr,
                                       momentum=0.4,
                                       weight_decay=0)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                        lr=self.lr,
                                        weight_decay=0)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 15], gamma=0.5)

    def save(self, ckt_path):
        state_dict = copy.copy(self.network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {'state_dict': state_dict}
        torch.save(params, ckt_path)

    def load(self, ckt_path):
        logger.info('Loading model %s' % ckt_path)
        saved_params = torch.load(ckt_path, map_location=lambda storage, loc: storage)
        state_dict = saved_params['state_dict']
        return self.network.load_state_dict(state_dict)

    def cuda(self):
        self.use_cuda = True
        self.network.cuda()