import os
import argparse
import logging

logger = logging.getLogger(__name__)

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')

parser = argparse.ArgumentParser()
parser.register('type', 'bool', str2bool)
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--epoch', type=int, default=50, help='Number of epoches to run')
parser.add_argument('--optimizer', type=str, default='adamax', help='optimizer, adamax or sgd')
parser.add_argument('--use_cuda', type='bool', default=True, help='use cuda or not')
parser.add_argument('--grad_clipping', type=float, default=10.0, help='maximum L2 norm for gradient clipping')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--embedding_file', type=str, default='./data/glove.840B.300d.txt', help='embedding file')
parser.add_argument('--hidden_size', type=int, default=96, help='default size for RNN layer')
parser.add_argument('--doc_layers', type=int, default=1, help='number of RNN layers for doc encoding')
parser.add_argument('--rnn_type', type=str, default='lstm', help='RNN type, lstm or gru')
parser.add_argument('--dropout_rnn_output', type=float, default=0.4, help='dropout for RNN output')
parser.add_argument('--rnn_padding', type='bool', default=True, help='Use padding or not')
parser.add_argument('--dropout_emb', type=float, default=0.4, help='dropout rate for embeddings')
parser.add_argument('--pretrained', type=str, default='', help='pretrained model path')
parser.add_argument('--finetune_topk', type=int, default=10, help='Finetune topk embeddings during training')
parser.add_argument('--pos_emb_dim', type=int, default=12, help='Embedding dimension for part-of-speech')
parser.add_argument('--ner_emb_dim', type=int, default=8, help='Embedding dimension for named entities')
parser.add_argument('--rel_emb_dim', type=int, default=10, help='Embedding dimension for ConceptNet relations')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--test_mode', type='bool', default=False, help='In test mode, validation data will be used for training')
args = parser.parse_args()

print(args)

if args.pretrained:
    assert all(os.path.exists(p) for p in args.pretrained.split(',')), 'Checkpoint %s does not exist.' % args.pretrained
