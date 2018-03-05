import torch
import torch.nn as nn
import torch.nn.functional as F

import layers
from utils import vocab, pos_vocab, ner_vocab, rel_vocab

class TriAN(nn.Module):

    def __init__(self, args):
        super(TriAN, self).__init__()
        self.args = args
        self.embedding_dim = 300
        self.embedding = nn.Embedding(len(vocab), self.embedding_dim, padding_idx=0)
        self.embedding.weight.data.fill_(0)
        self.embedding.weight.data[:2].normal_(0, 0.1)
        self.pos_embedding = nn.Embedding(len(pos_vocab), args.pos_emb_dim, padding_idx=0)
        self.pos_embedding.weight.data.normal_(0, 0.1)
        self.ner_embedding = nn.Embedding(len(ner_vocab), args.ner_emb_dim, padding_idx=0)
        self.ner_embedding.weight.data.normal_(0, 0.1)
        self.rel_embedding = nn.Embedding(len(rel_vocab), args.rel_emb_dim, padding_idx=0)
        self.rel_embedding.weight.data.normal_(0, 0.1)
        self.RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU}

        self.p_q_emb_match = layers.SeqAttnMatch(self.embedding_dim)
        self.c_q_emb_match = layers.SeqAttnMatch(self.embedding_dim)
        self.c_p_emb_match = layers.SeqAttnMatch(self.embedding_dim)

        # Input size to RNN: word emb + question emb + pos emb + ner emb + manual features
        doc_input_size = 2 * self.embedding_dim + args.pos_emb_dim + args.ner_emb_dim + 5 + 2 * args.rel_emb_dim

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers,
            dropout_rate=0,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding)

        # RNN question encoder: word emb + pos emb
        qst_input_size = self.embedding_dim + args.pos_emb_dim
        self.question_rnn = layers.StackedBRNN(
            input_size=qst_input_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            dropout_rate=0,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding)

        # RNN answer encoder
        choice_input_size = 3 * self.embedding_dim
        self.choice_rnn = layers.StackedBRNN(
            input_size=choice_input_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            dropout_rate=0,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding)

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        choice_hidden_size = 2 * args.hidden_size

        # Answer merging
        self.c_self_attn = layers.LinearSeqAttn(choice_hidden_size)
        self.q_self_attn = layers.LinearSeqAttn(question_hidden_size)

        self.p_q_attn = layers.BilinearSeqAttn(x_size=doc_hidden_size, y_size=question_hidden_size)

        self.p_c_bilinear = nn.Linear(doc_hidden_size, choice_hidden_size)
        self.q_c_bilinear = nn.Linear(question_hidden_size, choice_hidden_size)

    def forward(self, p, p_pos, p_ner, p_mask, q, q_pos, q_mask, c, c_mask, f_tensor, p_q_relation, p_c_relation):
        p_emb, q_emb, c_emb = self.embedding(p), self.embedding(q), self.embedding(c)
        p_pos_emb, p_ner_emb, q_pos_emb = self.pos_embedding(p_pos), self.ner_embedding(p_ner), self.pos_embedding(q_pos)
        p_q_rel_emb, p_c_rel_emb = self.rel_embedding(p_q_relation), self.rel_embedding(p_c_relation)

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            p_emb = nn.functional.dropout(p_emb, p=self.args.dropout_emb, training=self.training)
            q_emb = nn.functional.dropout(q_emb, p=self.args.dropout_emb, training=self.training)
            c_emb = nn.functional.dropout(c_emb, p=self.args.dropout_emb, training=self.training)
            p_pos_emb = nn.functional.dropout(p_pos_emb, p=self.args.dropout_emb, training=self.training)
            p_ner_emb = nn.functional.dropout(p_ner_emb, p=self.args.dropout_emb, training=self.training)
            q_pos_emb = nn.functional.dropout(q_pos_emb, p=self.args.dropout_emb, training=self.training)
            p_q_rel_emb = nn.functional.dropout(p_q_rel_emb, p=self.args.dropout_emb, training=self.training)
            p_c_rel_emb = nn.functional.dropout(p_c_rel_emb, p=self.args.dropout_emb, training=self.training)

        p_q_weighted_emb = self.p_q_emb_match(p_emb, q_emb, q_mask)
        c_q_weighted_emb = self.c_q_emb_match(c_emb, q_emb, q_mask)
        c_p_weighted_emb = self.c_p_emb_match(c_emb, p_emb, p_mask)
        p_q_weighted_emb = nn.functional.dropout(p_q_weighted_emb, p=self.args.dropout_emb, training=self.training)
        c_q_weighted_emb = nn.functional.dropout(c_q_weighted_emb, p=self.args.dropout_emb, training=self.training)
        c_p_weighted_emb = nn.functional.dropout(c_p_weighted_emb, p=self.args.dropout_emb, training=self.training)
        # print('p_q_weighted_emb', p_q_weighted_emb.size())

        p_rnn_input = torch.cat([p_emb, p_q_weighted_emb, p_pos_emb, p_ner_emb, f_tensor, p_q_rel_emb, p_c_rel_emb], dim=2)
        c_rnn_input = torch.cat([c_emb, c_q_weighted_emb, c_p_weighted_emb], dim=2)
        q_rnn_input = torch.cat([q_emb, q_pos_emb], dim=2)
        # print('p_rnn_input', p_rnn_input.size())

        p_hiddens = self.doc_rnn(p_rnn_input, p_mask)
        c_hiddens = self.choice_rnn(c_rnn_input, c_mask)
        q_hiddens = self.question_rnn(q_rnn_input, q_mask)
        # print('p_hiddens', p_hiddens.size())

        q_merge_weights = self.q_self_attn(q_hiddens, q_mask)
        q_hidden = layers.weighted_avg(q_hiddens, q_merge_weights)

        p_merge_weights = self.p_q_attn(p_hiddens, q_hidden, p_mask)
        # [batch_size, 2*hidden_size]
        p_hidden = layers.weighted_avg(p_hiddens, p_merge_weights)
        # print('p_hidden', p_hidden.size())

        c_merge_weights = self.c_self_attn(c_hiddens, c_mask)
        # [batch_size, 2*hidden_size]
        c_hidden = layers.weighted_avg(c_hiddens, c_merge_weights)
        # print('c_hidden', c_hidden.size())

        logits = torch.sum(self.p_c_bilinear(p_hidden) * c_hidden, dim=-1)
        logits += torch.sum(self.q_c_bilinear(q_hidden) * c_hidden, dim=-1)
        proba = F.sigmoid(logits)
        # print('proba', proba.size())

        return proba
