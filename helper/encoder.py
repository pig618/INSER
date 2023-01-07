import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import os
import helper.utils as utils


class Encoder(nn.Module):

    def __init__(self, embeddings, args):
        super(Encoder, self).__init__()
        self.args = args
        self.embedding_layer = nn.Embedding.from_pretrained(
            torch.from_numpy(embeddings).float()
        )
        self.embedding_layer.weight.requires_grad = True
        
        self.rnn = nn.LSTM(
            input_size=args.embedding_dim, hidden_size=args.hidden_dim_enc,
            batch_first=True, dropout=args.dropout_enc,
            num_layers=args.num_layers_enc,
            bidirectional=args.gen_bidirectional_enc
        )

        self.hidden = nn.Linear(
            2 * args.hidden_dim * args.embedding_length, args.num_class
        )

    def forward(self, x_indx, mask=None):
        """
        x_indx:  batch of word indices
        mask: Mask to apply over embeddings for tao ratioanles
        """
        x = self.embedding_layer(x_indx)
        if self.args.cuda:
            x = x.cuda()
            
        if not mask is None:
            x = x * mask
        
        activ, _ = self.rnn(x)
        output = activ.reshape(activ.shape[0], -1)
        pred = self.hidden(output)
        
        return pred
