import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, embeddings, args):
        super(Generator, self).__init__()
        self.embedding_layer = nn.Embedding.from_pretrained(
            torch.from_numpy(embeddings).float()
        )
        self.embedding_layer.weight.requires_grad = False
        self.args = args
        
        self.rnn = nn.LSTM(
            input_size=args.embedding_dim, 
            hidden_size=args.hidden_dim, batch_first=True, 
            dropout=args.dropout, num_layers=args.num_layers,
            bidirectional=args.gen_bidirectional
        )
        
        # Set forget gate bias to 1
        for names in self.rnn._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.rnn, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)
        
        self.z_dim = 2
        self.hidden = nn.Linear(2 * args.hidden_dim, self.z_dim)
        
    def forward(self, x_indx, input_mask):
        '''
        Given input x_indx of dim (batch, length), 
        return z (batch, length) such that z
        can act as element-wise mask on x
        '''
        x = self.embedding_layer(x_indx)
        if self.args["cuda"]:
            x = x.cuda()
        
        activ, _ = self.rnn(x)
        logits = self.hidden(activ)
        
        mask = self._independent_straight_through_sampling(logits)
        
        return torch.minimum(mask[:, :, 1:2], input_mask.unsqueeze(2))
    
    @staticmethod
    def gumbel_softmax(logits, temperature):
        # g = -log(-log(u)), u ~ U(0, 1)

        noise = torch.rand_like(logits)

        noise.add_(1e-9).log_().neg_()
        noise.add_(1e-9).log_().neg_()

        # # uncomment this for GPU use
        # if torch.cuda.is_available():
        #     noise = noise.cuda()
            
        x = (logits + noise) / temperature
        x = F.softmax(x.view(-1, x.size()[-1]), dim=-1)
        
        # probs: [batch_size, max_steps, 2]
        probs = x.view_as(logits)

        return probs
    
    def _independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            mask -- shape (batch_size, sequence_length, 2)        
        """
        mask = self.gumbel_softmax(rationale_logits, self.args.temperature)
        max_val, _ = torch.max(mask, -1, keepdim=True)
        mask_hard = torch.eq(mask, max_val).type(mask.dtype)
        mask = (mask_hard - mask).clone().detach().requires_grad_(False) + mask

        return mask
