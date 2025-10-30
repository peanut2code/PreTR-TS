import torch.nn as nn
import torch
from .transformer import TransformerBlock
from .embedding import BERTEmbedding

# print('1')
class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, spots_len=603, size = 25, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4
        # input()
        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(input_size = size, embed_size=hidden, spots_len=spots_len)
        #self.embedding_5 = BERTEmbedding(input_size = 5, embed_size=hidden, spots_len=spots_len)
        # print(self.embedding)
        # input()
        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info, label_bool):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (label_bool > 0).unsqueeze(1).repeat(1, label_bool.size(1), 1).unsqueeze(1)
        #print('mask:', mask.shape)
        # embedding the indexed sequence to sequence of vectors
        # print('segment_info:', segment_info)
        # print('segment_info shape:', segment_info.shape)
        '''
        if x.shape[2] > 5:
            x = self.embedding(x, segment_info)
        else:
            x = self.embedding_5(x, segment_info)
        '''
        x = self.embedding(x, segment_info)
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        '''
        h = 0
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
            if h == 0:
                avg_x = x
            else:
                avg_x = (avg_x + x)/2
        '''
        return x
