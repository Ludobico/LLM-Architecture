# Attention is all you need

import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)
    
    def forward(self, values : torch.Tensor, keys : torch.Tensor, query : torch.Tensor, mask : torch.Tensor):
        # size of dimension of the query
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, key_len, self.heads, self.head_dim)
        

        """
        Einstein summation convention 수행
        (n, query_len, heads, head_dim) @ (N, key_len, heads, heads_dim) 와 결과가같음
        두 번째 텐서의 경우 행렬곱셈의 규칙에 따라 transpose(-1,-2)가 선행되어야함
        """
        energy = torch.einsum("nqhd,nkhd -> nhqk", [queries, keys])
        # queries shape : (N, query_len, heads, heads_dim)
        # keys shape : (N, key_len, heads, head_dim)
        # energy shape : (N, heads, query_len, key_len)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        """
        Attension(Q, K, V) = softmax (\frac{QK^T}{\sqrt{d_k}}V)
        여기에서 dim=3는 energy의 4번째차원의 크기 (key_len) 을 의미
        key_len에 softmax를 적용시켜 attention score를 구함
        4번째 차원의 key_len에 attention score가 적용된 shape tensor를 반환, 처음과 크기가 변경되지않음
        (N, heads, query_len, key_len)
        """
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads*self.head_dim
        )
        # Attention shape : (N, heads, query_len, key_len)
        # value shape : (N, value_len, heads, heads_dim)
        # (N, query_len, heads, head_dim)
        # after einsum (N, query_len, heads, head_dim) then flatten last two dimension

        out = self.fc_out(out)
        return out
    


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention =SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )