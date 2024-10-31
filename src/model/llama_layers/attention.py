import torch
from torch import Tensor, nn

from src.model.llama_layers.rope import RotaryEmbedding


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        emb_size: int,
        n_heads: int,
        dropout: float = 0.0,
        use_rotary_embeddings: bool = False,
        bias_qkv: bool = False,
        bias_out: bool = False,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.n_heads = n_heads
        assert (
            self.emb_size % n_heads == 0
        ), "Embedding size needs to be divisible by heads"

        self.head_dim = emb_size // n_heads

        self.use_rotary_embeddings = use_rotary_embeddings
        if self.use_rotary_embeddings:
            self.rotary_embed = RotaryEmbedding(self.head_dim)

        self.qkv = nn.Linear(emb_size, emb_size * 3, bias=bias_qkv)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(emb_size, emb_size, bias=bias_out)

        self.scaling = self.head_dim**-0.5

    def forward(self, x: Tensor, att_mask: Tensor = None):
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.reshape(x.shape[0], -1, self.n_heads, self.head_dim).transpose(
                1, 2
            ),
            qkv,
        )  # [batch_size, n_heads, seq_len, head_dim]

        if self.use_rotary_embeddings:
            q, k = self.rotary_embed(q), self.rotary_embed(k)

        dots = (
            torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        )  # [batch_size, n_heads, seq_len, seq_len]

        if att_mask is not None:
            dots = dots + att_mask

        attn = self.dropout(torch.softmax(dots, dim=-1))
        out = (
            torch.matmul(attn, v).transpose(1, 2).reshape(x.shape[0], -1, self.emb_size)
        )
        out = self.out(out)

        return out
