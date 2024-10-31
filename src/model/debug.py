import math

import torch
from torch import Tensor, nn
from torch.nn import Sequential
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from xformers.components.attention.utils import maybe_merge_masks

from src.model.llama_layers import LLaMADecoderLayer
from src.transforms import MistralTokenizer


class PositionalEncoding(nn.Module):
    """
    Classic Attention-is-all-you-need positional encoding.
    From PyTorch docs.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(1), :].transpose(0, 1)
        return x


class CasualDecoder(nn.Module):
    def __init__(
        self,
        emb_size: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=emb_size,
            num_heads=n_heads,
            batch_first=True,
        )
        self.rmsnorm1 = nn.RMSNorm(emb_size, eps=1e-9)
        self.rmsnorm2 = nn.RMSNorm(emb_size, eps=1e-9)
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4, bias=False),
            nn.GELU(),
            nn.Linear(emb_size * 4, emb_size, bias=False),
        )
        self.n_heads = n_heads

    def forward(self, in_tuple) -> Tensor:
        """
        Args:
            in_tuple (tuple[Tensor, Tensor, Tensor]): tuple, containing 3 tensors:
                x (Tensor): input tensor    (bs, seq_len, dim)
                attn_mask (Tensor): attention mask  (seq_len, seq_len)
                padding_mask (Tensor): padding mask (bs, seq_len)

        Returns:
            Tensor: output tensor
        """
        assert len(in_tuple) == 3, "input tuple must have 3 elements"
        x, attn_mask, padding_mask = in_tuple
        x_norm = self.rmsnorm1(x)
        attn_output, _ = self.multihead_attn.forward(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=attn_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = attn_output + x
        return self.ffn(self.rmsnorm2(x)) + x, attn_mask, padding_mask


class DebugNet(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        n_layers: int = 16,
        n_heads: int = 16,
        dropout: int = 0.0,
        n_chckpnt_segments: int = 16,
        tokenizer=MistralTokenizer(),
        **kwargs,
    ):
        """
        Args:
            n_feats (int): number of input features.
            n_class (int): number of classes.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        self.tokenizer = tokenizer
        vocab_len = len(tokenizer)
        self.embed = nn.Embedding(
            vocab_len, embed_dim, padding_idx=self.tokenizer.pad_token_id
        )
        self.pos_embed = PositionalEncoding(d_model=embed_dim)
        self.n_heads = n_heads

        self.decoders = nn.Sequential(
            *[
                CasualDecoder(emb_size=embed_dim, n_heads=self.n_heads, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.head = nn.Linear(embed_dim, vocab_len, bias=False)

        # nn.init.xavier_uniform_(self.head.weight)
        # nn.init.xavier_uniform_(self.embed.weight)
        self.rmsnorm = nn.RMSNorm(embed_dim, eps=1e-9)
        self.n_segments = n_chckpnt_segments

    def forward(self, src, attn_mask, pad_mask, **batch):
        """
        Model forward method.

        Args:
            tokenized (Tensor): input text. shape: (batch_size, seq_len)
        Returns:
            output (dict): output dict containing logits.
        """
        x = self.embed(src)
        x = self.pos_embed(x)  # embeds shape: [batch_size, seq_len, embed_dim]
        x, _, _ = checkpoint_sequential(
            self.decoders, self.n_segments, input=(x, attn_mask, pad_mask)
        )
        # for decoder in self.decoders:
        #     x, _, _ = decoder((x, attn_mask, pad_mask))

        logits = self.head(self.rmsnorm(x))
        return {
            "logits": logits.permute(0, 2, 1)
        }  # logits shape: [batch_size, vocab_len, seq_len]

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )
        embedding_parameters = sum([p.numel() for p in self.embed.parameters()])

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"
        result_info = (
            result_info
            + f"\nWithout embedding: {trainable_parameters - embedding_parameters}"
        )

        return result_info
