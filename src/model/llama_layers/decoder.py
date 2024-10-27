import torch
from torch import Tensor, nn
from xformers.components import MultiHeadDispatch
from xformers.components.attention import ScaledDotProduct
from xformers.components.attention.utils import maybe_merge_masks

from src.model.llama_layers import SwiGLU


class LLaMADecoderLayer(nn.Module):
    def __init__(
        self,
        emb_size: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.multihead_attn = MultiHeadDispatch(
            dim_model=emb_size,
            num_heads=n_heads,
            attention=ScaledDotProduct(
                dropout=dropout,
            ),
            bias=(False, False, False, False),
            use_rotary_embeddings=True,
        )
        self.rmsnorm = nn.RMSNorm(emb_size)
        self.swiglu = SwiGLU(emb_size)
        self.n_heads = n_heads

    def forward(self, x: Tensor, attn_mask: Tensor, padding_mask: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor    (bs, seq_len, dim)
            attn_mask (Tensor): attention mask  (seq_len, seq_len)
            padding_mask (Tensor): padding mask (bs, seq_len)

        Returns:
            Tensor: output tensor
        """
        mask = maybe_merge_masks(
            attn_mask, padding_mask, x.shape[0], x.shape[1], self.n_heads
        )

        x = self.rmsnorm(self.multihead_attn(x, att_mask=mask) + x)
        return self.rmsnorm(self.swiglu(x) + x)
