import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn
from xformers.components import MultiHeadDispatch
from xformers.components.attention import ScaledDotProduct

from src.model.llama_layers.swiglu import SwiGLU


class LLaMADecoderLayer(nn.Module, PyTorchModelHubMixin):
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
        self.rmsnorm1 = nn.RMSNorm(emb_size, eps=1e-9)
        self.rmsnorm2 = nn.RMSNorm(emb_size, eps=1e-9)
        self.swiglu = SwiGLU(emb_size)
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
        assert len(in_tuple) == 2, "input tuple must have 2 elements"
        x, mask = in_tuple

        x = self.multihead_attn(self.rmsnorm1(x), att_mask=mask) + x
        return self.swiglu(self.rmsnorm2(x)) + x, mask
