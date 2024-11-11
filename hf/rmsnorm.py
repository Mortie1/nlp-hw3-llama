import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-9):
        super().__init__()

        self.dim = dim
        self.gamma = nn.Parameter(
            data=torch.nn.init.normal_(torch.zeros((dim,))), requires_grad=True
        )
        self.eps = eps

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): input tensor. shape: (bs, seq_len, embed_dim)

        Returns:
            Tensor: input tensor with rotary embeddings. shape: (bs, seq_len, embed_dim)
        """

        assert x.ndim == 3, "input must have 3 dimensions: (bs, seq_len, embed_dim)"

        return (
            x
            / torch.sqrt_(torch.mean(torch.square(x), dim=-1) + self.eps)[:, :, None]
            * self.gamma
        )

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"
