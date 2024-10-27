import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): input tensor. shape: (bs, seq_len, n_heads, dim_per_head)

        Returns:
            Tensor: input tensor with rotary embeddings. shape: (bs, seq_len, n_heads, dim_per_head)
        """

        assert x.ndim == 4
        assert x.shape[3] % 2 == 0, "dim_per_head must be divisible by 2"

        freqs = 1.0 / (10000 ** (torch.arange(0, x.shape[3], 2).float() / 6))
        freqs = torch.repeat_interleave(freqs, 2)
        r = freqs * torch.arange(x.shape[1]).float()[:, None]

        r1 = r.cos()
        r2 = r.sin()

        aranged = torch.arange(x.shape[3])

        return (
            x * r1[None, :, None, :]
            + x[
                :,
                :,
                :,
                torch.where(
                    aranged % 2 == 1,
                    aranged - 1,
                    aranged + 1,
                ),
            ]
            * torch.where(aranged % 2 == 0, -1, 1)
            * r2[None, :, None, :]
        )
