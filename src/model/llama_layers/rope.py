import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim_per_head: int,
        max_seq_len: int = 4096,
        device=None,
        dtype=None,
    ):
        super().__init__()

        freqs = 1.0 / (
            10000
            ** (
                torch.arange(0, dim_per_head, 2, device=device, dtype=dtype).float() / 6
            )
        )
        freqs = torch.repeat_interleave(freqs, 2)

        r = (
            freqs
            * torch.arange(max_seq_len, device=device, dtype=dtype).float()[:, None]
        )
        r1 = r.cos()
        self.register_buffer("r1", r1)

        r2 = r.sin()
        self.register_buffer("r2", r2)

        aranged = torch.arange(dim_per_head, device=device, dtype=dtype)

        mask1 = torch.where(
            aranged % 2 == 1,
            aranged - 1,
            aranged + 1,
        )
        self.register_buffer("mask1", mask1)

        mask2 = torch.where(aranged % 2 == 0, -1, 1)
        self.register_buffer("mask2", mask2)

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): input tensor. shape: (bs, seq_len, n_heads, dim_per_head)

        Returns:
            Tensor: input tensor with rotary embeddings. shape: (bs, seq_len, n_heads, dim_per_head)
        """

        assert (
            x.ndim == 4
        ), "input must have 4 dimensions: (bs, n_heads, seq_len, dim_per_head)"
        assert x.shape[3] % 2 == 0, "dim_per_head must be divisible by 2"

        x = x.transpose(1, 2)

        return (
            x * self.r1[None, : x.shape[1], None, :]
            + x[
                :,
                :,
                :,
                self.mask1,
            ]
            * self.mask2
            * self.r2[None, : x.shape[1], None, :]
        ).transpose(1, 2)
