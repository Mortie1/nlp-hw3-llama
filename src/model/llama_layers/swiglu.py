import torch
from torch import Tensor, nn


class SwiGLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear_inp1 = nn.Linear(dim, (8 * dim) // 3, bias=False)
        self.linear_inp2 = nn.Linear(dim, (8 * dim) // 3, bias=False)
        self.linear_out = nn.Linear((8 * dim) // 3, dim, bias=False)
        self.silu = nn.SiLU(inplace=True)

        nn.init.xavier_uniform_(self.linear_inp1.weight)
        nn.init.xavier_uniform_(self.linear_inp2.weight)
        nn.init.xavier_uniform_(self.linear_out.weight)

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): input tensor
        """
        return self.linear_out(self.silu(self.linear_inp1(x)) * self.linear_inp2(x))
