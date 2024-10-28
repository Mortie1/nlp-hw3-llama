import torch
from torch import nn

from src.transforms import MistralTokenizer


class CrossEntropyLoss(nn.Module):
    """
    Wrapper over PyTorch CrossEntropyLoss
    """

    def __init__(self, tokenizer=MistralTokenizer()):
        super().__init__()
        self.tokenizer = tokenizer
        self.loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

    def forward(self, logits: torch.Tensor, tgt: torch.Tensor, **batch):
        """
        Loss function calculation logic.

        Args:
            logits (Tensor): model output predictions.  shape: (batch_size, vocab_len, seq_len)
            labels (Tensor): ground-truth labels.       shape: (batch_size, seq_len)
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        return {"loss": self.loss(logits, tgt)}
