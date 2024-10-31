from typing import Optional

import torch
from torch import nn

from src.metrics.base_metric import BaseMetric
from src.transforms import MistralTokenizer


class CrossEntropy(BaseMetric):
    """
    Wrapper over PyTorch CrossEntropyLoss
    """

    def __init__(self, tokenizer=MistralTokenizer(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

    def __call__(self, logits: torch.Tensor, tgt: torch.Tensor, **batch):
        """
        Loss function calculation logic.

        Args:
            logits (Tensor): model output predictions.  shape: (batch_size, vocab_len, seq_len)
            labels (Tensor): ground-truth labels.       shape: (batch_size, seq_len)
        Returns:
            loss: dict containing calculated loss functions.
        """
        return self.loss(logits, tgt)
