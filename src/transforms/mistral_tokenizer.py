from torch import nn
from transformers import AutoTokenizer


class MistralTokenizer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-v0.1", *args, **kwargs
        )

    def forward(self, text):
        return self.tokenizer(text, return_tensors="pt")

    def __len__(self):
        return len(self.tokenizer)
