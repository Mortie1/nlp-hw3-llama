from torch import nn
from transformers import AutoTokenizer


class MistralTokenizer(nn.Module):
    def __init__(self, max_length=1024, *args, **kwargs):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-v0.1", *args, **kwargs
        )
        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.special_tokens_ids = {
            token: id
            for token, id in zip(
                self.tokenizer.special_tokens_map.keys(), self.tokenizer.all_special_ids
            )
        }
        self.max_length = max_length
        self.pad_token_id = self.tokenizer.pad_token_id

    def forward(self, text):
        return self.tokenizer(
            text,
            return_tensors="pt",
            return_attention_mask=False,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            padding_side="right",
        )

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)

    def decode(self, x):
        return self.tokenizer.batch_decode(x)

    def __len__(self):
        return len(self.tokenizer)
