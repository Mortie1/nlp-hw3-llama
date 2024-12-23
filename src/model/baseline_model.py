from torch import nn
from torch.nn import Sequential

from src.transforms import MistralTokenizer


class BaselineModel(nn.Module):
    """
    Simple MLP
    """

    def __init__(
        self, embed_dim=512, fc_hidden=1024, tokenizer=MistralTokenizer(), **kwargs
    ):
        """
        Args:
            n_feats (int): number of input features.
            n_class (int): number of classes.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        self.tokenizer = tokenizer
        vocab_len = len(tokenizer)

        self.net = Sequential(
            # people say it can approximate any function...
            nn.Linear(in_features=embed_dim, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=vocab_len),
        )
        self.embed = nn.Embedding(vocab_len, embed_dim)

    def forward(self, tokenized, **batch):
        """
        Model forward method.

        Args:
            tokenized (Tensor): input text. shape: (batch_size, seq_len)
        Returns:
            output (dict): output dict containing logits.
        """
        print(type(tokenized))
        print(tokenized)
        print(tokenized.shape)
        embeds = self.embed(tokenized)  # embeds shape: [batch_size, seq_len, embed_dim]
        logits = self.net(embeds)  # logits shape: [batch_size, seq_len, vocab_len]
        return {
            "logits": logits.permute(0, 2, 1)
        }  # logits shape: [batch_size, vocab_len, seq_len]

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )
        embedding_parameters = sum([p.numel() for p in self.embed.parameters()])

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"
        result_info = (
            result_info
            + f"\nWithout embedding: {trainable_parameters - embedding_parameters}"
        )

        return result_info
