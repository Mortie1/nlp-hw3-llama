from torch import nn
from torch.nn import Sequential

from src.transforms import MistralTokenizer


class BaselineModel(nn.Module):
    """
    Simple MLP
    """

    def __init__(
        self, embed_dim=1024, fc_hidden=2048, tokenizer=MistralTokenizer(), **kwargs
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
        self.embeds = nn.Embedding(vocab_len, embed_dim)

    def forward(self, text, **batch):
        """
        Model forward method.

        Args:
            img (Tensor): input img.
        Returns:
            output (dict): output dict containing logits.
        """
        embeds = self.embeds(text)
        return {"logits": self.net(embeds)}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
