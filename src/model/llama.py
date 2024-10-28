from torch import nn
from torch.nn import Sequential
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

from src.model.llama_layers import LLaMADecoderLayer
from src.transforms import MistralTokenizer


class LLaMa(nn.Module):
    """
    Simple MLP
    """

    def __init__(
        self,
        embed_dim: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: int = 0.0,
        n_chckpnt_segments: int = 1,
        tokenizer=MistralTokenizer(),
        **kwargs,
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
        self.embed = nn.Embedding(vocab_len, embed_dim)

        self.decoders = nn.Sequential(
            *[
                LLaMADecoderLayer(emb_size=embed_dim, n_heads=n_heads, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.head = nn.Linear(embed_dim, vocab_len, bias=False)
        self.rmsnorm = nn.RMSNorm(embed_dim, eps=1e-5)
        self.n_segments = n_chckpnt_segments

    def forward(self, src, attn_mask, pad_mask, **batch):
        """
        Model forward method.

        Args:
            tokenized (Tensor): input text. shape: (batch_size, seq_len)
        Returns:
            output (dict): output dict containing logits.
        """
        x = self.embed(src)  # embeds shape: [batch_size, seq_len, embed_dim]
        # x, _, _ = checkpoint_sequential(
        #     self.decoders, self.n_segments, input=(x, attn_mask, pad_mask)
        # )
        for decoder in self.decoders:
            x, _, _ = decoder((x, attn_mask, pad_mask))

        logits = self.head(self.rmsnorm(x))
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
