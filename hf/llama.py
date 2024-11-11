from torch import Tensor, nn
from torch.nn import Sequential
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from xformers.components.attention.utils import maybe_merge_masks

from hf.decoder import CustomAttentionLLaMaDecoder
from hf.rmsnorm import RMSNorm
from hf.tokenizer import MistralTokenizer


class LLaMaBase(nn.Module):
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
        self.vocab_len = len(tokenizer)
        self.n_heads = n_heads
        self.dropout = dropout
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.n_segments = n_chckpnt_segments

        self.embed = nn.Embedding(
            self.vocab_len, embed_dim, padding_idx=self.tokenizer.pad_token_id
        )
        self.head = nn.Linear(embed_dim, self.vocab_len, bias=False)

    def forward(self, src: Tensor, attn_mask: Tensor, pad_mask: Tensor, **batch):
        """
        Model forward method.

        Args:
            tokenized (Tensor): input text. shape: (batch_size, seq_len)
        Returns:
            output (dict): output dict containing logits.
        """

        raise NotImplementedError

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


class CustomAttentionLLaMa(LLaMaBase):
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
        super().__init__(
            embed_dim,
            n_layers,
            n_heads,
            dropout,
            n_chckpnt_segments,
            tokenizer,
        )

        self.decoders = nn.Sequential(
            *[
                CustomAttentionLLaMaDecoder(
                    emb_size=embed_dim, n_heads=self.n_heads, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        self.rmsnorm = RMSNorm(embed_dim, eps=1e-9)

    def forward(self, src: Tensor, attn_mask: Tensor, pad_mask: Tensor, **batch):
        """
        Model forward method.

        Args:
            tokenized (Tensor): input text. shape: (batch_size, seq_len)
        Returns:
            output (dict): output dict containing logits.
        """
        x = self.embed(src)  # embeds shape: [batch_size, seq_len, embed_dim]
        sizes = x.shape
        mask = maybe_merge_masks(
            attn_mask, pad_mask, sizes[0], sizes[1], self.n_heads
        ).view(x.shape[0], self.n_heads, sizes[1], sizes[1])
        x, _ = checkpoint_sequential(self.decoders, self.n_segments, input=(x, mask))
        # for decoder in self.decoders:
        #     x, _, _ = decoder((x, attn_mask, pad_mask))

        logits = self.head(self.rmsnorm(x))
        return {
            "logits": logits.permute(0, 2, 1)
        }  # logits shape: [batch_size, vocab_len, seq_len]
