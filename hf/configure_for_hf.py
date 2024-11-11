from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)

from hf.llama import CustomAttentionLLaMa


class MyLLaMaConfig(PretrainedConfig):
    model_type = "LLaMa"

    def __init__(
        self,
        embed_dim: int = 1536,
        n_layers: int = 24,
        n_heads: int = 24,
        n_chckpnt_segments: int = 24,
        **kwargs,
    ):
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_chckpnt_segments = n_chckpnt_segments
        super().__init__(**kwargs)


class MyLLaMa(PreTrainedModel):
    config_class = MyLLaMaConfig

    def __init__(self, config: MyLLaMaConfig):
        super().__init__(config)
        self.model = CustomAttentionLLaMa(
            config.embed_dim,
            config.n_layers,
            config.n_heads,
            dropout=0,
            n_chckpnt_segments=config.n_chckpnt_segments,
        )

    def forward(self, tensor, labels=None):
        logits = self.model(tensor)["logits"]
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


AutoConfig.register("LLaMa", MyLLaMaConfig)
AutoModel.register(MyLLaMaConfig, MyLLaMa)
AutoModelForCausalLM.register(MyLLaMaConfig, MyLLaMa)
