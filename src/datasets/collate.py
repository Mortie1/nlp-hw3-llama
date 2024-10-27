import torch
from torch.nn.utils.rnn import pad_sequence

from src.transforms import MistralTokenizer

tokenizer = MistralTokenizer()


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    result_batch["text"] = [item["text"] for item in dataset_items]
    tokenizer_output = tokenizer([item["text"] for item in dataset_items])[
        "input_ids"
    ]  # shape: (batch_size, seq_len)
    result_batch["src"] = tokenizer_output[:, :-1]
    result_batch["tgt"] = tokenizer_output[:, 1:]

    attention_mask = (
        torch.triu(
            torch.ones((result_batch["src"].shape[1], result_batch["src"].shape[1]))
        )
        == 1
    ).transpose(0, 1)
    result_batch["attn_mask"] = attention_mask

    padding_mask = torch.where(
        result_batch["src"] == tokenizer.pad_token_id, False, True
    )
    result_batch["pad_mask"] = padding_mask

    return result_batch
