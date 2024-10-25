import os.path
import shutil

import numpy as np
import safetensors
from torch import tensor
from tqdm.auto import tqdm
from transformers import AutoTokenizer

import datasets
from src.datasets.base_dataset import BaseDataset
from src.transforms import MistralTokenizer
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class OpenWebText(BaseDataset):
    """
    MNIST dataset

    https://yann.lecun.com/exdb/mnist/
    """

    def __init__(self, split="train", tokenizer=MistralTokenizer(), *args, **kwargs):
        """
        Args:
            split (str): partition name
        """
        index_path = ROOT_PATH / "data" / "openwebtext" / split / "index.json"

        self.tokenizer = tokenizer
        # each nested dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, etc.
        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(split)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, split):
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            split (str): partition name
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        index = []
        data_path = ROOT_PATH / "data" / "openwebtext" / split
        data_path.mkdir(exist_ok=True, parents=True)

        # transform = torchvision.transforms.ToTensor()
        # mnist_data = torchvision.datasets.MNIST(
        #     str(data_path), train=(name == "train"), download=True, transform=transform
        # )
        openwebtext_data = datasets.load_dataset(
            "ashaba1in/small_openwebtext",
            cache_dir=data_path,
            split=split,
            trust_remote_code=True,
        )

        print(f"Parsing OpenWebText (small) Dataset metadata for part {split}...")
        # wrapper over torchvision dataset to get individual objects
        # with some small changes in BaseDataset, torchvision dataset
        # can be used as is without this wrapper
        # but we use wrapper
        for i in tqdm(range(len(openwebtext_data))):
            # create dataset
            save_path = data_path / f"{i:06}.safetensors"
            if not os.path.isfile(save_path):
                text = openwebtext_data[i]["text"]

                save_dict = {"tensor": self.tokenizer(text)["input_ids"]}

                safetensors.torch.save_file(save_dict, save_path)

            # parse dataset metadata and append it to index
            index.append({"path": str(save_path)})

        # shutil.rmtree(data_path / "MNIST")  # remove

        # write index to disk
        write_json(index, str(data_path / "index.json"))

        return index
