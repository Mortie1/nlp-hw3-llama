import json
import shutil

import numpy as np
import torchvision
from tqdm.auto import tqdm

import datasets
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class OpenWebText(BaseDataset):
    """
    MNIST dataset

    https://yann.lecun.com/exdb/mnist/
    """

    def __init__(self, name="train", *args, **kwargs):
        """
        Args:
            name (str): partition name
        """
        index_path = ROOT_PATH / "data" / "openwebtext" / name / "index.json"

        # each nested dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, etc.
        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(name)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, name):
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            name (str): partition name
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        index = []
        data_path = ROOT_PATH / "data" / "openwebtext" / name
        data_path.mkdir(exist_ok=True, parents=True)

        # transform = torchvision.transforms.ToTensor()
        # mnist_data = torchvision.datasets.MNIST(
        #     str(data_path), train=(name == "train"), download=True, transform=transform
        # )
        openwebtext_data = datasets.load_dataset(
            "ashaba1in/small_openwebtext",
            cache_dir=data_path,
            trust_remote_code=True,
        )

        print(f"Parsing small_openwebtext Dataset metadata for part {name}...")
        # wrapper over torchvision dataset to get individual objects
        # with some small changes in BaseDataset, torchvision dataset
        # can be used as is without this wrapper
        # but we use wrapper
        for i in tqdm(range(len(openwebtext_data))):
            # create dataset
            text = openwebtext_data[i]

            save_dict = {"text": text}
            save_path = data_path / f"{i:06}.txt"
            # safetensors.torch.save_file(save_dict, save_path)
            with open(save_path, "w") as f:
                json.dump(save_dict, f)

            # parse dataset metadata and append it to index
            index.append({"path": str(save_path)})

        shutil.rmtree(data_path / "MNIST")  # remove

        # write index to disk
        write_json(index, str(data_path / "index.json"))

        return index
