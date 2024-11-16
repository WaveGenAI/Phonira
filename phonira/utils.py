import glob
import os

import torch
import webdataset as wds
from datasets import load_dataset


def skip_small_samples(input_key: str, size: int):
    """Skip small samples.

    Args:
        input_key (str): the input key
        size (int): the minimum size of the sample
    """

    def _skip_small_samples(sample):
        if sample[input_key].shape[-1] < size:
            return None
        return sample

    return _skip_small_samples


def load_webdataset(
    dataset_name: str, split_name: str, map_func: callable = None, shuffle: bool = True
) -> wds.WebDataset:
    """Load a webdataset dataset.

    Args:
        dataset_name (str): the name of the dataset or the path to the dataset
        split_name (str): the name of the split
        map_func (callable): the map function. Defaults to None.
        shuffle (bool, optional): shuffle the datase. Defaults to True.

    Returns:
        wds.WebDataset: the webdataset dataset
    """

    if not os.path.exists(dataset_name):
        dataset = load_dataset(dataset_name, streaming=True)
        org, dataset_name = dataset_name.split("/")
        n_shards = dataset[split_name].n_shards

        url = f"https://huggingface.co/datasets/{org}/{dataset_name}/resolve/main/data/{split_name}-{{000000..{n_shards - 1}}}.tar"
        url = f"pipe:curl --connect-timeout 30 --retry 30 --retry-delay 2 -f -s -L {url} -H 'Authorization:Bearer {get_token()}'"
    else:
        n_shards = len(glob.glob(os.path.join(dataset_name, f"{split_name}*.tar")))
        url = os.path.join(dataset_name, f"{split_name}-{{000000..{n_shards - 1}}}.tar")

    if shuffle:
        dataset = wds.DataPipeline(
            wds.SimpleShardList(url),
            wds.detshuffle(),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_stop),
            wds.decode(),
            wds.map(map_func) if map_func else None,
        )
    else:
        dataset = wds.DataPipeline(
            wds.SimpleShardList(url),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_stop),
            wds.decode(),
            wds.map(map_func) if map_func else None,
        )

    return dataset


def collate_fn(num_quantizers: int, column_code: str):
    """Collate function.

    Args:
        num_quantizers (int): the number of quantizers to return
        column_code (str): the column name that contains the codebooks codes of the audio codec
    """

    def _collate_fn(samples):
        codes = [torch.tensor(item[column_code], dtype=torch.long) for item in samples]
        return codes

    return _collate_fn
