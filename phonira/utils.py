import glob
import os

import torch
import webdataset as wds
from datasets import load_dataset
from einops import rearrange
from huggingface_hub import get_token


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


def delay_pattern(x: torch.Tensor, padding_value: int = 1025) -> torch.Tensor:
    """Delay pattern.

    Args:
        x (torch.Tensor): the input tensor
        padding_value (int, optional): the padding value. Defaults to 1025.

    Returns:
        torch.Tensor: the delayed pattern tensor
    """

    b, cdbk, n = x.shape
    assert b == 1, "Batch size must be 1, otherwise use stereo delay pattern"

    out = torch.full_like(x, padding_value)

    for k in range(cdbk):
        out[:, k, k:n] = x[:, k, : n - k]

    return out


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths < lengths.unsqueeze(-1)


def reverse_delay_pattern(x: torch.Tensor) -> torch.Tensor:
    """Reverse delay pattern.

    Args:
        x (torch.Tensor): the input tensor

    Returns:
        torch.Tensor: the reversed delay pattern tensor
    """

    b, cdbk, n = x.shape
    assert b == 1, "Batch size must be 1, otherwise use stereo delay pattern"
    assert cdbk <= n, "The codebook size must be less than the sequence size"

    out = torch.full_like(x, 1025)

    for k in range(cdbk):
        out[:, k, : n - k] = x[:, k, k:n]

    return out[:, :, : n - cdbk]


def collate_fn(
    num_quantizers: int,
    column_code: str,
    column_prompt: str,
    conditionning_model,
    tokenizer,
    padding_value: int = 1024,
):
    """Collate function.

    Args:
        num_quantizers (int): the number of quantizers to return
        column_code (str): the column name that contains the codebooks codes of the audio codec
        column_prompt (str): the column name that contains the prompt
        conditionning_model: the conditionning model
        tokenizer: the tokenizer of the conditionning model
        padding_value (int, optional): the padding value. Defaults to 1024.
    """

    @torch.no_grad()
    def _collate_fn(samples):
        # convert the codes to tensors and get the first channel
        codes = [
            torch.tensor(item[column_code], dtype=torch.long)[:1, :, :]
            for item in samples
        ]

        # add padding to the start of the codes like a <sos> token
        sos_token = torch.full_like(codes[0][..., :1], padding_value)

        codes = [torch.cat([sos_token, code], dim=-1) for code in codes]

        # apply the delay pattern and remove the batch dimension (useless in every case because it correspond to the channel
        # so when delay pattern is applied, there is alway only one channel, for stereo and mono)
        codes = [
            delay_pattern(code, padding_value).squeeze(0)[:num_quantizers, :]
            for code in codes
        ]

        # create the padding mask
        lengths = torch.tensor([code.shape[-1] for code in codes])
        padding_maks = make_pad_mask(lengths, max_len=max(lengths))

        codes_stacked = [rearrange(code, "k n -> n k") for code in codes]
        codes_stacked = torch.nn.utils.rnn.pad_sequence(
            codes_stacked, padding_value=padding_value
        )
        codes_stacked = rearrange(codes_stacked, "n b k -> b k n")

        # generate the conditionning embeddings

        sentences = [item[column_prompt] for item in samples]

        inputs = tokenizer(
            [sentence for sentence in sentences],
            return_tensors="pt",
            padding=True,
        )

        output_embeddings = conditionning_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        return (
            codes_stacked,
            padding_maks,
            output_embeddings.last_hidden_state,
            inputs["attention_mask"].bool(),
        )

    return _collate_fn
