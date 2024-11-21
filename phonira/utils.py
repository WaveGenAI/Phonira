import glob
import os

import torch
import torch.nn.functional as F
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


def delay_mask(x: torch.Tensor, padding_value: int, start_pos: int = 0):
    """
    Pad the tensor to follow the delay pattern
    1, 1, 1, 1
    1, 1, 1, 1
    1, 1, 1, 1
    to
    0, 1, 1, 1
    0, 1, 1, 1
    1, 1, 1, 1
    """
    b, k, n = x.shape

    for i in range(k):
        x[:, : (k - i) - 1, min(i + start_pos, n - 1)] = padding_value

    return x


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


def collate_fn(
    num_quantizers: int,
    column_code: str,
    column_prompt: str,
    conditionning_model,
    tokenizer,
    delay_pattern,
    padding_value: int = 1024,
    max_length: int = 512,
):
    """Collate function.

    Args:
        num_quantizers (int): the number of quantizers to return
        column_code (str): the column name that contains the codebooks codes of the audio codec
        column_prompt (str): the column name that contains the prompt
        conditionning_model: the conditionning model
        tokenizer: the tokenizer of the conditionning model
        delay_pattern : the delay pattern class
        padding_value (int, optional): the padding value. Defaults to 1024.
        max_length (int, optional): the maximum length. Defaults to 512.
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
            delay_pattern.apply_pattern(code).squeeze(0)[:num_quantizers, :]
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
            truncation=True,
            max_length=max_length,
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


@torch.no_grad()
def inference(
    model,
    conditionning_model,
    tokenizer,
    delay_pattern,
    prompt: str,
    num_quantizers: int,
    num_gen: int,
    padding_value: int = 1024,
    temperature: int = 1.0,
    top_k: int = 150,
):
    assert (
        num_gen >= num_quantizers
    ), "num_gen must be greater or equal to num_quantizers"

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    prepend_embed = conditionning_model(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
    ).last_hidden_state

    prepend_mask = inputs["attention_mask"].bool()

    x = torch.fill_(torch.empty(1, num_quantizers, 1, dtype=torch.long), padding_value)
    for _ in range(num_gen):
        x = delay_pattern.apply_pattern(x, start_pos=1)

        padding_mask = torch.ones_like(x[:, 0, :]).bool()
        out = model(x, prepend_embed, padding_mask, prepend_mask)[0]
        out = out[:, :, -1, :]

        out = out / temperature
        out = F.softmax(out, dim=-1)

        topk_tokens, indices = out.topk(top_k, dim=-1)
        topk_tokens = topk_tokens.view(-1, top_k)

        samples = torch.multinomial(topk_tokens, 1).unsqueeze(0)
        indices = torch.gather(indices, -1, samples)

        x = torch.cat([x, indices], dim=-1)
        x = delay_pattern.reverse_pattern(x, start_pos=1)

    return x[..., 1 : x.size(-1) - num_quantizers + 1]
