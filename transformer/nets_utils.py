# -*- coding: utf-8 -*-

"""Network related utility tools."""

import logging
from typing import Dict

import numpy as np
import torch


def to_device(m, x):
    """Send tensor into the device of the module.
    Args:
        m (torch.nn.Module): Torch module.
        x (Tensor): Torch tensor.
    Returns:
        Tensor: Torch tensor located in the same place as torch module.
    """
    if isinstance(m, torch.nn.Module):
        device = next(m.parameters()).device
    elif isinstance(m, torch.Tensor):
        device = m.device
    else:
        raise TypeError("Expected torch.nn.Module or torch.tensor, " f"bot got: {type(m)}")
    return x.to(device)


def to_cuda(m, x):
    assert isinstance(m, torch.nn.Module)
    device_id = torch.cuda.device_of(next(m.parameters()).data).idx
    if device_id == -1:
        return x
    return x.cuda(device_id)


def pad_list(xs, pad_value):
    """Perform padding for the list of tensors.
    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.
    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).
    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])
    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad


def make_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of padded part.
    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.
    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)
    Examples:
        With only lengths.
        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
        With the reference tensor.
        >>> xs = torch.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)
        With the reference tensor and dimension indicator.
        >>> xs = torch.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)
    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(slice(None) if i in (0, length_dim) else None for i in range(xs.dim()))
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of non-padded part.
    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.
    Returns:
        ByteTensor: mask tensor containing indices of padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)
    Examples:
        With only lengths.
        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
        With the reference tensor.
        >>> xs = torch.zeros((3, 2, 4))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1],
                 [1, 1, 1, 1]],
                [[1, 1, 1, 0],
                 [1, 1, 1, 0]],
                [[1, 1, 0, 0],
                 [1, 1, 0, 0]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)
        With the reference tensor and dimension indicator.
        >>> xs = torch.zeros((3, 6, 6))
        >>> make_non_pad_mask(lengths, xs, 1)
        tensor([[[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)
        >>> make_non_pad_mask(lengths, xs, 2)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)
    """
    return ~make_pad_mask(lengths, xs, length_dim)


def mask_by_length(xs, lengths, fill=0):
    """Mask tensor according to length.
    Args:
        xs (Tensor): Batch of input tensor (B, `*`).
        lengths (LongTensor or List): Batch of lengths (B,).
        fill (int or float): Value to fill masked part.
    Returns:
        Tensor: Batch of masked input tensor (B, `*`).
    Examples:
        >>> x = torch.arange(5).repeat(3, 1) + 1
        >>> x
        tensor([[1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5]])
        >>> lengths = [5, 3, 2]
        >>> mask_by_length(x, lengths)
        tensor([[1, 2, 3, 4, 5],
                [1, 2, 3, 0, 0],
                [1, 2, 0, 0, 0]])
    """
    assert xs.size(0) == len(lengths)
    ret = xs.data.new(*xs.size()).fill_(fill)
    for i, l in enumerate(lengths):
        ret[i, :l] = xs[i, :l]
    return ret


def th_accuracy(pad_outputs, pad_targets, ignore_label):
    """Calculate accuracy.
    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax, D).
        ignore_label (int): Ignore label id.
    Returns:
        float: Accuracy value (0.0 - 1.0).
    """
    pad_pred = pad_outputs.view(pad_targets.size(0), pad_targets.size(1), pad_outputs.size(1)).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return float(numerator) / float(denominator)


def get_subsample(train_args, mode, arch):
    """Parse the subsampling factors from the args for the specified `mode` and `arch`.
    Args:
        train_args: argument Namespace containing options.
        mode: one of ('asr', 'mt', 'st')
        arch: one of ('rnn', 'rnn-t', 'rnn_mix', 'rnn_mulenc', 'transformer')
    Returns:
        np.ndarray / List[np.ndarray]: subsampling factors.
    """
    if arch == "transformer":
        return np.array([1])

    elif mode == "mt" and arch == "rnn":
        # +1 means input (+1) and layers outputs (train_args.elayer)
        subsample = np.ones(train_args.elayers + 1, dtype=np.int)
        logging.warning("Subsampling is not performed for machine translation.")
        logging.info("subsample: " + " ".join([str(x) for x in subsample]))
        return subsample

    elif (mode == "asr" and arch in ("rnn", "rnn-t")) or (mode == "mt" and arch == "rnn") or (mode == "st" and arch == "rnn"):
        subsample = np.ones(train_args.elayers + 1, dtype=np.int)
        if train_args.etype.endswith("p") and not train_args.etype.startswith("vgg"):
            ss = train_args.subsample.split("_")
            for j in range(min(train_args.elayers + 1, len(ss))):
                subsample[j] = int(ss[j])
        else:
            logging.warning("Subsampling is not performed for vgg*. " "It is performed in max pooling layers at CNN.")
        logging.info("subsample: " + " ".join([str(x) for x in subsample]))
        return subsample

    elif mode == "asr" and arch == "rnn_mix":
        subsample = np.ones(train_args.elayers_sd + train_args.elayers + 1, dtype=np.int)
        if train_args.etype.endswith("p") and not train_args.etype.startswith("vgg"):
            ss = train_args.subsample.split("_")
            for j in range(min(train_args.elayers_sd + train_args.elayers + 1, len(ss))):
                subsample[j] = int(ss[j])
        else:
            logging.warning("Subsampling is not performed for vgg*. " "It is performed in max pooling layers at CNN.")
        logging.info("subsample: " + " ".join([str(x) for x in subsample]))
        return subsample

    elif mode == "asr" and arch == "rnn_mulenc":
        subsample_list = []
        for idx in range(train_args.num_encs):
            subsample = np.ones(train_args.elayers[idx] + 1, dtype=np.int)
            if train_args.etype[idx].endswith("p") and not train_args.etype[idx].startswith("vgg"):
                ss = train_args.subsample[idx].split("_")
                for j in range(min(train_args.elayers[idx] + 1, len(ss))):
                    subsample[j] = int(ss[j])
            else:
                logging.warning(
                    "Encoder %d: Subsampling is not performed for vgg*. " "It is performed in max pooling layers at CNN.",
                    idx + 1,
                )
            logging.info("subsample: " + " ".join([str(x) for x in subsample]))
            subsample_list.append(subsample)
        return subsample_list

    else:
        raise ValueError("Invalid options: mode={}, arch={}".format(mode, arch))
def get_activation(act):
    """Return activation function."""
    # Lazy load to avoid unused import
    from conformer.swish import Swish

    activation_funcs = {
        "hardtanh": torch.nn.Hardtanh,
        "tanh": torch.nn.Tanh,
        "relu": torch.nn.ReLU,
        "selu": torch.nn.SELU,
        "swish": Swish,
    }

    return activation_funcs[act]()