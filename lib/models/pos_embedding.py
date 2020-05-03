# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn

def get_sinusoidal_embedding(length, d_model):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                        "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                        -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    pe.requires_grad = False
    return pe

if __import__('os').environ.get('fairseq_style_embedding'):
    def get_sinusoidal_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: int = None
    ):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        emb.requires_grad = False
        return emb

def get_random_init_embedding(
    num_embeddings: int, embedding_dim: int, padding_idx: int = None
):
    weights = torch.zeros(num_embeddings, embedding_dim, dtype=torch.float, requires_grad=False)
    nn.init.normal_(weights, mean=0, std=embedding_dim ** -0.5)

    return weights

def PositionalEmbedding(
        num_embeddings: int,
        embedding_dim: int,
        learned: bool = False,
):

    if learned:
        return get_random_init_embedding(num_embeddings, embedding_dim)
    else:
        return get_sinusoidal_embedding(num_embeddings, embedding_dim)