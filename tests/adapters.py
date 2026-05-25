from __future__ import annotations

import os
import heapq
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy as np
import numpy.typing as npt
import regex as re
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

import math
from collections import defaultdict
from collections import Counter

GPT2_PRETOKENIZER_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    return in_features @ weights.T
    raise NotImplementedError


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    return weights[token_ids]
    raise NotImplementedError


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    a1 = in_features @ w1_weight.T
    a1 = run_silu(a1)
    a2 = in_features @ w3_weight.T 
    a1 = a1 * a2
    return a1 @ w2_weight.T
    raise NotImplementedError


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d = Q.size(-1)
    score = Q @ K.transpose(-2, -1)/math.sqrt(d)
    if mask is not None:
        score = score.masked_fill(~mask, float('-inf'))
    weight = torch.softmax(score, -1)
    return weight @ V
    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    head_dim = d_model // num_heads
    Q = in_features @ q_proj_weight.T
    K = in_features @ k_proj_weight.T
    V = in_features @ v_proj_weight.T
    *dims, seq_len = Q.shape[:-1]
    Q = Q.reshape(*dims, seq_len, num_heads, head_dim)
    K = K.reshape(*dims, seq_len, num_heads, head_dim)
    V = V.reshape(*dims, seq_len, num_heads, head_dim)
    Q = Q.transpose(-2, -3)  
    K = K.transpose(-2, -3) 
    V = V.transpose(-2, -3)
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device)
    )
    out = run_scaled_dot_product_attention(Q, K, V, mask = causal_mask)  
    out = out.transpose(-2, -3)
    out = out.contiguous().reshape(*dims, seq_len, -1)
    return out @ o_proj_weight.T
    raise NotImplementedError


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    head_dim = d_model // num_heads
    Q = in_features @ q_proj_weight.T
    K = in_features @ k_proj_weight.T
    V = in_features @ v_proj_weight.T
    *dims, seq_len = Q.shape[:-1]
    Q = Q.reshape(*dims, seq_len, num_heads, head_dim)
    K = K.reshape(*dims, seq_len, num_heads, head_dim)
    V = V.reshape(*dims, seq_len, num_heads, head_dim)
    Q = Q.transpose(-2, -3)  
    K = K.transpose(-2, -3) 
    V = V.transpose(-2, -3)
    
    if token_positions is None:
        positions = torch.arange(seq_len, device=in_features.device)
        positions = positions.view(1, seq_len).expand(*dims, seq_len)
    else:
        positions = token_positions
    
    Q = run_rope(head_dim, theta, max_seq_len, Q, positions)
    K = run_rope(head_dim, theta, max_seq_len, K, positions)
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device)
    )
    out = run_scaled_dot_product_attention(Q, K, V, mask = causal_mask)  
    out = out.transpose(-2, -3)
    out = out.contiguous().reshape(*dims, seq_len, -1)
    return out @ o_proj_weight.T
    raise NotImplementedError


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    *dims, seq_len = in_query_or_key.shape[:-1]
    # 5. 计算 RoPE 的 cos 和 sin
    # 频率: theta^{-2i/head_dim} for i in 0..(head_dim//2 - 1)
    freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=in_query_or_key.device).float() / d_k))
    
    positions = token_positions.to(in_query_or_key.device)
    while positions.ndim < in_query_or_key.ndim - 1:
        positions = positions.unsqueeze(-2)

    # 角度: positions * freqs
    angles = positions.unsqueeze(-1) * freqs  # [..., seq_len, head_dim//2]
    cos = torch.cos(angles)  # [..., seq_len, head_dim//2]
    sin = torch.sin(angles)  # [..., seq_len, head_dim//2]

    # 6. 应用 RoPE 到 Q 和 K
    # 重塑为 [..., num_heads, seq_len, head_dim//2, 2]
    out = in_query_or_key.reshape(*dims, seq_len, d_k // 2, 2)
    
    # 分离分量
    out1, out2 = out[..., 0], out[..., 1]
    
    # 应用旋转矩阵
    out_rot1 = out1 * cos - out2 * sin
    out_rot2 = out1 * sin + out2 * cos
    
    # 合并回原始形状
    ans = torch.stack([out_rot1, out_rot2], dim=-1).reshape(*dims, seq_len, d_k)
    return ans
    raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    batch, seq_len, d = in_features.shape
    h = run_rmsnorm(d_model, 1e-5, weights["ln1.weight"], in_features)
    Wq = weights["attn.q_proj.weight"]
    Wk = weights["attn.k_proj.weight"]
    Wv = weights["attn.v_proj.weight"]
    Wo = weights["attn.output_proj.weight"]
    out = run_multihead_self_attention_with_rope(d_model, num_heads, max_seq_len,
        theta, Wq, Wk, Wv, Wo, h)
    out = in_features + out
    h2 = run_rmsnorm(d_model, 1e-5, weights["ln2.weight"], out)
    W1 = weights["ffn.w1.weight"]
    W2 = weights["ffn.w2.weight"]
    W3 = weights["ffn.w3.weight"]
    out1 = out + run_swiglu(d_model, d_ff, W1, W2, W3, h2)
    return out1


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    batch, seq_len = in_indices.shape
    # (B, T, d_model)
    x = weights["token_embeddings.weight"][in_indices]
    for i in range(num_layers):
        prefix = f"layers.{i}."
        block_weights = {
            k[len(prefix):]: v
            for k, v in weights.items()
            if k.startswith(prefix)
        }
        x = run_transformer_block(
            d_model,
            num_heads,
            d_ff,
            context_length,
            rope_theta,
            block_weights,
            x
        )    
    # =========================
    # 3. Final RMSNorm
    # =========================
    x = run_rmsnorm(
        d_model,
        1e-5,
        weights["ln_final.weight"],
        x
    )
    # =========================
    # 4. LM head
    # =========================
    logits = x @ weights["lm_head.weight"].T
    return logits


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
     # 1. 计算均方根
    rms = in_features.pow(2).mean(dim=-1, keepdim=True)
    # 2. 归一化
    normed = in_features / torch.sqrt(rms + eps)
    # 3. learnable scale（逐维缩放）
    return normed * weights
    raise NotImplementedError


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    return in_features * torch.sigmoid(in_features)
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    n = len(dataset)
    starts = torch.randint(0, n - context_length, (batch_size,))
    offsets = np.arange(context_length)
    start_indices = starts.cpu().numpy()[:, None]
    x_np = np.asarray(dataset[start_indices + offsets], dtype=np.int64)
    y_np = np.asarray(dataset[start_indices + offsets + 1], dtype=np.int64)
    x = torch.from_numpy(x_np).to(device=device)
    y = torch.from_numpy(y_np).to(device=device)

    return x, y
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
     # 1. 数值稳定：减去最大值
    x_max = in_features.max(dim=dim, keepdim=True).values

    x = in_features - x_max

    # 2. exp
    exp_x = torch.exp(x)

    # 3. normalize
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
    raise NotImplementedError


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
     # 1. 数值稳定：log-sum-exp trick
    max_logits = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - max_logits

    # 2. log sum exp
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))

    # 3. gather correct class logits
    correct_logits = shifted[torch.arange(shifted.shape[0], device=shifted.device), targets]

    # 4. loss per sample
    loss = log_sum_exp - correct_logits

    # 5. mean
    return loss.mean()
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    total_norm = 0.0

    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.norm(2)
        total_norm += param_norm ** 2

    total_norm = total_norm ** 0.5

    # 2. 计算缩放系数
    clip_coef = max_l2_norm / (total_norm + 1e-6)

    # 3. 如果没超，就不动
    if clip_coef >= 1:
        return
    for p in parameters:
        if p.grad is None:
            continue
        p.grad.mul_(clip_coef)


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return torch.optim.AdamW
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it <= warmup_iters:
        return max_learning_rate * it / warmup_iters
    # 2. Cosine decay phase
    elif it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)  # in [0,1]
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate + (
            max_learning_rate - min_learning_rate
        ) * cosine_decay
    # 3. After decay → constant min lr
    return min_learning_rate


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    import regex as re

    GPT2_PRETOKENIZER_PATTERN = (
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    class Tokenizer:
        def __init__(self):
            self.id_to_token = dict(vocab)
            self.token_to_id = {token: idx for idx, token in vocab.items()}
            self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}

            self.special_tokens = special_tokens or []
            self.special_token_to_id = {
                tok: self.token_to_id[tok.encode("utf-8")] for tok in self.special_tokens
            }

            if self.special_tokens:
                escaped = sorted((re.escape(tok) for tok in self.special_tokens), key=len, reverse=True)
                self.special_pattern = re.compile("|".join(escaped))
            else:
                self.special_pattern = None

        def _split_special(self, text: str) -> list[str]:
            if self.special_pattern is None:
                return [text]
            # 切出特殊词
            parts = self.special_pattern.split(text)
            # 找到特殊词
            matches = self.special_pattern.findall(text)

            out = []
            for i, part in enumerate(parts):
                # 跳过空串
                if part:
                    out.append(part)
                # 一定是交错插入的
                if i < len(matches):
                    out.append(matches[i])
            return out

        def _bpe_encode_bytes(self, data: bytes) -> list[int]:
            # 初始态
            tokens = [bytes([b]) for b in data]

            while len(tokens) >= 2:
                best_idx = None
                best_rank = None

                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    rank = self.merge_ranks.get(pair)
                    # 编号越小越优先
                    if rank is not None and (best_rank is None or rank < best_rank):
                        best_rank = rank
                        best_idx = i

                if best_idx is None:
                    break

                merged = tokens[best_idx] + tokens[best_idx + 1]
                tokens = tokens[:best_idx] + [merged] + tokens[best_idx + 2 :]

            return [self.token_to_id[token] for token in tokens]

        def encode(self, text: str) -> list[int]:
            ids = []

            for chunk in self._split_special(text):
                if chunk in self.special_token_to_id:
                    ids.append(self.special_token_to_id[chunk])
                    continue

                for piece in re.findall(GPT2_PRETOKENIZER_PATTERN, chunk):
                    ids.extend(self._bpe_encode_bytes(piece.encode("utf-8")))

            return ids

        def decode(self, ids: list[int]) -> str:
            data = b"".join(self.id_to_token[i] for i in ids)
            return data.decode("utf-8", errors="replace")

        def encode_iterable(self, iterable):
            for chunk in iterable:
                yield from self.encode(chunk)

    return Tokenizer()
    raise NotImplementedError


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # "cat" -> (b'c', b'a', b't')
    def word_to_tokens(word: str) -> tuple[bytes, ...]:
        return tuple(bytes([b]) for b in word.encode("utf-8"))
    # 计数
    def count_pairs(tokens: tuple[bytes, ...]) -> Counter[tuple[bytes, bytes]]:
        counts = Counter()
        for i in range(len(tokens) - 1):
            counts[(tokens[i], tokens[i + 1])] += 1
        return counts
    # 合并
    def merge_tokens(tokens: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
        a, b = pair
        merged = a + b
        out = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        return tuple(out)

    class PairHeapEntry:
        __slots__ = ("count", "pair")

        def __init__(self, count: int, pair: tuple[bytes, bytes]):
            self.count = count
            self.pair = pair

        def __lt__(self, other: "PairHeapEntry") -> bool:
            if self.count != other.count:
                return self.count > other.count
            return self.pair > other.pair

    token_pattern = re.compile(GPT2_PRETOKENIZER_PATTERN)

    word_freqs: Counter[tuple[bytes, ...]] = Counter()
    pretoken_buffer = ""

    def update_word_freqs(chunk: str) -> None:
        if not chunk:
            return
        to_tokens = word_to_tokens
        for piece in re.findall(GPT2_PRETOKENIZER_PATTERN, chunk):
            word_freqs[to_tokens(piece)] += 1

    def consume_pretokens(text: str, final: bool = False) -> None:
        nonlocal pretoken_buffer
        if text:
            pretoken_buffer += text
        if not pretoken_buffer:
            return
        to_tokens = word_to_tokens
        matches = list(token_pattern.finditer(pretoken_buffer))
        if not matches:
            return

        if final:
            for match in matches:
                word_freqs[to_tokens(match.group(0))] += 1
            pretoken_buffer = ""
            return

        for match in matches[:-1]:
            word_freqs[to_tokens(match.group(0))] += 1
        pretoken_buffer = pretoken_buffer[matches[-1].start() :]

    flush_threshold = 1 << 20
    text_buffer = ""

    def append_text(text: str) -> None:
        nonlocal text_buffer
        if not text:
            return
        text_buffer += text
        while len(text_buffer) >= flush_threshold:
            consume_pretokens(text_buffer[:flush_threshold])
            text_buffer = text_buffer[flush_threshold:]

    with open(input_path, "r", encoding="utf-8") as f:
        if not special_tokens:
            while True:
                chunk = f.read(flush_threshold)
                if not chunk:
                    break
                append_text(chunk)
        else:
            text = f.read()
            if len(special_tokens) == 1:
                for chunk in text.split(special_tokens[0]):
                    update_word_freqs(chunk)
            else:
                escaped = sorted((re.escape(tok) for tok in special_tokens), key=len, reverse=True)
                special_pattern = re.compile("|".join(escaped))
                chunk_start = 0
                for match in special_pattern.finditer(text):
                    if chunk_start < match.start():
                        update_word_freqs(text[chunk_start : match.start()])
                    chunk_start = match.end()
                if chunk_start < len(text):
                    update_word_freqs(text[chunk_start:])

    if text_buffer:
        consume_pretokens(text_buffer)
    if pretoken_buffer:
        consume_pretokens("", final=True)
    # 分词计数
    # ((b'h', b'i'), 3)

    vocab = {}
    idx = 0
    # 初始化词汇表 特殊字
    for tok in special_tokens:
        vocab[idx] = tok.encode("utf-8")
        idx += 1
    
    # 遍历一个 Counter 默认遍历的是它的 key
    seen_tokens = set(vocab.values())
    for byte_value in range(256):
        token = bytes([byte_value])
        if token not in seen_tokens:
            vocab[idx] = token
            seen_tokens.add(token)
            idx += 1

    merges = []
    # 编号-(token-频率)
    word_states: dict[int, tuple[tuple[bytes, ...], int, Counter[tuple[bytes, bytes]]]] = {}
    # pair频率表
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    # pair出现的词
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    # 访问完整值而不是key，enumerate加编号
    # (0, ((b'h', b'i'), 3))
    for wid, (tokens, freq) in enumerate(word_freqs.items()):
        local_pairs = count_pairs(tokens)
        word_states[wid] = (tokens, freq, local_pairs)
        for pair, c in local_pairs.items():
            pair_counts[pair] += c * freq
            pair_to_words[pair].add(wid)

    pair_heap = [PairHeapEntry(count, pair) for pair, count in pair_counts.items() if count > 0]
    heapq.heapify(pair_heap)

    def push_pair_if_live(pair: tuple[bytes, bytes]) -> None:
        count = pair_counts.get(pair, 0)
        if count > 0:
            heapq.heappush(pair_heap, PairHeapEntry(count, pair))

    while len(vocab) < vocab_size and pair_counts:
        best_pair = None
        best_count = 0
        while pair_heap:
            entry = heapq.heappop(pair_heap)
            current_count = pair_counts.get(entry.pair)
            if current_count == entry.count and current_count > 0:
                best_pair = entry.pair
                best_count = current_count
                break
        if best_pair is None or best_count <= 0:
            break

        # 找到被影响单词
        affected_wids = list(pair_to_words[best_pair])
        if not affected_wids:
            del pair_counts[best_pair]
            continue

        for wid in affected_wids:
            old_tokens, freq, old_local_pairs = word_states[wid]
            for pair, c in old_local_pairs.items():
                pair_counts[pair] -= c * freq
                pair_to_words[pair].discard(wid)
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
                else:
                    push_pair_if_live(pair)

            new_tokens = merge_tokens(old_tokens, best_pair)
            new_local_pairs = count_pairs(new_tokens)
            word_states[wid] = (new_tokens, freq, new_local_pairs)
            for pair, c in new_local_pairs.items():
                pair_counts[pair] += c * freq
                pair_to_words[pair].add(wid)
                push_pair_if_live(pair)

        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        if new_token not in seen_tokens:
            vocab[idx] = new_token
            seen_tokens.add(new_token)
            idx += 1

    return vocab, merges
    raise NotImplementedError
