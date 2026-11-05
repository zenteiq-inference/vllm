# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configurations for Dhir 2B (dense) and Dhir 5B MoE.

The checkpoints on the Hub ship these classes as remote code; registering them here lets vLLM load the models
without `--trust-remote-code`.
"""

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig


@strict
class DhirConfig(PreTrainedConfig):
    model_type = "dhir"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 1_000_000.0

    vocab_size: int = 131072
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 16384
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_parameters: dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    pad_token_id: int | None = 0
    bos_token_id: int | None = 2
    eos_token_id: int | list[int] | None = None


@strict
class DhirMoeConfig(PreTrainedConfig):
    model_type = "dhir_moe"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 10_000_000.0

    vocab_size: int = 131072
    hidden_size: int = 2560
    moe_intermediate_size: int = 1536
    num_hidden_layers: int = 28
    num_attention_heads: int = 20
    num_key_value_heads: int = 5
    head_dim: int = 128
    num_experts: int = 12
    num_experts_per_tok: int = 2
    norm_topk_prob: bool = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 16384
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    pad_token_id: int | None = 0
    bos_token_id: int | None = 2
    eos_token_id: int | list[int] | None = None
