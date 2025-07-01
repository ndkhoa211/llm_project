import torch
import torch.nn as nn
from pathlib import Path
import os
import tiktoken
from tiktoken.load import load_tiktoken_bpe



LLAMA3_CONFIG_8B = {
    "vocab_size": 128_256,   # NEW: 4x vocabulary size
    "context_length": 8192,  # NEW: 2x context length
    "emb_dim": 4096,         # Embedding dimension
    "n_heads": 32,           # Number of attention heads
    "n_layers": 32,          # Number of layers
    "hidden_dim": 14_336,    # NEW: Larger size of the intermediate dimension in FeedForward
    "num_kv_groups": 8,        # NEW: Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,  # NEW: The base in RoPE's "theta" was increased to 500_000
    "rope_freq": None,       # NEW: Additional configuration for adjusting the RoPE frequencies
    "dtype": torch.bfloat16  # Lower-precision dtype to reduce memory usage
}


# Llama 3.1 8B
LLAMA31_CONFIG_8B = {
    "vocab_size": 128_256,      # Vocabulary size
    "context_length": 131_072,  # NEW: Larger supported context length
    "emb_dim": 4096,            # Embedding dimension
    "n_heads": 32,              # Number of attention heads
    "n_layers": 32,             # Number of layers
    "hidden_dim": 14_336,       # Size of the intermediate dimension in FeedForward
    "num_kv_groups": 8,           # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,     # The base in RoPE's "theta"
    "dtype": torch.bfloat16,    # Lower-precision dtype to reduce memory usage
    "rope_freq": {              # NEW: RoPE frequency scaling
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}


# Llama 3.2 1B
LLAMA32_CONFIG = {
    "vocab_size": 128_256,           # Vocabulary size
    "context_length": 131_072,       # Context length that was used to train the model
    "emb_dim": 2048,                 # Embedding dimension
    "n_heads": 32,                   # Number of attention heads
    "n_layers": 16,                  # Number of layers
    "hidden_dim": 8192,              # Size of the intermediate dimension in FeedForward
    "num_kv_groups": 8,                # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,          # The base in RoPE's "theta"
    "dtype": torch.bfloat16,         # Lower-precision dtype to reduce memory usage
    "rope_freq": {                   # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}


# Llama 3.2 3B
LLAMA32_CONFIG = {
    "vocab_size": 128_256,           # Vocabulary size
    "context_length": 131_072,       # Context length that was used to train the model
    "emb_dim": 3072,                 # Embedding dimension
    "n_heads": 24,                   # Number of attention heads
    "n_layers": 28,                  # Number of layers
    "hidden_dim": 8192,              # Size of the intermediate dimension in FeedForward
    "num_kv_groups": 8,                # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,          # The base in RoPE's "theta"
    "dtype": torch.bfloat16,         # Lower-precision dtype to reduce memory usage
    "rope_freq": {                   # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}

LLAMA_SIZE_STR = "1B" if LLAMA32_CONFIG["emb_dim"] == 2048 else "3B"



def precompute_rope_params(head_dim,
                           theta_base=10_000,
                           context_length=4096,
                           freq_config=None):
  assert head_dim % 2 == 0, "Embedding dimension must be even"

  # (Eq. 1) compute the inverse frequencies
  # theta_i = 10000 ^ (-2(i-1)/dim) for i = 1, 2, ..., dim/2
  inv_freq = 1.0 / (
      theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim))

  ################################ NEW ###############################################
  # Frequency adjustments
  if freq_config is not None:
    low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
    high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

    wavelen = 2 * torch.pi / inv_freq

    inv_freq_llama = torch.where(
        wavelen > low_freq_wavelen,
        inv_freq / freq_config["factor"],
        inv_freq
    )

    smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
        freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
    )

    smooth_inv_freq = (
        (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
    )

    is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq,
                                 smooth_inv_freq,
                                 inv_freq_llama)
    inv_freq = inv_freq_llama
  ####################################################################################


  # generate position indices
  positions = torch.arange(context_length)

  # compute the angles
  # positions -> row vector
  # inv_freq -> column vector
  angles = positions[:, None] * inv_freq[None, :] # (context_length, head_dim // 2)

  # expand angles to match the head_dim
  angles = torch.cat([angles, angles], dim=1) # (context_length, head_dim)

  # precompute sine and cosine
  cos = torch.cos(angles)
  sin = torch.sin(angles)

  return cos, sin



def compute_rope(x, cos, sin):
  # x -> (batch_size, num_heads, seq_len, head_dim)
  batch_size, num_heads, seq_len, head_dim = x.shape
  assert head_dim % 2 == 0, "Embedding dimension must be even"

  # split x into two subspaces
  x1 = x[..., :head_dim // 2] # first half
  x2 = x[..., head_dim // 2 :] # second half

  # adjust sin and cos shapes
  cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
  sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

  # apply the rotary transformation
  rotated = torch.cat((-x2, x1), dim=-1)
  x_rotated = (x * cos) + (rotated * sin)

  # It's ok to use lower-precision after applying cos and sin rotation
  return x_rotated.to(dtype=x.dtype)



class SiLU(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return x * torch.sigmoid(x)



class FeedForward(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.silu = SiLU()
    self.fc1 = nn.Linear(config["emb_dim"],
                         config["hidden_dim"],
                         dtype=config["dtype"],
                         bias=False)
    self.fc2 = nn.Linear(config["emb_dim"],
                         config["hidden_dim"],
                         dtype=config["dtype"],
                         bias=False)
    self.fc3 = nn.Linear(config["hidden_dim"],
                         config["emb_dim"],
                         dtype=config["dtype"],
                         bias=False)

  def forward(self, x):
    x_fc1 = self.fc1(x)
    x_silu = self.silu(x_fc1)
    x_fc2 = self.fc2(x)
    x_swiglu = x_silu * x_fc2
    return self.fc3(x_swiglu)



class RMSNorm(nn.Module):
  def __init__(self, emb_dim, eps=1e-5):
    super().__init__()
    self.eps = eps
    self.emb_dim = emb_dim
    self.weight = nn.Parameter(torch.ones(emb_dim)).float()

  def forward(self, x):
    means = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = (x *
                torch.rsqrt(means + self.eps) # reciprocal of the square root
                )
    return (x_normed * self.weight).to(dtype=x.dtype)



# MultiHeadAttention only for comparison purposes
class MultiHeadAttention(nn.Module):
    def __init__(self,
                 input_embedding_dim,
                 output_embedding_dim,
                 context_length,
                 #dropout,
                 num_heads,
                 #qkv_bias=False,
                 dtype=None):
        super().__init__()
        assert (output_embedding_dim % num_heads == 0), \
            "output_embedding_dim must be divisible by num_heads"

        self.output_embedding_dim = output_embedding_dim
        self.num_heads = num_heads
        self.head_dim = output_embedding_dim // num_heads

        ################################### NEW ###################################
        # Set bias=False and dtype=dtype for all linear layers below
        ###########################################################################
        self.W_query = nn.Linear(input_embedding_dim,
                                 output_embedding_dim,
                                 bias=False,
                                 dtype=dtype)
        self.W_key = nn.Linear(input_embedding_dim,
                               output_embedding_dim,
                               bias=False,
                               dtype=dtype)
        self.W_value = nn.Linear(input_embedding_dim,
                                 output_embedding_dim,
                                 bias=False,
                                 dtype=dtype)
        self.output_projection = nn.Linear(output_embedding_dim,
                                           output_embedding_dim,
                                           bias=False,
                                           dtype=dtype)  # to combine head outputs
        # self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1))

        ################################### NEW ###################################
        cos, sin = precompute_rope_params(head_dim=self.head_dim,
                                          context_length=context_length)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        ###########################################################################

    def forward(self, inputs):
        batch, num_tokens, input_embedding_dim = inputs.shape

        # qkv shapes : (batch, num_tokens, output_embedding_dim)
        keys = self.W_key(inputs)
        values = self.W_value(inputs)
        queries = self.W_query(inputs)

        # qkv shapes : (batch, num_tokens, num_heads, head_dim)
        keys = keys.view(batch, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch, num_tokens, self.num_heads, self.head_dim)

        # qkv shapes : (batch, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        ################################### NEW ###################################
        keys = compute_rope(keys, self.cos, self.sin)
        queries = compute_rope(queries, self.cos, self.sin)
        ###########################################################################

        # compute attention scores for each head
        attention_scores = queries @ keys.transpose(3, 2)
        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], - torch.inf)

        # compute attention weights + dropout
        masked_attention_weight = torch.softmax(
            attention_scores / (keys.shape[-1] ** 0.5),
            dim=-1)
        # masked_attention_dropout_weight = self.dropout(masked_attention_weight)

        # compute context vectors
        # shape : (batch, num_tokens, num_heads, head_dim)
        #context_vector = (masked_attention_dropout_weight @ values).transpose(1, 2)
        context_vector = (masked_attention_weight @ values).transpose(1, 2)

        # combine heads, where self.d_out = self.num_heads * self.head_dim
        # shape : (batch, num_tokens, output_embedding_dim)
        context_vector = context_vector.contiguous().view(
            batch, num_tokens, self.output_embedding_dim)

        # linear projection (optional)
        context_vector = self.output_projection(context_vector)

        return context_vector



class GroupedQueryAttention(nn.Module):
  def __init__(
      self,
      input_embedding_dim,
      output_embedding_dim,
      num_heads,
      num_kv_groups,
      dtype=None,
      ):
    super().__init__()
    assert output_embedding_dim % num_heads == 0, "output_embedding_dim must be divisible by num_heads"
    assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

    self.output_embedding_dim = output_embedding_dim
    self.num_heads = num_heads
    self.head_dim = output_embedding_dim // num_heads


    self.W_key = nn.Linear(input_embedding_dim,
                           num_kv_groups * self.head_dim,
                           bias=False,
                           dtype=dtype)
    self.W_value = nn.Linear(input_embedding_dim,
                           num_kv_groups * self.head_dim,
                           bias=False,
                           dtype=dtype)
    self.num_kv_groups = num_kv_groups
    self.group_size = num_heads // num_kv_groups


    self.W_query = nn.Linear(input_embedding_dim,
                            output_embedding_dim,
                            bias=False,
                            dtype=dtype)
    self.output_projection = nn.Linear(output_embedding_dim,
                                        output_embedding_dim,
                                        bias=False,
                                        dtype=dtype) # to combine head outputs


  def forward(self,
              inputs,
              mask,#=None,
              cos,#=None,
              sin,#=None
             ):
    batch, num_tokens, input_embedding_dim = inputs.shape

    # qkv shapes : (batch, num_tokens, output_embedding_dim)
    keys = self.W_key(inputs)
    values = self.W_value(inputs)
    queries = self.W_query(inputs)


    # kv shapes : (batch, num_tokens, num_kv_groups, head_dim)
    keys = keys.view(batch, num_tokens, self.num_kv_groups, self.head_dim)
    values = values.view(batch, num_tokens, self.num_kv_groups, self.head_dim)
    # q shape : (batch, num_tokens, num_heads, head_dim)
    queries = queries.view(batch, num_tokens, self.num_heads, self.head_dim)


    # kv shapes : (batch, num_kv_groups, num_tokens, head_dim)
    # q shape : (batch, num_heads, num_tokens, head_dim)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)
    queries = queries.transpose(1, 2)


    # Apply RoPE
    if cos is not None:
      keys = compute_rope(keys, cos, sin)
      queries = compute_rope(queries, cos, sin)


    # Expand keys and values to match the number of heads
    # kv shapes : (batch, num_heads, num_tokens, head_dim)
    keys = keys.repeat_interleave(self.group_size, dim=1)
    values = values.repeat_interleave(self.group_size, dim=1)
    # For example, before repeat_interleave along dim=1 (query groups):
    #   [K1, K2]
    # After repeat_interleave (each query group is repeated group_size times):
    #   [K1, K1, K2, K2]
    # If we used regular repeat instead of repeat_interleave, we'd get:
    #   [K1, K2, K1, K2]


    # compute attention scores for each head with a causal mask
    # shape : (batch, num_heads, num_tokens, num_tokens)
    attention_scores = queries @ keys.transpose(2, 3)


    # Create mask on the fly
    # if mask is None:
    #   mask = torch.triu(torch.ones(num_tokens,
    #                                num_tokens,
    #                                device=inputs.device,
    #                                dtype=torch.bool,
    #                                ),
    #                     diagonal=1)


    # use the mask to fill attention scores
    attention_scores.masked_fill(mask, - torch.inf)


    # compute attention weights
    masked_attention_weight = torch.softmax(
        attention_scores / (keys.shape[-1] ** 0.5),
        dim=-1)
    assert keys.shape[-1] == self.head_dim


    # compute context vectors
    # shape : (batch, num_tokens, num_heads, head_dim)
    context_vector = (masked_attention_weight @ values).transpose(1, 2)


    # combine heads, where self.d_out = self.num_heads * self.head_dim
    # shape : (batch, num_tokens, output_embedding_dim)
    context_vector = context_vector.contiguous().view(
        batch, num_tokens, self.output_embedding_dim)


    # linear projection (optional)
    context_vector = self.output_projection(context_vector)


    return context_vector



class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = GroupedQueryAttention(input_embedding_dim=config["emb_dim"],
                                               output_embedding_dim=config["emb_dim"],
                                               num_heads=config["n_heads"],
                                               num_kv_groups=config["num_kv_groups"],   # NEW
                                               dtype=config["dtype"],
                                               )
        self.feed_forward = FeedForward(config)
        self.norm1 = nn.RMSNorm(config["emb_dim"],
                                eps=1e-5,
                                dtype=config["dtype"])
        self.norm2 = nn.RMSNorm(config["emb_dim"],
                                eps=1e-5,
                                dtype=config["dtype"])


    def forward(self,
                x,
                mask,#=None,
                cos,#=None,
                sin,#=None
                ):
        # The forward method now accepts `mask` instead of accessing it via self.mask.
        # Also, we now have cos and sin as input for RoPE


        # skip connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x, mask, cos, sin)  # shape: [batch_size, num_tokens, emb_size]
        x = shortcut + x  # skip connection


        # skip connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = shortcut + x  # skip connection

        return x



class Llama3Model(nn.Module):
    def __init__(self, config):
        super().__init__()


        # main model parameters
        self.token_emb = nn.Embedding(config["vocab_size"],
                                      config["emb_dim"],
                                      dtype=config["dtype"])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layers"])])

        self.final_norm = nn.RMSNorm(config["emb_dim"],
                                     eps=1e-5,
                                     dtype=config["dtype"])

        self.out_head = nn.Linear(config["emb_dim"],
                                  config["vocab_size"],
                                  bias=False,
                                  dtype=config["dtype"])

        # reusable utilities
        cos, sin = precompute_rope_params(
            head_dim=config["emb_dim"] // config["n_heads"],
            theta_base=config["rope_base"],
            context_length=config["context_length"],
            freq_config=config["rope_freq"]
        )

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.config = config

    def forward(self, input_token):
        token_embeds = self.token_emb(input_token)
        x = token_embeds

        num_tokens = x.shape[1]
        mask = torch.triu(torch.ones(num_tokens,
                                     num_tokens,
                                     device=x.device,
                                     dtype=torch.bool),
                          diagonal=1)

        for block in self.transformer_blocks:
            x = block(x, mask, self.cos, self.sin)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.config["dtype"]))

        return logits



class Tokenizer:
    """Thin wrapper around tiktoken that keeps track of Llama-3 special IDs."""
    def __init__(self, model_path):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)

        mergeable = load_tiktoken_bpe(model_path)

        # hard-coded from Meta's tokenizer.json
        self.special = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        self.special.update({f"<|reserved_{i}|>": 128002 + i
                             for i in range(256)
                             if 128002 + i not in self.special.values()})

        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
                    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
                    r"|\p{N}{1,3}"
                    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
                    r"|\s*[\r\n]+"
                    r"|\s+(?!\S)"
                    r"|\s+",
            mergeable_ranks=mergeable,
            special_tokens=self.special,
        )

    def encode(self, text, bos=False, eos=False):
        ids = ([self.special["<|begin_of_text|>"]] if bos else []) \
              + self.model.encode(text)
        if eos:
            ids.append(self.special["<|end_of_text|>"])
        return ids

    def decode(self, ids):
        return self.model.decode(ids)



class ChatFormat:

    def __init__(self, tokenizer: Tokenizer, *,
                 default_system="You are a helpful assistant."):
        self.tok = tokenizer
        self.default_system = default_system

    def _header(self, role):
        """Encode <|start_header_id|>role<|end_header_id|>\n\n"""
        return (
            [self.tok.special["<|start_header_id|>"]]
            + self.tok.encode(role)
            + [self.tok.special["<|end_header_id|>"]]
            + self.tok.encode("\n\n")
        )

    def encode(self, user_message, system_message=None):
        sys_msg = system_message if system_message is not None else self.default_system

        ids = [self.tok.special["<|begin_of_text|>"]]

        # system
        ids += self._header("system")
        ids += self.tok.encode(sys_msg)
        ids += [self.tok.special["<|eot_id|>"]]

        # user
        ids += self._header("user")
        ids += self.tok.encode(user_message)
        ids += [self.tok.special["<|eot_id|>"]]

        # assistant header (no content yet)
        ids += self._header("assistant")

        return ids