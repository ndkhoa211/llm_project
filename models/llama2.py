#from models.layers.attentions import MultiHeadAttention
import torch
import torch.nn as nn

LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,     # Vocabulary size
    "context_length": 4096,  # Context length
    "emb_dim": 4096,         # Embedding dimension
    "n_heads": 32,           # Number of attention heads
    "n_layers": 32,          # Number of layers
    "hidden_dim": 11008,     # NEW: Size of the intermediate dimension in FeedForward
    "dtype": torch.bfloat16  # NEW: Lower-precision dtype to reduce memory usage
}

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



def precompute_rope_params(head_dim,
                           theta_base=10_000,
                           context_length=4096):
  assert head_dim % 2 == 0, "Embedding dimension must be even"

  # (Eq. 1) compute the inverse frequencies
  # theta_i = 10000 ^ (-2(i-1)/dim) for i = 1, 2, ..., dim/2
  inv_freq = 1.0 / (
      theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim))

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

  return x_rotated.to(dtype=x.dtype)



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



class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(input_embedding_dim=config["emb_dim"],
                                            output_embedding_dim=config["emb_dim"],
                                            context_length=config["context_length"],
                                            #dropout=config["drop_rate"],
                                            num_heads=config["n_heads"],
                                            #qkv_bias=config["qkv_bias"],
                                            dtype=config["dtype"],   # NEW
                                            )
        self.feed_forward = FeedForward(config)

        ################################### NEW ###################################
        # self.layer_norm1 = LayerNorm(config["emb_dim"])
        # self.layer_norm2 = LayerNorm(config["emb_dim"])
        self.norm1 = RMSNorm(config["emb_dim"])
        self.norm2 = RMSNorm(config["emb_dim"])
        ###########################################################################

        # self.drop_skip = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        # skip connection for attention block
        shortcut = x
        # x = self.layer_norm1(x)
        x = self.norm1(x)
        x = self.attention(x)  # shape: [batch_size, num_tokens, emb_size]
        # x = self.drop_skip(x)
        x = shortcut + x  # skip connection

        # skip connection for feed forward block
        shortcut = x
        # x = self.layer_norm2(x)
        x = self.norm2(x)
        x = self.feed_forward(x)
        # x = self.drop_skip(x)
        x = shortcut + x  # skip connection

        return x



class Llama2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_emb = nn.Embedding(config["vocab_size"],
                                      config["emb_dim"],
                                      dtype=config["dtype"])   # NEW
        # self.position_emb = nn.Embedding(config["context_length"],
        #                                  config["emb_dim"])
        # self.drop_emb = nn.Dropout(config["drop_rate"])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layers"])])

        ################################### NEW ###################################
        # self.final_norm = LayerNorm(config["emb_dim"])
        self.final_norm = RMSNorm(config["emb_dim"])
        ###########################################################################

        self.out_head = nn.Linear(config["emb_dim"],
                                  config["vocab_size"],
                                  bias=False,
                                  dtype=config["dtype"])   # NEW

    def forward(self, input_token):
        # batch_size, sequence_length = input_token.shape
        token_embeds = self.token_emb(input_token)
        # position_embeds = self.position_emb(
        #     torch.arange(sequence_length,
        #                  device=input_token.device))
        embeds = token_embeds # + position_embeds
        # x = self.drop_emb(embeds)
        x = embeds
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits



class LlamaTokenizer:
  def __init__(self, tokenizer_file):
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_file)
    self.tokenizer = sp

  def encode(self, text):
    return self.tokenizer.encode(text, out_type=int)

  def decode(self, tokens):
    return self.tokenizer.decode(tokens)



