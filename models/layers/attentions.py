import torch
import torch.nn as nn

##############################################GPT-2############################################
##############################################GPT-2############################################
##############################################GPT-2############################################
##############################################GPT-2############################################
##############################################GPT-2############################################
class SelfAttentionV1(nn.Module):
    def __init__(self, input_embedding_dim, output_embedding_dim):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(input_embedding_dim, output_embedding_dim))
        self.W_key = nn.Parameter(torch.rand(input_embedding_dim, output_embedding_dim))
        self.W_value = nn.Parameter(torch.rand(input_embedding_dim, output_embedding_dim))

    def forward(self, inputs):
        keys = inputs @ self.W_key
        values = inputs @ self.W_value
        queries = inputs @ self.W_query

        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vectors = attention_weights @ values
        return context_vectors


class SelfAttentionV2(nn.Module):
    def __init__(self, input_embedding_dim, output_embedding_dim, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(input_embedding_dim, output_embedding_dim,
                                 bias=qkv_bias)
        self.W_key = nn.Linear(input_embedding_dim, output_embedding_dim,
                               bias=qkv_bias)
        self.W_value = nn.Linear(input_embedding_dim, output_embedding_dim,
                                 bias=qkv_bias)

    def forward(self, inputs):
        keys = self.W_key(inputs)
        values = self.W_value(inputs)
        queries = self.W_query(inputs)

        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vectors = attention_weights @ values
        return context_vectors


class CausalAttention(nn.Module):
    def __init__(self,
                 input_embedding_dim,
                 output_embedding_dim,
                 context_length,
                 dropout,
                 qkv_bias=False):
        super().__init__()
        self.output_embedding_dim = output_embedding_dim
        self.W_query = nn.Linear(input_embedding_dim, output_embedding_dim,
                                 bias=qkv_bias)
        self.W_key = nn.Linear(input_embedding_dim, output_embedding_dim,
                               bias=qkv_bias)
        self.W_value = nn.Linear(input_embedding_dim, output_embedding_dim,
                                 bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1))

    def forward(self, inputs):
        batch, num_tokens, input_embedding_dim = inputs.shape
        keys = self.W_key(inputs)
        values = self.W_value(inputs)
        queries = self.W_query(inputs)

        attention_scores = queries @ keys.transpose(1, 2)
        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], - torch.inf)
        masked_attention_weight = torch.softmax(
            attention_scores / (keys.shape[-1] ** 0.5),
            dim=-1)
        masked_attention_dropout_weight = self.dropout(masked_attention_weight)

        context_vector = masked_attention_dropout_weight @ values
        return context_vector


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self,
                 input_embedding_dim,
                 output_embedding_dim,
                 context_length,
                 dropout,
                 num_heads,
                 qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(input_embedding_dim,
                             output_embedding_dim,
                             context_length,
                             dropout,
                             qkv_bias)
             for _ in range(num_heads)
             ]
        )

    def forward(self, inputs):
        return torch.cat(
            [heads(inputs) for heads in self.heads],  # process SEQUENTIALLY
            dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 input_embedding_dim,
                 output_embedding_dim,
                 context_length,
                 dropout,
                 num_heads,
                 qkv_bias=False):
        super().__init__()
        assert (output_embedding_dim % num_heads == 0), \
            "output_embedding_dim must be divisible by num_heads"

        self.output_embedding_dim = output_embedding_dim
        self.num_heads = num_heads
        self.head_dim = output_embedding_dim // num_heads
        self.W_query = nn.Linear(input_embedding_dim, output_embedding_dim,
                                 bias=qkv_bias)
        self.W_key = nn.Linear(input_embedding_dim, output_embedding_dim,
                               bias=qkv_bias)
        self.W_value = nn.Linear(input_embedding_dim, output_embedding_dim,
                                 bias=qkv_bias)
        self.output_projection = nn.Linear(output_embedding_dim,
                                           output_embedding_dim)  # to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1))

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

        # compute attention scores for each head
        attention_scores = queries @ keys.transpose(3, 2)
        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], - torch.inf)

        # compute attention weights + dropout
        masked_attention_weight = torch.softmax(
            attention_scores / (keys.shape[-1] ** 0.5),
            dim=-1)
        masked_attention_dropout_weight = self.dropout(masked_attention_weight)

        # compute context vectors
        # shape : (batch, num_tokens, num_heads, head_dim)
        context_vector = (masked_attention_dropout_weight @ values).transpose(1, 2)

        # combine heads, where self.d_out = self.num_heads * self.head_dim
        # shape : (batch, num_tokens, output_embedding_dim)
        context_vector = context_vector.contiguous().view(
            batch, num_tokens, self.output_embedding_dim)

        # linear projection (optional)
        context_vector = self.output_projection(context_vector)

        return context_vector


##############################################GPT-2############################################
##############################################GPT-2############################################
##############################################GPT-2############################################
##############################################GPT-2############################################
##############################################GPT-2############################################




