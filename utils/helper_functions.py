import torch
import numpy as np

def generate_text_simple(model,
                         input_batch,  # [batch, num_tokens]
                         max_new_tokens,  # numbers of new tokens to be predicted
                         context_size):
    for _ in range(max_new_tokens):
        # crop current context if it exceeds the supported context_size
        crop_input_batch = input_batch[:, -context_size:]

        # predict next token
        with torch.no_grad():
            logits = model(crop_input_batch)

        # consider only logits of the last token
        logits = logits[:, -1, :]  # (batch, n_tokens, vocab_size) -> (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)
        predicted_tokens = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)
        # update input_batch (append predicted tokens to the sequences)
        input_batch = torch.cat([input_batch, predicted_tokens], dim=-1)  # [batch, num_tokens+1]

    return input_batch


def generate_text_advanced(model,
                           input_batch,
                           max_new_tokens,
                           context_size,
                           temperature=1.0,
                           top_k=None,
                           top_p=None,
                           repetition_penalty=1.0,
                           eos_id=None):
    """
    Advanced text generation with multiple decoding strategies.

    Args:
        model: The language model
        input_batch: Input token ids [batch_size, seq_len]
        max_new_tokens: Number of tokens to generate
        context_size: Maximum context length the model can handle
        temperature: Sampling temperature (1.0 = neutral, <1.0 = more focused, >1.0 = more random)
        top_k: If set, only sample from the top k most likely tokens
        top_p: If set, sample from the smallest set of tokens whose cumulative probability exceeds p
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty, >1.0 = penalize repetitions)
        eos_id: Optional end of sequence token id to stop generation early

    Returns:
        Tensor of generated token ids [batch_size, seq_len + max_new_tokens]
    """
    for _ in range(max_new_tokens):
        # Crop context if needed
        crop_input_batch = input_batch[:, -context_size:]

        # Get model predictions
        with torch.no_grad():
            logits = model(crop_input_batch)

        # Consider only the last token's logits
        logits = logits[:, -1, :]  # [batch_size, vocab_size]

        # Apply repetition penalty
        if repetition_penalty != 1.0:
            # Get unique tokens in the input
            used_tokens = torch.unique(input_batch)
            # Penalize previously used tokens
            logits.index_fill_(dim=-1, index=used_tokens,
                               value=logits.index_select(dim=-1, index=used_tokens) / repetition_penalty)

        # Apply temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Convert to probabilities
        probas = torch.softmax(logits, dim=-1)

        # Apply top-k filtering
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            top_logits, top_indices = torch.topk(logits, top_k)
            # Create a mask for non-top-k values
            mask = torch.zeros_like(logits).scatter_(dim=-1, index=top_indices, value=1)
            # Set non-top-k values to -inf before softmax
            logits = torch.where(mask > 0, logits, torch.tensor(-float('inf')).to(logits.device))
            # Recompute probabilities
            probas = torch.softmax(logits, dim=-1)

        # Apply nucleus (top-p) sampling
        if top_p is not None:
            sorted_probas, sorted_indices = torch.sort(probas, descending=True)
            cumsum_probas = torch.cumsum(sorted_probas, dim=-1)
            # Remove tokens after cumsum exceeds top_p
            mask = cumsum_probas <= top_p
            # Always keep at least one token
            mask[..., 0] = True
            sorted_indices = sorted_indices[mask]
            probas = torch.zeros_like(probas).scatter_(-1, sorted_indices, sorted_probas[mask])
            probas.div_(probas.sum(dim=-1, keepdim=True))

        # Sample next token
        predicted_tokens = torch.multinomial(probas, num_samples=1)

        # Stop if EOS token is generated
        if eos_id is not None and (predicted_tokens == eos_id).any():
            break

        # Append prediction to input
        input_batch = torch.cat([input_batch, predicted_tokens], dim=-1)

    return input_batch


def calc_loss_batch(input_batch,
                    target_batch,
                    model,
                    device):
  input_batch = input_batch.to(device)
  target_batch = target_batch.to(device)

  logits = model(input_batch)
  loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1),
                                           target_batch.flatten())
  return loss


def calc_loss_loader(dataloader,
                     model,
                     device,
                     num_batches=None):
  total_loss = 0.
  if len(dataloader) == 0:
    return float("nan")
  elif num_batches is None:
    num_batches = len(dataloader)
  else:
    # reduce the number of batches to match the total number of batches in the data loader
    # if num_batches exceeds the number of batches in the data loader
    num_batches = min(num_batches, len(dataloader))
  for i, (input_batch, target_batch) in enumerate(dataloader):
    if i < num_batches:
      loss = calc_loss_batch(input_batch,
                             target_batch,
                             model,
                             device)
      total_loss += loss.item()
    else:
      break
  return total_loss / num_batches


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))




def load_weights_into_gpt(gpt, params):
    gpt.position_emb.weight = assign(gpt.position_emb.weight, params['wpe'])
    gpt.token_emb.weight = assign(gpt.token_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.transformer_blocks[b].attention.W_query.weight = assign(
            gpt.transformer_blocks[b].attention.W_query.weight, q_w.T)
        gpt.transformer_blocks[b].attention.W_key.weight = assign(
            gpt.transformer_blocks[b].attention.W_key.weight, k_w.T)
        gpt.transformer_blocks[b].attention.W_value.weight = assign(
            gpt.transformer_blocks[b].attention.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.transformer_blocks[b].attention.W_query.bias = assign(
            gpt.transformer_blocks[b].attention.W_query.bias, q_b)
        gpt.transformer_blocks[b].attention.W_key.bias = assign(
            gpt.transformer_blocks[b].attention.W_key.bias, k_b)
        gpt.transformer_blocks[b].attention.W_value.bias = assign(
            gpt.transformer_blocks[b].attention.W_value.bias, v_b)

        gpt.transformer_blocks[b].attention.output_projection.weight = assign(
            gpt.transformer_blocks[b].attention.output_projection.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.transformer_blocks[b].attention.output_projection.bias = assign(
            gpt.transformer_blocks[b].attention.output_projection.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.transformer_blocks[b].feed_forward.layers[0].weight = assign(
            gpt.transformer_blocks[b].feed_forward.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.transformer_blocks[b].feed_forward.layers[0].bias = assign(
            gpt.transformer_blocks[b].feed_forward.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.transformer_blocks[b].feed_forward.layers[2].weight = assign(
            gpt.transformer_blocks[b].feed_forward.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.transformer_blocks[b].feed_forward.layers[2].bias = assign(
            gpt.transformer_blocks[b].feed_forward.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.transformer_blocks[b].layer_norm1.scale = assign(
            gpt.transformer_blocks[b].layer_norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.transformer_blocks[b].layer_norm1.shift = assign(
            gpt.transformer_blocks[b].layer_norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.transformer_blocks[b].layer_norm2.scale = assign(
            gpt.transformer_blocks[b].layer_norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.transformer_blocks[b].layer_norm2.shift = assign(
            gpt.transformer_blocks[b].layer_norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])