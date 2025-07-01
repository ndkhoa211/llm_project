import torch
import numpy as np


def text_to_token_ids(text, tokenizer):
  encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

  # turn the list of token IDs into tensor with batch dimension
  encoded_tensor = torch.tensor(encoded).unsqueeze(0)
  return encoded_tensor

def token_ids_to_text(encoded_tensor, tokenizer):
  # turn tensor without batch dimension to list
  token_ids = encoded_tensor.squeeze(0).tolist()
  text = tokenizer.decode(token_ids)
  return text



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



def generate_and_print_sample(model,
                              tokenizer,
                              device,
                              start_context):
  # set model to evaluation mode
  model.eval()
  context_size = model.position_emb.weight.shape[0]
  encoded = text_to_token_ids(start_context, tokenizer).to(device)
  with torch.no_grad():
    token_ids = generate_text_simple(model=model,
                                     input_batch=encoded,
                                     max_new_tokens=50,
                                     context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " ")) # compact print format
  # set model back to training mode
  model.train()




def generate_text(model,
                  input_batch,
                  max_new_tokens,
                  context_size,
                  temperature=0.0,
                  top_k=None,
                  eos_id=None):
  for _ in range(max_new_tokens):
    # crop current context if it exceeds the supported context_size
    crop_input_batch = input_batch[:, -context_size:]

    # predict next token
    with torch.no_grad():
      logits = model(crop_input_batch)

    # consider only logits of the last token
    logits = logits[:, -1, :] # (batch, n_tokens, vocab_size) -> (batch, vocab_size)

    # NEW: filter logits with top_k sampling
    if top_k is not None:
      # keep only top_k values
      top_logits, _ = torch.topk(logits, top_k)
      min_val = top_logits[:, -1] # min value among the top_k values
      # all values other than top_k values will be set to -inf
      logits = torch.where(logits < min_val,
                           torch.tensor(-torch.inf).to(logits.device),
                           logits)

    # NEW: temperature scaling
    if temperature > 0.0:
      logits = logits / temperature

      probas = torch.softmax(logits, dim=-1) # (batch, vocab_size)
      predicted_tokens = torch.multinomial(probas, num_samples=1) # (batch, 1)

    else: # same as before
      #probas = torch.softmax(logits, dim=-1) # (batch, vocab_size)
      predicted_tokens = torch.argmax(logits, dim=-1, keepdim=True) # (batch, 1)

    if predicted_tokens == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

    # update input_batch (append predicted tokens to the sequences)
    input_batch = torch.cat([input_batch, predicted_tokens], dim=1) # [batch, num_tokens+1]

  return input_batch





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



def calc_accuracy_loader(data_loader,
                         model,
                         device,
                         num_batches=None):
  model.eval()
  correct_predictions, num_examples = 0, 0

  if num_batches is None:
    num_batches = len(data_loader)
  else:
    num_batches = min(num_batches, len(data_loader))

  for batch_id, (input_batch, label_batch) in enumerate(data_loader):
    if batch_id < num_batches:
      input_batch, label_batch = input_batch.to(device), label_batch.to(device)

      with torch.no_grad():
        logits = model(input_batch)[:, -1, :] # logits of the last output token
      predicted_labels = torch.argmax(logits, dim=-1)

      num_examples += predicted_labels.shape[0]
      correct_predictions += (predicted_labels == label_batch).sum().item()
    else:
      break
  return correct_predictions / num_examples


def calc_classification_loss_batch(input_batch,
                    label_batch,
                    model,
                    device):
  input_batch, label_batch = input_batch.to(device), label_batch.to(device)
  logits = model(input_batch)[:, -1, :] # logits of the last output token
  loss = torch.nn.functional.cross_entropy(logits, label_batch)
  return loss


# same as pretraining
def calc_classification_loss_loader(data_loader,
                     model,
                     device,
                     num_batches=None):
  total_loss = 0.0
  if len(data_loader) == 0:
    return float("nan")
  elif num_batches is None:
    num_batches = len(data_loader)
  else:
    num_batches = min(num_batches, len(data_loader))

  for batch_id, (input_batch, label_batch) in enumerate(data_loader):
    if batch_id < num_batches:
      loss = calc_classification_loss_batch(input_batch, label_batch, model, device)
      total_loss += loss.item()
    else:
      break
  return total_loss / num_batches


def classify_review(text,
                    model,
                    tokenizer,
                    device,
                    max_length=None,
                    pad_token_id=50256):
  model.eval()

  # input preprocessing
  input_ids = tokenizer.encode(text)
  supported_context_length = model.position_emb.weight.shape[0]
  input_ids = input_ids[:min(max_length, supported_context_length)]
  input_ids += [pad_token_id] * (max_length  - len(input_ids))
  input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

  # model inference
  with torch.no_grad():
    logits = model(input_tensor)[:, -1, :] # logits of the last output token
  predicted_label = torch.argmax(logits, dim=-1).item()

  # return the classifier result
  return "spam" if predicted_label == 1 else "not spam"