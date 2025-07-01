import torch
import math

from utils.helper_functions import  (calc_loss_batch,
                                     calc_loss_loader,
                                     generate_and_print_sample,
                                     calc_accuracy_loader,
                                     calc_classification_loss_loader)



def train_model_simple(model,
                       train_loader,
                       val_loader,
                       optimizer,
                       device,
                       num_epochs,
                       eval_freq,
                       eval_iter,
                       start_context,
                       tokenizer):

  # initialize lists to track losses and tokens seen
  train_losses = []
  val_losses = []
  track_tokens_seen = []
  token_seen = 0
  global_step = -1

  # main training loop - iterate over training epochs
  for epoch in range(num_epochs):
    # set model to training mode
    model.train()

    # iterate over batches in each training epoch
    for input_batch, target_batch in train_loader:
      # reset loss gradients from previous batch iteration
      optimizer.zero_grad()

      # calculate loss on current batch
      loss = calc_loss_batch(input_batch,
                             target_batch,
                             model,
                             device)

      # backward pass to calculate loss gradients
      loss.backward()

      # update model weights using loss gradients
      optimizer.step()
      token_seen += input_batch.numel()
      global_step += 1

      # optional evaluation step
      if global_step % eval_freq == 0:
        train_loss, val_loss = evaluate_model(model,
                                              train_loader,
                                              val_loader,
                                              device,
                                              eval_iter)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        track_tokens_seen.append(token_seen)
        # print training and evaluation set loss
        print(f"Ep {epoch+1} (Step {global_step:06d}): "
              f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

    # generative sample text for visual inspection
    generate_and_print_sample(model,
                              tokenizer,
                              device,
                              start_context)

  return train_losses, val_losses, track_tokens_seen


def evaluate_model(model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter):
  # set model to evaluation mode
  model.eval()
  with torch.no_grad():
    # calculate loss
    train_loss = calc_loss_loader(train_loader,
                                  model,
                                  device,
                                  num_batches=eval_iter)
    val_loss = calc_loss_loader(val_loader,
                                model,
                                device,
                                num_batches=eval_iter)

  # set model back to training mode
  model.train()
  return train_loss, val_loss


def train_classifier_simple(model,
                            train_loader,
                            val_loader,
                            optimizer,
                            device,
                            num_epochs,
                            eval_freq,
                            eval_iter # number of batches to evaluate from
                            ):
  train_losses, val_losses = [], []
  train_accuracies, val_accuracies = [], []
  examples_seen, global_step = 0, -1

  for epoch in range(num_epochs):
    model.train()

    for input_batch, label_batch in train_loader:
      optimizer.zero_grad()
      loss = calc_loss_batch(input_batch, label_batch, model, device)
      loss.backward()
      optimizer.step()
      examples_seen += input_batch.shape[0] # New: track examples instead of tokens
      global_step += 1


      # optional evaluation step
      if global_step % eval_freq == 0:
        train_loss, val_loss = evaluate_model(model,
                                              train_loader,
                                              val_loader,
                                              device,
                                              eval_iter)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

    # calculate accuracy after each epoch
    train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
    val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
    print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

  return train_losses, val_losses, train_accuracies, val_accuracies, examples_seen



def evaluate_classification_model(model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter):
  # set model to evaluation mode
  model.eval()
  with torch.no_grad():
    # calculate loss
    train_loss = calc_classification_loss_loader(train_loader,
                                  model,
                                  device,
                                  num_batches=eval_iter)
    val_loss = calc_classification_loss_loader(val_loader,
                                model,
                                device,
                                num_batches=eval_iter)

  # set model back to training mode
  model.train()
  return train_loss, val_loss


ORIG_BOOK_VERSION = False
def train_model(model,
                train_loader,
                val_loader,
                optimizer,
                device,
                n_epochs,
                eval_freq,
                eval_iter,
                start_context,
                tokenizer,
                warmup_steps,
                initial_lr=3e-05,
                min_lr=1e-6):

  train_losses, val_losses = [], []
  track_tokens_seen, track_lrs = [], []

  token_seen = 0
  global_step = -1

  # retrieve the maximum/peak learning rate from the optimizer
  peak_lr = optimizer.param_groups[0]["lr"]

  # calculate the total number of iterations in the training process
  total_training_steps = len(train_loader) * n_epochs

  # calculate the learning rate increment during the warmup phase
  lr_increment = (peak_lr - initial_lr) / warmup_steps

  for epoch in range(n_epochs):
    model.train()
    for input_batch, target_batch in train_loader:
      optimizer.zero_grad()
      global_step += 1

      # adjust the learning rate based on the current phase (warmup or cosine)
      if global_step < warmup_steps:
        lr = initial_lr + global_step * lr_increment
      else:
        # cosine annealing after warmup
        progress = ((global_step - warmup_steps) /
                    (total_training_steps - warmup_steps))
        lr = (min_lr +
         (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress)))

      # apply the calculated learning rate to the optimizer
      for param_group in optimizer.param_groups:
        param_group["lr"] = lr
      track_lrs.append(lr) # store the current learning rate

      # calculate and backpropagate the loss
      loss = calc_loss_batch(input_batch,
                             target_batch,
                             model,
                             device)
      loss.backward()

      # apply gradient clipping after the warmup phase to avoid exploding gradients
      if ORIG_BOOK_VERSION:
        if global_step > warmup_steps:
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      else:
        # the book originally used global_step > warmup_steps, which led to a skipped clipping step after warmup
        if global_step >= warmup_steps:
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

      optimizer.step()
      token_seen += input_batch.numel()

      # periodically evaluate the model on the training and validation sets
      if global_step % eval_freq == 0:
        train_loss, val_loss = evaluate_model(model,
                                              train_loader,
                                              val_loader,
                                              device,
                                              eval_iter)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        track_tokens_seen.append(token_seen)
        # print the current losses
        print(f"Ep {epoch+1} (Iter {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")

    # generate and print a sample from the model to monitor progess
    generate_and_print_sample(
        model=model,
        tokenizer=tokenizer,
        device=device,
        start_context=start_context
    )

  return train_losses, val_losses, track_tokens_seen, track_lrs