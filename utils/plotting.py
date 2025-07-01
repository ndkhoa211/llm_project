import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_losses(epoch_seen,
                tokens_seen,
                train_losses,
                val_losses,
                fig_path="loss_plot.pdf"):
  """Plot training and validation loss."""

  fig, ax1 = plt.subplots(figsize=(5, 3))

  # plot training and validation loss against epochs
  ax1.plot(epoch_seen, train_losses, label="Training Loss")
  ax1.plot(epoch_seen, val_losses, linestyle="-.", label="Validation Loss")
  ax1.set_xlabel("Epoch")
  ax1.set_ylabel("Loss")
  ax1.legend(loc="upper right")
  ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # only show integer labels on x-axis

  # create a second x-axis for token seen
  ax2 = ax1.twiny() # create a second x-axis that shares the same y-axis
  ax2.plot(tokens_seen, train_losses, alpha=0) # invisible plot for aligning ticks
  ax2.set_xlabel("Tokens Seen")

  fig.tight_layout() # adjust layout to make room
  plt.savefig(fig_path)
  plt.show()

def plot_values(epochs_seen,
                examples_seen,
                train_values,
                val_values,
                label="loss"):
  fig, ax1 = plt.subplots(figsize=(5, 3))

  # Plot training and validation loss against epochs
  ax1.plot(epochs_seen, train_values, label=f"Training {label}")
  ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
  ax1.set_xlabel("Epochs")
  ax1.set_ylabel(label.capitalize())
  ax1.legend()

  # Create a second x-axis for examples seen
  ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
  ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
  ax2.set_xlabel("Examples seen")

  fig.tight_layout()  # Adjust layout to make room
  plt.savefig(f"{label}-plot.pdf")
  plt.show()


  def run_chatgpt(prompt, client, model=MODEL):
    response = client.chat.completions.create(
      model=model,
      messages=[{"role": "user", "content": prompt}],
      temperature=0.0,
      seed=123,
    )
    return response.choices[0].message.content



  def generate_model_scores(json_data, json_key, model=MODEL):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
      prompt = (
        f"Given the input `{format_input(entry)}` "
        f"and correct output `{entry['output']}`, "
        f"score the model response `{entry[json_key]}`"
        f" on a scale from 0 to 100, where 100 is the best score. "
        f"Respond with the integer number only."
      )
      score = run_chatgpt(prompt, client, model)
      try:
        scores.append(int(score))
      except ValueError:
        print(f"Could not convert score: {score}")
        continue

    return scores