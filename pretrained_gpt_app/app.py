### 1. Setup ###
import gradio as gr
import time
import torch
import tiktoken

from model import (BASE_CONFIG,
                   GPT2Model,
                   replace_linear_with_lora,
                   generate_text,
                   text_to_token_ids,
                   token_ids_to_text,
                   )


### 2. Model preparation ###
bpe_tokenizer = tiktoken.get_encoding("gpt2")
model = GPT2Model(BASE_CONFIG)
replace_linear_with_lora(model, rank=16, alpha=16)
model.load_state_dict(torch.load("stable_training_with_lora_gpt2.pth", map_location="cpu", weights_only=True))
model = model.to("cpu")
model.eval()




### 3. Text generation function ###
def generate_answer(input_prompt,
                    #model,
                    max_new_tokens,
                    top_k,
                    temperature,
                    ):
  # start the timer
  start_time = time.time()
  torch.manual_seed(211)
  model.eval()
  token_ids = generate_text(
      model=model,
      input_batch=text_to_token_ids(input_prompt, bpe_tokenizer).to("cpu"),
      max_new_tokens=max_new_tokens,
      context_size=BASE_CONFIG["context_length"],
      top_k=top_k,
      temperature=temperature
  )
  # Calculate the prediction time
  pred_time = round(time.time() - start_time)

  return token_ids_to_text(token_ids, bpe_tokenizer), pred_time, "stable_training_with_lora_loss_plot.png"


### 4. Gradio app ###

# Create title, description and article strings
title = "GPT2 From Scratch"
description = "Pretrained GPT2 (124M) from scratch with LoRA."
article = "Learning LLM from scratch"

# Create the Gradio demo
demo = gr.Interface(fn=generate_answer, # mapping function from input to output
                    inputs=[
                        gr.Textbox(label="Input Text"),
                        gr.Slider(minimum=0, maximum=500, step=1, value=50, label="Choose number of generating tokens"),
                        gr.Slider(minimum=0, maximum=50, step=1, value=10, label="Choose top k"),
                        gr.Slider(minimum=0.0, maximum=5.0, step=0.1, value=1.5, label="Choose temperature"),
                        ],
                    outputs=[
                        gr.Textbox(label="Result"),
                        gr.Number(label="Time Taken (s)"),
                        gr.Image(label="Training Loss"),
                        ],
                    #examples=example_list,
                    title=title,
                    description=description,
                    article=article,
                    )

# Launch the demo!
demo.launch(debug=True, # print errors locally?
            share=True) # generate a publicly shareable URL?

