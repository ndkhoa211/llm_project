### 1. Setup ###
import gradio as gr
import time
import torch
import tiktoken
from huggingface_hub import hf_hub_download


from model import (BASE_CONFIG,
                   GPT2Model,
                   replace_linear_with_lora,
                   generate_text,
                   text_to_token_ids,
                   token_ids_to_text,
                   format_input,
                   )


### 2. Model preparation ###
bpe_tokenizer = tiktoken.get_encoding("gpt2")
model = GPT2Model(BASE_CONFIG)
replace_linear_with_lora(model, rank=16, alpha=16)
#model.load_state_dict(torch.load("instruction_finetune_gpt2.pth", map_location="cpu", weights_only=True))
repo_id = "ndk211/intruction_finetune_gpt2_774m"
filename = "instruction_finetune_gpt2.pth"
local_path = hf_hub_download(repo_id=repo_id, filename=filename)

state_dict = torch.load(local_path, map_location="cpu")
model.load_state_dict(state_dict, strict=False)

model = model.to("cpu")
model.eval()




### 3. Text generation function ###
def generate_answer(instruction: str = "", input: str = ""):
    # start the timer
    start_time = time.time()

    torch.manual_seed(211)
    model.eval()

    input_prompt = {'instruction': instruction, 'input': input}
    input_text = format_input(input_prompt)

    token_ids = generate_text(
        model=model,
        input_batch=text_to_token_ids(input_text, bpe_tokenizer).to("cpu"),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )

    generated_text = token_ids_to_text(token_ids, bpe_tokenizer)
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )

    # Calculate the prediction time
    pred_time = round(time.time() - start_time)

    # print("Output text:\n", token_ids_to_text(token_ids, bpe_tokenizer))
    return response_text.strip(), pred_time, "instruction_finetune_gpt2_loss_plot.png"


### 4. Gradio app ###

# Create title, description and article strings
title = "GPT2 From Scratch"
description = "Instruction Finetune GPT2 (774M)."
article = "Learning LLM from scratch"


# example list
example_list = [
    ["Generate a sentence using the word 'curiousity'.", ""],
    ["What is the formula for area of a circle?", ""],
    ["How to train a dragon?", ""],
    ["Convert this sentence to passive voice", "The chef cooked a tasty meal."],
    ["What is the capital of Taiwan?", ""],
    ["List 3 antonyms for 'intelligent'.", ""],
    ["what is a language model?", ""],
    ["Identify the correct spelling of the following word.", "ingelligent"],
]


# Create the Gradio demo
demo = gr.Interface(fn=generate_answer, # mapping function from input to output
                    inputs=[
                        gr.Textbox(label="Instruction"),
                        gr.Textbox(label="Input",
                                   placeholder="can be left empty"),
                        ],
                    outputs=[
                        gr.Textbox(label="Result"),
                        gr.Number(label="Time Taken (s)"),
                        gr.Image(label="Training Loss"),
                        ],
                    title=title,
                    description=description,
                    article=article,
                    examples=example_list,
                    #examples_per_page=3,
                    )

# Launch the demo!
demo.launch(debug=True, # print errors locally?
            share=True) # generate a publically shareable URL?