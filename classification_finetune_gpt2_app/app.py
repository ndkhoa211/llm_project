### 1. Setup ###
import gradio as gr
import time
import torch
import tiktoken

from model import (BASE_CONFIG,
                   GPT2Model,
                   classify_review,
                   )


### 2. Model preparation ###
bpe_tokenizer = tiktoken.get_encoding("gpt2")
model = GPT2Model(BASE_CONFIG)
num_classes = 2
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
model.load_state_dict(torch.load("classification_finetune_gpt2.pth", map_location="cpu", weights_only=True))
model = model.to("cpu")


### 3. Text generation function ###
def generate_answer(input_prompt):

  # start the timer
  start_time = time.time()

  torch.manual_seed(211)

  model.eval()

  prediction = classify_review(input_prompt,
                              model,
                              bpe_tokenizer,
                              "cpu",
                              max_length=120)

  # Calculate the prediction time
  pred_time = round(time.time() - start_time, 5)

  # print("Output text:\n", token_ids_to_text(token_ids, bpe_tokenizer))
  return prediction, pred_time


### 4. Gradio app ###

examples = [
    ["Well done ENGLAND! Get the official poly ringtone or colour flag on yer mob"
     "ile! text TONE or FLAG to 84199 NOW! Opt-out txt ENG STOP. Box39822 W111WX £1.50"],  # spam
    ["Hi its in durban are you still on this number"],  # ham
    ["Compliments to you. Was away from the system. How your side."],  # ham
    ["Can ü call me at 10:10 to make sure dat i've woken up..."],  # ham
    ["tddnewsletter@emc1.co.uk (More games from TheDailyDraw) Dear Helen,"
     " Dozens of Free Games - with great prizesWith.."],  # spam

]

with gr.Blocks() as demo:
    gr.Markdown("# GPT2 From Scratch")

    # tab 1
    with gr.Tab("Instruction Finetune GPT2 (124M)"):
        with gr.Row():
            inp = gr.Textbox(label="Input Text", placeholder="Type here...")
            out = [gr.Textbox(label="Classification", value="", interactive=False),
                   gr.Number(label="Time Taken (s)"),
                   ]
        inp.change(fn=generate_answer, inputs=inp, outputs=out)

        with gr.Column():
            gr.Examples(examples=examples, inputs=inp, label="📚 Examples")

    # tab 2
    with gr.Tab("Graphs:"):
        with gr.Row():
            gr.Image(value="classification_finetune_loss_gpt2.png", label="Loss Curve", interactive=False)
            gr.Image(value="classification_finetune_accuracy_gpt2.png", label="Loss Curve", interactive=False)

# Launch the demo!
demo.launch(debug=True,  # print errors locally?
            share=True)  # generate a publically shareable URL?

