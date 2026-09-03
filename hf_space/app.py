"""Gradio Space: a small live demo + callable API for the from-scratch
decoder transformer. Loads the model straight from its HF Hub repo via
trust_remote_code, exactly the way any other user would.

Deliberately minimal: this exists so someone can verify the model is real
and try it, not as a product. It reuses the model's own generate_simple()
method rather than reimplementing sampling logic here, so there's exactly
one place that decoding behavior lives.
"""

import os

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set to your HF Hub repo id once the trained model is pushed, e.g.
# "your-username/transformer-pretraining-from-scratch"
MODEL_REPO = os.environ.get("MODEL_REPO", "your-username/transformer-pretraining-from-scratch")

print(f"loading {MODEL_REPO} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_REPO, trust_remote_code=True)
model.eval()
print("loaded.")


def complete(prompt: str, max_new_tokens: int, temperature: float, top_k: int):
    if not prompt.strip():
        return "Enter a prompt first."
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        out_ids = model.generate_simple(
            input_ids,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
        )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


with gr.Blocks(title="Transformer Pretrained From Scratch") as demo:
    gr.Markdown(
        f"""
# Transformer Pretrained From Scratch

A ~57M-parameter decoder-only transformer, pretrained from scratch in PyTorch
(not fine-tuned from an existing checkpoint) — architecture, tokenizer, and
training loop all written from scratch. See the
[GitHub repo](https://github.com/Hariprashad-Ravikumar/transformer-pretraining-from-scratch)
and [model card]({MODEL_REPO}) for training details, measured perplexity/BPB,
and an honest Limitations section.

**This is a small research/learning-scale model, not a product** — expect
noticeably rougher completions than a large hosted model. That's expected and
documented, not a bug.
        """
    )
    with gr.Row():
        prompt = gr.Textbox(
            label="Prompt", value="The transformer architecture", lines=3
        )
    with gr.Row():
        max_new_tokens = gr.Slider(10, 200, value=60, step=10, label="Max new tokens")
        temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature")
        top_k = gr.Slider(1, 100, value=40, step=1, label="Top-k")
    output = gr.Textbox(label="Completion", lines=6)
    run_btn = gr.Button("Generate")
    run_btn.click(complete, inputs=[prompt, max_new_tokens, temperature, top_k], outputs=output)

if __name__ == "__main__":
    demo.launch()
