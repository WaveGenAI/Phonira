from argparse import ArgumentParser

import dac
import gradio as gr
import torch
from audiotools import AudioSignal
from model import Phonira
from pattern import DelayPattern
from safetensors.torch import load_model
from transformers import AutoTokenizer, T5EncoderModel

args = ArgumentParser()

args.add_argument(
    "--model_path",
    help="Resume training",
    type=str,
)
args.add_argument(
    "--num_quantizers",
    type=int,
    default=9,
    help="Number of quantizers to use in the model",
)
args.add_argument(
    "--padding_value",
    type=int,
    default=1024,
    help="Padding value for the collate function",
)
args.add_argument(
    "--codebook_size",
    type=int,
    default=1024,
    help="The number of codebooks of the audio codec",
)
args.add_argument(
    "--hidden_size",
    type=int,
    default=512,
    help="The hidden size of the model",
)
args.add_argument(
    "--depth",
    type=int,
    default=6,
    help="The depth of the model",
)
args.add_argument(
    "--num_heads",
    type=int,
    default=8,
    help="The number of heads in the model",
)
args.add_argument(
    "--conditionning_model",
    type=str,
    default="google-t5/t5-small",
    help="The conditionning model to use",
)
args.add_argument(
    "--share",
    action="store_true",
    help="Share the application",
)
args = args.parse_args()

conditionning_model = T5EncoderModel.from_pretrained(args.conditionning_model)
tokenizer = AutoTokenizer.from_pretrained(args.conditionning_model)

pattern_manager = DelayPattern(args.padding_value)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Phonira(
    num_quantizers=args.num_quantizers,
    codebook_size=args.codebook_size,
    hidden_size=args.hidden_size,
    depth=args.depth,
    padding_token=args.padding_value,
    dropout_p=0.0,
    delay_pattern=pattern_manager,
    num_heads=args.num_heads,
    proj_dim=conditionning_model.config.d_model,
)

# Load the model
load_model(model, args.model_path)
model.eval()

model_path = dac.utils.download(model_type="44khz")
model_dac = dac.DAC.load(model_path)

model_dac = model_dac.to(device)
model.to(device)


@torch.no_grad()
def generate(
    prompt: str,
    duration: int,
    temperature: float = 1.0,
    top_k: int = 150,
    guidance_scale: float = 3.0,
):
    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    prepend_embed = conditionning_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    ).last_hidden_state

    prepend_embed = prepend_embed[0].unsqueeze(0)
    prepend_mask = inputs["attention_mask"][0].unsqueeze(0).bool()

    audio = model.generate(
        prepend_embed.to(device),
        prepend_mask.to(device),
        args.num_quantizers,
        duration,
        temperature=temperature,
        top_k=top_k,
        guidance_scale=guidance_scale,
    )

    audio = model_dac.quantizer.from_codes(audio)[0]
    audio = model_dac.decode(audio).squeeze(1)

    audio_signal = AudioSignal(audio.cpu(), sample_rate=model_dac.sample_rate)

    return audio_signal.audio_data


def gradio_generate(
    prompt,
    duration,
    temperature: float = 1.0,
    top_k: int = 150,
    guidance_scale: float = 3.0,
):
    audio = generate(prompt, duration, temperature, top_k, guidance_scale)
    sample_rate = model_dac.sample_rate
    return (sample_rate, audio.numpy())


demo = gr.Interface(
    fn=lambda prompt, duration, temperature, top_k, guidance_scale: gradio_generate(
        prompt, duration * 86, temperature, top_k, guidance_scale
    ),
    inputs=[
        gr.Textbox(lines=3, placeholder="Enter your prompt here..."),
        gr.Slider(
            minimum=1, maximum=30, value=10, step=1, label="Audio Duration (seconds)"
        ),
        gr.Slider(minimum=0.1, maximum=2.0, value=1, step=0.1, label="Temperature"),
        gr.Slider(minimum=1, maximum=300, value=150, step=1, label="Top K"),
        gr.Slider(
            minimum=0.1, maximum=10.0, value=3.0, step=0.1, label="Guidance Scale"
        ),
    ],
    outputs=gr.Audio(type="numpy", label="Generated audio"),
    title="WaveAI Music Generator",
    description="Enter a text prompt and select the audio duration to generate a matching music clip.",
)


if __name__ == "__main__":
    demo.launch(share=args.share)
