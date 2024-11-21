import os
import tempfile
import time
from argparse import ArgumentParser

import dac
import torch
from accelerate import Accelerator
from audiotools import AudioSignal
from model import Phonira
from pattern import DelayPattern
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5EncoderModel
from utils import collate_fn, load_webdataset, skip_small_samples

import wandb

args = ArgumentParser()
args.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Path to the dataset or the dataset name in huggingface Hub",
)
args.add_argument(
    "--split",
    type=str,
    required=True,
    help="Split of the dataset to use",
)
args.add_argument(
    "--column_code",
    type=str,
    required=True,
    help="Column name that contains the codebooks codes of the audio codec",
)
args.add_argument(
    "--column_prompt",
    type=str,
    required=True,
    help="Column name that contains the prompt of the audio",
)
args.add_argument(
    "--num_quantizers",
    type=int,
    default=9,
    help="Number of quantizers to use in the model",
)
args.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="Number of epochs to train the model",
)
args.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="Batch size",
)
args.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Gradient accumulation steps",
)
args.add_argument(
    "--lr",
    type=float,
    default=3e-4,
    help="Learning rate",
)
args.add_argument(
    "--betas",
    type=float,
    nargs=2,
    default=[0.9, 0.99],
    help="Adam optimizer betas",
)
args.add_argument(
    "--dataset_size",
    type=int,
    help="The size of the dataset to use",
)
args.add_argument(
    "--output_dir",
    type=str,
    default="results",
    help="Output directory to save the model checkpoints",
)
args.add_argument(
    "--project_name",
    type=str,
    default="phonira",
    help="Wandb project name",
)
args.add_argument(
    "--log_interval",
    type=int,
    default=100,
    help="Log interval",
)
args.add_argument(
    "--gradient_clip_val",
    type=float,
    default=0.5,
    help="Gradient clipping value",
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
    "--dropout",
    type=float,
    default=0.1,
    help="Dropout probability",
)
args.add_argument(
    "--model_path",
    type=str,
    help="Path to the model checkpoint to resume training",
)
args.add_argument(
    "--conditionning_model",
    type=str,
    default="google-t5/t5-small",
    help="The conditionning model to use",
)
args.add_argument(
    "--max_prompt_length",
    type=int,
    default=512,
    help="The maximum prompt length",
)
args = args.parse_args()

conditionning_model = T5EncoderModel.from_pretrained(args.conditionning_model)
tokenizer = AutoTokenizer.from_pretrained(args.conditionning_model)

dataset = load_webdataset(
    args.dataset,
    args.split,
    map_func=skip_small_samples(args.column_code, max(args.num_quantizers, 20)),
)


pattern_manager = DelayPattern(args.padding_value)

training_dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=os.cpu_count(),
    collate_fn=collate_fn(
        args.num_quantizers,
        args.column_code,
        args.column_prompt,
        conditionning_model,
        tokenizer,
        pattern_manager,
        args.padding_value,
        args.max_prompt_length,
    ),
)

model = Phonira(
    num_quantizers=args.num_quantizers,
    codebook_size=args.codebook_size,
    hidden_size=args.hidden_size,
    depth=args.depth,
    padding_token=args.padding_value,
    dropout_p=args.dropout,
    delay_pattern=pattern_manager,
    proj_dim=conditionning_model.config.d_model,
)

# weight initialization
for p in model.parameters():
    if p.dim() > 1:
        torch.nn.init.normal_(p, mean=0, std=0.02)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, eps=1e-8)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=(
        args.epochs
        * args.dataset_size
        // (args.gradient_accumulation_steps * args.batch_size)
    ),
)

accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps, log_with="wandb"
)
accelerator.init_trackers(project_name=args.project_name, config=vars(args))


model, optimizer, scheduler, training_dataloader = accelerator.prepare(
    model, optimizer, scheduler, training_dataloader
)

accelerator.register_for_checkpointing(scheduler)
if args.model_path:
    accelerator.load_state(args.model_path)

model.train()

model_path = dac.utils.download(model_type="44khz")
model_dac = dac.DAC.load(model_path)
model_dac.to(accelerator.device)

start_traing_time = time.time()
avg_loss = 0

for epoch in range(args.epochs):
    for i, batch in enumerate(training_dataloader):
        # if dataset size is provided, break the loop when the dataset size is reached
        if (i + 1) * args.batch_size > args.dataset_size:
            break

        x, padding_mask, prepend_embeds, prompt_input = batch
        prepend_mask = prompt_input["attention_mask"].bool()

        with accelerator.accumulate(model):
            _, loss = model(
                x,
                prepend_embeds,
                padding_mask=padding_mask,
                prepend_mask=prepend_mask,
                training=True,
            )
            accelerator.backward(loss)

            avg_loss += loss.item()
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.gradient_clip_val)
                scheduler.step()

                avg_loss /= args.gradient_accumulation_steps
                accelerator.print(
                    f"Epoch {epoch}, step {i//args.gradient_accumulation_steps}, loss: {avg_loss} ({round((i+1)/(time.time() - start_traing_time), 2)} s/step), lr: {round(scheduler.get_last_lr()[0], 7)}",
                    end="\r",
                )
                avg_loss = 0

            optimizer.step()
            optimizer.zero_grad()

        # logging section
        accelerator.log({"train_loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
        if (i / args.gradient_accumulation_steps) % args.log_interval == 0:
            accelerator.save_state(output_dir=args.output_dir)

            prepend_embed = prepend_embeds[0].unsqueeze(0)
            prepend_mask = prepend_mask[0].unsqueeze(0)

            prompt = tokenizer.decode(
                prompt_input["input_ids"][0], skip_special_tokens=True
            )

            with torch.no_grad():
                model.eval()
                audio = model.generate(
                    prepend_embed, prepend_mask, args.num_quantizers, 864
                )
                model.train()

                audio = model_dac.quantizer.from_codes(audio)[0]
                audio = model_dac.decode(audio).squeeze(1)

                audio = AudioSignal(audio.cpu(), sample_rate=model_dac.sample_rate)

                with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                    audio.write(f.name)
                    accelerator.log({"audio": wandb.Audio(f.name, caption=prompt)})

accelerator.end_training()
