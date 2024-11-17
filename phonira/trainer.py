import os
import time
from argparse import ArgumentParser

import dac
import torch
from accelerate import Accelerator
from audiotools import AudioSignal
from einops import rearrange
from model import Phonira
from torch.utils.data import DataLoader
from utils import collate_fn, load_webdataset, skip_small_samples

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
    default=1000,
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
    default=1025,
    help="The number of codebooks of the audio codec",
)
args.add_argument(
    "--hidden_size",
    type=int,
    default=1024,
    help="The hidden size of the model",
)
args = args.parse_args()

dataset = load_webdataset(
    args.dataset,
    args.split,
    map_func=skip_small_samples(args.column_code, max(args.num_quantizers, 20)),
)

training_dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=os.cpu_count(),
    collate_fn=collate_fn(args.num_quantizers, args.column_code, args.padding_value),
)


model = Phonira(
    num_quantizers=args.num_quantizers,
    codebook_size=args.codebook_size,
    hidden_size=args.hidden_size,
)

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


model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, training_dataloader, scheduler
)

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

        x, padding_mask = batch

        with accelerator.accumulate(model):
            _, loss = model(x, training=True)
            accelerator.backward(loss)

            avg_loss += loss.item()
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.gradient_clip_val)
                scheduler.step()

                avg_loss /= args.gradient_accumulation_steps
                accelerator.print(
                    f"Epoch {epoch}, step {i}, loss: {avg_loss} ({round((i+1)/(time.time() - start_traing_time), 2)} s/step), lr: {round(scheduler.get_last_lr()[0], 7)}",
                    end="\r",
                )
                avg_loss = 0

            optimizer.step()
            optimizer.zero_grad()

        # logging section
        accelerator.log({"train_loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
        if i % args.log_interval == 0:
            accelerator.save_state(output_dir=args.output_dir)

accelerator.end_training()
