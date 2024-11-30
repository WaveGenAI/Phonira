# Phonira
An audio model based on Soundstorm.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
accelerate launch phonira/trainer.py  --dataset /media/works/data/data/ --split train --column_code codes.npy --column_prompt prompt.txt --dataset_size 200000 --batch_size 1 --gradient_accumulation_steps 32 --depth 16 --hidden_size 1024 --num_heads 16 --dropout 0
```
