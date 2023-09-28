#!/bin/bash

accelerate launch finetune.py \
    --wandb yarn \
    --output-dir output/yarn-7b-64k \
    --model NousResearch/Llama-2-7b-hf

accelerate launch finetune.py \
    --wandb yarn \
    --output-dir output/yarn-7b-128k \
    --model output/yarn-7b-64k \
    --max-train-steps 200 \
    --scaling-factor 32 \
    --seed 31337

accelerate launch finetune.py \
    --wandb yarn \
    --model NousResearch/Llama-2-13b-hf \
    --output-dir output/yarn-13b-64k

accelerate launch finetune.py \
    --wandb yarn \
    --output-dir output/yarn-13b-128k \
    --model output/yarn-13b-64k \
    --max-train-steps 200 \
    --scaling-factor 32 \
    --seed 31337

# ablations

python3 truncate.py 8192 output/truncated-8k

accelerate launch finetune.py \
    --wandb yarn \
    --output-dir output/linear-7b-8k \
    --model NousResearch/Llama-2-7b-hf \
    --scaling-type linear \
    --scaling-factor 2 \
    --dataset output/truncated-8k

accelerate launch finetune.py \
    --wandb yarn \
    --output-dir output/ntk-7b-8k \
    --model NousResearch/Llama-2-7b-hf \
    --scaling-type ntk \
    --scaling-factor 1 \
    --rope-theta 20000 \
    --dataset output/truncated-8k

accelerate launch finetune.py \
    --wandb yarn \
    --output-dir output/yarn-7b-8k \
    --model NousResearch/Llama-2-7b-hf \
    --scaling-factor 2 \
    --dataset output/truncated-8k