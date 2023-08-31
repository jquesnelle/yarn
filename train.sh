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
    --yarn-factor 32 \
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
    --yarn-factor 32 \
    --seed 31337