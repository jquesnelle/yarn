#!/bin/bash

# run `accelerate config` first. pass --deepspeed to finetune.py if using DeepSpeed

accelerate launch finetune.py \
    --output-dir output/yarn-7b-64k \
    --model NousResearch/Llama-2-7b-hf

accelerate launch finetune.py \
    --output-dir output/yarn-7b-128k \
    --model output/yarn-7b-64k \
    --max-train-steps 200 \
    --scaling-factor 32 \
    --seed 31337

accelerate launch finetune.py \
    --model NousResearch/Llama-2-13b-hf \
    --output-dir output/yarn-13b-64k

accelerate launch finetune.py \
    --output-dir output/yarn-13b-128k \
    --model output/yarn-13b-64k \
    --max-train-steps 200 \
    --scaling-factor 32 \
    --seed 31337

accelerate launch finetune.py \
    --model NousResearch/Llama-2-70b-hf \
    --output-dir output/yarn-70b-32k \
    --learning-rate 0.00001 \
    --lr-schedule constant \
    --scaling-factor 8 \
    --dataset emozilla/yarn-train-tokenized-8k-llama

# ablations

python3 truncate.py 8192 output/truncated-8k

accelerate launch finetune.py \
    --output-dir output/linear-7b-8k \
    --model NousResearch/Llama-2-7b-hf \
    --scaling-type linear \
    --scaling-factor 2 \
    --dataset output/truncated-8k

accelerate launch finetune.py \
    --output-dir output/ntk-7b-8k \
    --model NousResearch/Llama-2-7b-hf \
    --scaling-type ntk \
    --scaling-factor 1 \
    --rope-theta 20000 \
    --dataset output/truncated-8k

accelerate launch finetune.py \
    --output-dir output/yarn-7b-8k \
    --model NousResearch/Llama-2-7b-hf \
    --scaling-factor 2 \
    --dataset output/truncated-8k

# mistral

accelerate launch finetune.py \
    --output-dir output/yarn-mistral-7b-64k \
    --model mistralai/Mistral-7B-v0.1 \
    --architecture mistral \
    --scaling-factor 8 \
    --max-position-embeddings 16384 \
    --dataset emozilla/yarn-train-tokenized-16k-mistral \
    --sliding-window-attention-schedule 65536 \
    --lr-schedule constant \
    --learning-rate 0.000001 \
    --max-train-steps 1000

accelerate launch finetune.py \
    --output-dir output/yarn-mistral-7b-128k \
    --model output/yarn-mistral-7b-64k \
    --architecture mistral \
    --scaling-factor 16 \
    --max-position-embeddings 16384 \
    --dataset emozilla/yarn-train-tokenized-16k-mistral \
    --sliding-window-attention-schedule 131072 \
    --lr-schedule constant \
    --learning-rate 0.000001 \
    --max-train-steps 500 \
    --seed 31337