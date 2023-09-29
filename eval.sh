#!/bin/bash

# python eval/perplexity.py -m meta-llama/Llama-2-7b-hf --dataset pg19 --split test --feature text --save-tokenized output/pg19-test-tokenized
PG19="--tokenized output/pg19-test-tokenized"

# python eval/perplexity.py -m meta-llama/Llama-2-7b-hf --dataset tau/scrolls --subset gov_report --split test --feature input --save-tokenized output/govreport-test-tokenized
GOVREPORT="--tokenized output/govreport-test-tokenized --dataset-min-tokens 16384 --samples 50"

# python eval/perplexity.py -m meta-llama/Llama-2-7b-hf --dataset hoskinson-center/proof-pile --split test --feature text --save-tokenized output/proofpile-test-tokenized
PROOFPILE="--tokenized output/proofpile-test-tokenized --dataset-min-tokens 32768 --samples 50"
PROOFPILE_LONG_SMALL="--tokenized output/proofpile-test-tokenized --dataset-min-tokens 131072 --samples 10 --truncate"

CUSTOM="--custom-model-together"

python eval/perplexity.py \
    ${PROOFPILE_LONG_SMALL} ${CUSTOM} \
    --output-file data/proofpile-long-small.csv \
    --min-tokens 2048 --max-tokens 131072 --tokens-step 2048 --aggressive-memory \
    -m codellama/CodeLlama-13b-hf \
    -m output/yarn-13b-64k \
    -m output/yarn-13b-128k \
    -m togethercomputer/LLaMA-2-7B-32K \
    -m codellama/CodeLlama-7b-hf \
    -m output/yarn-7b-64k \
    -m output/yarn-7b-128k

python eval/perplexity.py \
    ${GOVREPORT} ${CUSTOM} \
    --output-file data/govreport.csv \
    --min-tokens 32768 --max-tokens 32768 \
    -m codellama/CodeLlama-13b-hf \
    -m output/yarn-13b-64k \
    -m output/yarn-13b-128k \
    -m togethercomputer/LLaMA-2-7B-32K \
    -m codellama/CodeLlama-7b-hf \
    -m output/yarn-7b-64k \
    -m output/yarn-7b-128k

python eval/perplexity.py \
    ${PROOFPILE_LONG_SMALL} ${CUSTOM} \
    --output-file data/proofpile-long-small-8k.csv \
    --min-tokens 2048 --max-tokens 16384 --tokens-step 2048 \
    -m output/yarn-7b-8k  \
    -m output/ntk-7b-8k \
    -m conceptofmind/LLongMA-2-7b