# YaRN
[YaRN: Efficient Context Window Extension of Large Language Models](paper/yarn.pdf)

This repo contains the code and data for the YaRN context window extension method.

```
Awaiting arXiv announcement, citation will go here!
```

## Models

We publish 7B and 13B variants of [LLaMA 2](https://about.fb.com/news/2023/07/llama-2/) fine-tuned with YaRN at 64K and 128K context window length.
They are available under the LLaMA 2 license on 🤗 Hugging Face.

| Size | Context | Link   |
| ---: | ------: | :----- |
|   7B |     64K | [NousResearch/Yarn-Llama-2-7b-64k](https://huggingface.co/NousResearch/Yarn-Llama-2-7b-64k)     |
|   7B |    128K | [NousResearch/Yarn-Llama-2-7b-128k](https://huggingface.co/NousResearch/Yarn-Llama-2-7b-128k)   |
|  13B |     64K | [NousResearch/Yarn-Llama-2-13b-64k](https://huggingface.co/NousResearch/Yarn-Llama-2-13b-64k)   |
|  13B |    128K | [NousResearch/Yarn-Llama-2-13b-128k](https://huggingface.co/NousResearch/Yarn-Llama-2-13b-128k) |

## Reproduction

We strongly believe in open science, and thus publish all code and data to reproduce the results in our paper.
To reproduce, clone the repository and perform a local installation.

```python
git clone https://github.com/jquesnelle/yarn
cd yarn
pip install -e .
```

### Training

To train the models, run `accelerate config` and enable DeepSpeed acceleration. `deepspeed/zero3.json` was the configuration file used for training.

```sh
# ./train.sh
```

The tokenized training data is available on [Hugging Face](https://huggingface.co/datasets/emozilla/pg_books-tokenized-bos-eos-chunked-65536) and was derived from the [pg19](https://huggingface.co/datasets/emozilla/pg19) dataset.

### Evaluation

To reproduce the evaluations, install [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with `pip install git+https://github.com/EleutherAI/lm-evaluation-harness` and then run the two provided scripts.

```sh
# ./eval.sh
# ./eval-harness.sh
```
