# YaRN

This repo contains the code and data for the YaRN context window extension method.

## Paper

Paper (ICLR 2024): [YaRN: Efficient Context Window Extension of Large Language Models](https://openreview.net/forum?id=wHBfxhZu1u)  
Old Preprint [(arXiv)](https://arxiv.org/abs/2309.00071)

## Models

### LLaMA

We publish variants of [Llama 2](https://about.fb.com/news/2023/07/llama-2/) fine-tuned with YaRN at 32K, 64K and 128K context window length.
They are available under the Llama 2 license on 🤗 Hugging Face.

| Size | Context | Link   |
| ---: | ------: | :----- |
|   7B |     64K | [NousResearch/Yarn-Llama-2-7b-64k](https://huggingface.co/NousResearch/Yarn-Llama-2-7b-64k)     |
|   7B |    128K | [NousResearch/Yarn-Llama-2-7b-128k](https://huggingface.co/NousResearch/Yarn-Llama-2-7b-128k)   |
|  13B |     64K | [NousResearch/Yarn-Llama-2-13b-64k](https://huggingface.co/NousResearch/Yarn-Llama-2-13b-64k)   |
|  13B |    128K | [NousResearch/Yarn-Llama-2-13b-128k](https://huggingface.co/NousResearch/Yarn-Llama-2-13b-128k) |
|  70B |     32K | [NousResearch/Yarn-Llama-2-70b-32k](https://huggingface.co/NousResearch/Yarn-Llama-2-70b-32k)   |

In addition, we also publish 8K context window versions of Llama 2 7B fine-tuned with [NTK-aware](https://huggingface.co/emozilla/NTK-Llama-2-7b-8k) and [YaRN](https://huggingface.co/emozilla/Yarn-Llama-2-7b-8k) (Table 1 in the conference paper).

### Mistral

With the release of v2 of our paper we are also publishing 64K and 128K variants of [Mistral 7B v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1).

| Size | Context | Link   |
| ---: | ------: | :----- |
|   7B |     64K | [NousResearch/Yarn-Mistral-7b-64k](https://huggingface.co/NousResearch/Yarn-Mistral-7b-64k)     |
|   7B |    128K | [NousResearch/Yarn-Mistral-7b-128k](https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k)   |

### SOLAR

The [SOLAR 10.7B v1.0](https://huggingface.co/upstage/SOLAR-10.7B-v1.0) model utilizes [depth-up scaling](https://arxiv.org/abs/2312.15166) to add layers to [Mistral 7B v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1), which may potentially improve long context performance on a per-parameter basis.
We publish 32K and 64K variants.

|    Size | Context | Link   |
| ------: | ------: | :----- |
|   10.7B |     32K | [NousResearch/Yarn-Solar-10b-32k](https://huggingface.co/NousResearch/Yarn-Solar-10b-32k)   |
|   10.7B |     64K | [NousResearch/Yarn-Solar-10b-64k](https://huggingface.co/NousResearch/Yarn-Solar-10b-64k)   |

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

The tokenized training data is available on [🤗Hugging Face](https://huggingface.co/datasets/emozilla/pg_books-tokenized-bos-eos-chunked-65536) and was derived from the [pg19](https://huggingface.co/datasets/emozilla/pg19) dataset.
For the Mistral models, a mix of the pretrain and fine-tune splits of [Long-Data-Collections](https://huggingface.co/datasets/togethercomputer/Long-Data-Collections) was used and the tokenized dataset is also available on [🤗Hugging Face](https://huggingface.co/datasets/emozilla/yarn-train-tokenized-16k-mistral).

Here is a more dedicated fast try for beginners, take Llama-2-7b-8k as example, it may need 4 hours on 4xA100:
```sh
# **Step1.** Accelerate config
$ accelerate config
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
This machine                                                                                                                                                            
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Which type of machine are you using?                                                                                                                                    
multi-GPU                                                                                                                                                               
How many different machines will you use (use more than 1 for multi-node training)? [1]:                                                                                
Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]:                                          
Do you wish to optimize your script with torch dynamo?[yes/NO]:                                                                                                         
Do you want to use DeepSpeed? [yes/NO]: yes                                                                                                                             
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: yes                                                                                                 
Please enter the path to the json DeepSpeed config file: /workspace/yarn/deepspeed/zero3.json                                                                           
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]: yes
How many GPU(s) should be used for distributed training? [1]:4
accelerate configuration saved at /root/.cache/huggingface/accelerate/default_config.yaml

# **Step2.** Modify deepspeed/zero3.json according to [deepspeed configuration json](https://www.deepspeed.ai/docs/config-json/) in case of OOM

# **Step3.** Enable wandb and train
$ accelerate launch finetune.py --output-dir  output/yarn-7b-8k  --model NousResearch/Llama-2-7b-hf --scaling-factor 2  --wandb ${YOUR_WANDB_PROJECT}  --dataset emozilla/yarn-train-tokenized-8k-llama    --deepspeed
```

### Evaluation

To reproduce the evaluations, install [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with `pip install git+https://github.com/EleutherAI/lm-evaluation-harness` and then run the two provided scripts.

```sh
# ./eval.sh
# ./eval-harness.sh
```

### Citation

```
@inproceedings{
      peng2024yarn,
      title={Ya{RN}: Efficient Context Window Extension of Large Language Models},
      author={Bowen Peng and Jeffrey Quesnelle and Honglu Fan and Enrico Shippole},
      booktitle={The Twelfth International Conference on Learning Representations},
      year={2024},
      url={https://openreview.net/forum?id=wHBfxhZu1u}
}
```
