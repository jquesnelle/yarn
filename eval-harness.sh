#!/bin/bash

LM_EVALUATION_HARNESS_PATH="../lm-evaluation-harness"
ARGS="--model=hf-causal-experimental --batch_size 2"
MODEL_ARGS="use_accelerate=True,dtype=bfloat16,trust_remote_code=True"
ARC="--tasks=arc_challenge --num_fewshot=25"
HELLASWAG="--tasks=hellaswag --num_fewshot=10"
TRUTHFULQA="--tasks=truthfulqa_mc --num_fewshot=0"
MMLU="--tasks=hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions --num_fewshot=5"

### ARC-Challenge

python ${LM_EVALUATION_HARNESS_PATH}/main.py ${ARGS} \
    ${ARC} \
    --model_args="pretrained=NousResearch/Yarn-Llama-2-7b-64k,${MODEL_ARGS}" \
    --output_path="data/Yarn-Llama-2-7b-64k-arc.json"

python ${LM_EVALUATION_HARNESS_PATH}/main.py ${ARGS} \
    ${ARC} \
    --model_args="pretrained=NousResearch/Yarn-Llama-2-7b-128k,${MODEL_ARGS}" \
    --output_path="data/Yarn-Llama-2-7b-128k-arc.json"

python ${LM_EVALUATION_HARNESS_PATH}/main.py ${ARGS} \
    ${ARC} \
    --model_args="pretrained=NousResearch/Yarn-Llama-2-13b-64k,${MODEL_ARGS}" \
    --output_path="data/Yarn-Llama-2-13b-64k-arc.json"

python ${LM_EVALUATION_HARNESS_PATH}/main.py ${ARGS} \
    ${ARC} \
    --model_args="pretrained=NousResearch/Yarn-Llama-2-13b-128k,${MODEL_ARGS}" \
    --output_path="data/Yarn-Llama-2-13b-128k-arc.json"

python ${LM_EVALUATION_HARNESS_PATH}/main.py ${ARGS} \
    ${ARC} \
    --model_args="pretrained=NousResearch/Yarn-Mistral-7b-64k,${MODEL_ARGS}" \
    --output_path="data/Yarn-Mistral-7b-64k-arc.json"

python ${LM_EVALUATION_HARNESS_PATH}/main.py ${ARGS} \
    ${ARC} \
    --model_args="pretrained=NousResearch/Yarn-Mistral-7b-128k,${MODEL_ARGS}" \
    --output_path="data/Yarn-Mistral-7b-128k-arc.json"

### Hellaswag

python ${LM_EVALUATION_HARNESS_PATH}/main.py ${ARGS} \
    ${HELLASWAG} \
    --model_args="pretrained=NousResearch/Yarn-Llama-2-7b-64k,${MODEL_ARGS}" \
    --output_path="data/Yarn-Llama-2-7b-64k-hellaswag.json"

python ${LM_EVALUATION_HARNESS_PATH}/main.py ${ARGS} \
    ${HELLASWAG} \
    --model_args="pretrained=NousResearch/Yarn-Llama-2-7b-128k,${MODEL_ARGS}" \
    --output_path="data/Yarn-Llama-2-7b-128k-hellaswag.json"

python ${LM_EVALUATION_HARNESS_PATH}/main.py ${ARGS} \
    ${HELLASWAG} \
    --model_args="pretrained=NousResearch/Yarn-Llama-2-13b-64k,${MODEL_ARGS}" \
    --output_path="data/Yarn-Llama-2-13b-64k-hellaswag.json"

python ${LM_EVALUATION_HARNESS_PATH}/main.py ${ARGS} \
    ${HELLASWAG} \
    --model_args="pretrained=NousResearch/Yarn-Llama-2-13b-128k,${MODEL_ARGS}" \
    --output_path="data/Yarn-Llama-2-13b-128k-hellaswag.json"

python ${LM_EVALUATION_HARNESS_PATH}/main.py ${ARGS} \
    ${HELLASWAG} \
    --model_args="pretrained=NousResearch/Yarn-Mistral-7b-64k,${MODEL_ARGS}" \
    --output_path="data/Yarn-Mistral-7b-64k-hellaswag.json"

python ${LM_EVALUATION_HARNESS_PATH}/main.py ${ARGS} \
    ${HELLASWAG} \
    --model_args="pretrained=NousResearch/Yarn-Mistral-7b-128k,${MODEL_ARGS}" \
    --output_path="data/Yarn-Mistral-7b-128k-hellaswag.json"

### MMLU

python ${LM_EVALUATION_HARNESS_PATH}/main.py ${ARGS} \
    ${MMLU} \
    --model_args="pretrained=NousResearch/Yarn-Llama-2-7b-64k,${MODEL_ARGS}" \
    --output_path="data/Yarn-Llama-2-7b-64k-mmlu.json"

python ${LM_EVALUATION_HARNESS_PATH}/main.py ${ARGS} \
    ${MMLU} \
    --model_args="pretrained=NousResearch/Yarn-Llama-2-7b-128k,${MODEL_ARGS}" \
    --output_path="data/Yarn-Llama-2-7b-128k-mmlu.json"

python ${LM_EVALUATION_HARNESS_PATH}/main.py ${ARGS} \
    ${MMLU} \
    --model_args="pretrained=NousResearch/Yarn-Llama-2-13b-64k,${MODEL_ARGS}" \
    --output_path="data/Yarn-Llama-2-13b-64k-mmlu.json"

python ${LM_EVALUATION_HARNESS_PATH}/main.py ${ARGS} \
    ${MMLU} \
    --model_args="pretrained=NousResearch/Yarn-Llama-2-13b-128k,${MODEL_ARGS}" \
    --output_path="data/Yarn-Llama-2-13b-128k-mmlu.json"

python ${LM_EVALUATION_HARNESS_PATH}/main.py ${ARGS} \
    ${MMLU} \
    --model_args="pretrained=NousResearch/Yarn-Mistral-7b-64k,${MODEL_ARGS}" \
    --output_path="data/Yarn-Mistral-7b-64k-mmlu.json"

python ${LM_EVALUATION_HARNESS_PATH}/main.py ${ARGS} \
    ${MMLU} \
    --model_args="pretrained=NousResearch/Yarn-Mistral-7b-128k,${MODEL_ARGS}" \
    --output_path="data/Yarn-Mistral-7b-128k-mmlu.json"

## TruthfulQA

python ${LM_EVALUATION_HARNESS_PATH}/main.py ${ARGS} \
    ${TRUTHFULQA} \
    --model_args="pretrained=NousResearch/Yarn-Llama-2-7b-64k,${MODEL_ARGS}" \
    --output_path="data/Yarn-Llama-2-7b-64k-truthfulqa.json"

python ${LM_EVALUATION_HARNESS_PATH}/main.py ${ARGS} \
    ${TRUTHFULQA} \
    --model_args="pretrained=NousResearch/Yarn-Llama-2-7b-128k,${MODEL_ARGS}" \
    --output_path="data/Yarn-Llama-2-7b-128k-truthfulqa.json"

python ${LM_EVALUATION_HARNESS_PATH}/main.py ${ARGS} \
    ${TRUTHFULQA} \
    --model_args="pretrained=NousResearch/Yarn-Llama-2-13b-64k,${MODEL_ARGS}" \
    --output_path="data/Yarn-Llama-2-13b-64k-truthfulqa.json"

python ${LM_EVALUATION_HARNESS_PATH}/main.py ${ARGS} \
    ${TRUTHFULQA} \
    --model_args="pretrained=NousResearch/Yarn-Llama-2-13b-128k,${MODEL_ARGS}" \
    --output_path="data/Yarn-Llama-2-13b-128k-truthfulqa.json"

python ${LM_EVALUATION_HARNESS_PATH}/main.py ${ARGS} \
    ${TRUTHFULQA} \
    --model_args="pretrained=NousResearch/Yarn-Mistral-7b-64k,${MODEL_ARGS}" \
    --output_path="data/Yarn-Mistral-7b-64k-truthfulqa.json"

python ${LM_EVALUATION_HARNESS_PATH}/main.py ${ARGS} \
    ${TRUTHFULQA} \
    --model_args="pretrained=NousResearch/Yarn-Mistral-7b-128k,${MODEL_ARGS}" \
    --output_path="data/Yarn-Mistral-7b-128k-truthfulqa.json"
