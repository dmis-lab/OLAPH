# OLAPH: Improving Factuality in Biomedical Long-form Question Answering

This is a repository for [OLAPH: Improving Factuality in Biomedical Long-form Question Answering](https://arxiv.org/abs/2405.12701) by Minbyul Jeong, Hyeon Hwang, Chanwoong Yoon, Taewhoo Lee, and Jaewoo Kang.

[MedLFQA](https://huggingface.co/datasets/dmis-lab/MedLFQA) | [Self-BioRAG (OLAPH)](https://huggingface.co/dmis-lab/self-biorag-7b-olaph) | [BioMistral (OLAPH)](https://huggingface.co/dmis-lab/biomistral-7b-olaph) | [Mistral (OLAPH)](https://huggingface.co/dmis-lab/mistral-7b-olaph) | [Summary](https://www.linkedin.com/posts/minbyul-jeong-183000194_introducing-medlfqa-olaph-a-biomedical-activity-7198887412050112512-5eHq?utm_source=share&utm_medium=member_desktop) | [Paper](https://arxiv.org/abs/2405.12701) 

![](figures/olaph.png)

1) **MedLFQA** is a reconstructed format of long-form question-answering (LFQA) benchmark datasets in biomedical domain to facilitate automatic evaluation.
2) **OLAPH** is a framework that reduces hallucinations and includes crucial claims by utilizing automatic evaluation to select the best response in sampling predictions and designing to answer questions in preferred manner.

## Content
1. [Installation](#installation)
2. [Quick Usage](#quick-usage)
3. [Datasets](#datasets)
4. [Training](#training)
5. [Inference](#inference)
6. [Iterative Learning](#iterative-learning)
7. [FactScore](#factscore)
8. [FAQ](#faq)
9. [Citation](#citation)
10. [Contact Information](#contact-information)

## Installation
Please create a conda environment by running the command below.
Note that we use two different environments to train and inference.
I will ensure that everything is integrated into a single environment and functions properly in the future.

First, you have to install following [alignment-handbook](https://github.com/huggingface/alignment-handbook/tree/main).
We use PyTorch v2.1.2, which is important for reproducibility! \
Since this is dependent on your environmental settings, please follow and use compatible version of Pytorch from [here](https://pytorch.org/get-started/locally/) \

Then, we install the remaining package dependencies as follows:

```
conda create -n olaph python=3.10
cd ./alignment-handbook/ \
python -m pip install .
```

This could lead us to install for the most recent version of torch.
However, we use CUDA 11.8 version in our experimental settings.
Thus, we recommend you to download a below code to reproduce our results \


<!-- conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia -->
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

You will need Flash Attention 2 installed \

```
python -m pip install flash-attn==2.5.6 --no-build-isolation \
```

We need further requirments to install for automatic evaluation or vllm for boosting inference speed \
```
pip install -r requirements.txt --no-build-isolation \
pip install git+https://github.com/lucadiliello/bleurt-pytorch.git \
```

Also, you will need to log into your Huggingface account (make sure your account token should be in WRITE status)
Then, install the Git LFS to upload your models as follows:

```
huggingface-cli login \
sudo apt-get install git-lfs \
```
<!-- 
For training,
```
conda env create -f training.yaml
conda activate olaph_training
```

For inference,
```
conda env create -f inference.yaml
conda activate olaph_inference
``` -->


## Quick Usage
You can download 7B models trained with our OLAPH framework from HuggingFace hub.
```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "dmis-lab/self-biorag-7b-olaph" # ["mistralai/Mistral-7B-v0.1", "BioMistral/BioMistral-7B", "meta-llama/Llama-2-7b-hf", "dmis-lab/selfbiorag_7b", "epfl-llm/meditron-7b"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

query = "Alright so I don't know much about Lexapro would you tell me more about it?"

input_ids = tokenizer.encode(query, return_tensors="pt").to(device)
output = model.generate(input_ids, max_length=512, no_repeat_ngram_size=2, do_sample=False, top_p=1.0).to(device)
response = tokenizer.decode(output[0], skip_special_tokens=True).strip()

print ("Model prediction: ", response)
```

## Datasets
**MedLFQA** is a reconstructed format of long-form question-answering (LFQA) benchmark datasets in biomedical domain to facilitate automatic evaluation.
We construct the **MedLFQA** with four biomedical LFQA benchmark datasets: [LiveQA](https://github.com/abachaa/LiveQA_MedicalTask_TREC2017), [MedicationQA](https://github.com/abachaa/Medication_QA_MedInfo2019), [HealthSearchQA](https://huggingface.co/datasets/katielink/healthsearchqa), and [K-QA](https://github.com/Itaymanes/K-QA).
Our **MedLFQA** instance is comprised of four components: question (Q), long-form answer (A), Must Have Statements (MH), Nice to Have Statements (NH).
We provide the reconstructed datasets for automatic evaluation of long-form generated responses.

## Inference

* Sampling Predictions (Including Automatic Evaluation)

```
# For first sampling predictions \ 

export DATA_NAME=live_qa \
export HUGGINGFACE_MODEL_DIR=dmis-lab/selfbiorag_7b \

CUDA_VISIBLE_DEVICES=0 python pdata_collection.py \
--model_name_or_path ${HUGGINGFACE_MODEL_DIR} \
--eval_data ${DATA_NAME} \
```

```
# Sampling prediction during Iterative learning (i.e., after SFT or DPO) \

export DATA_NAME=live_qa \
export HUGGINGFACE_MODEL_DIR=your_trained_model \

CUDA_VISIBLE_DEVICES=0 python pdata_collection.py \
--model_name_or_path ${HUGGINGFACE_MODEL_DIR} \
--eval_data ${DATA_NAME} \
```


## Training

* Supervised Fine-Tuning (SFT)

After we obtain sampled predictions from previous step, we use SFT to recognize the question-answering task.
Rather than training on human-annotated answer or pseudo-optimal responses generated by GPT-4, we set a self-generated response as a labeled asnwer to remove the depedency on resources in annotation datasets.
We use a representative 7B model for Self-BioRAG.
If you want to use another models with difference configuration, you should change directions of recipes.

```
cd alignment-handbook \

CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/accelerate_configs/deepspeed_zero3.yaml  \
--num_processes 4 \
scripts/run_sft.py \
recipes/selfbiorag_7b/sft/config_full.yaml \
```

* Make synthetic preference set based on sampling predictions

```
export HUGGINGFACE_MODEL_DIR=your_trained_model
export DATA_NAME=live_qa
export WODATA_NAME=kqa_golden

python pred_to_preference.py \
--model_name ${HUGGINGFACE_MODEL_DIR} \
--wodata_name ${WODATA_NAME} \
--data_names live_qa+medication_qa+healthsearch_qa+kqa_golden+kqa_silver_wogold \
--alpha 1.0 \
--beta 1.0 \
--gamma 1.0 \
--threshold 200 \

python pred_to_preference.py \
--model_name ${HUGGINGFACE_MODEL_DIR} \
--wodata_name ${WODATA_NAME} \
--data_names ${DATA_NAME} \
--alpha 1.0 \
--beta 1.0 \
--gamma 1.0 \
--threshold 200 \
```

* Direct Preference Optimization (DPO)

```
cd alignment-handbook \

CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/accelerate_configs/deepspeed_zero3.yaml  \
--num_processes 4 \
scripts/run_dpo.py \
recipes/selfbiorag_7b/sft/config_full.yaml \
```

## Iterative Learning
We train and generate sampling predictions through separate files and do several times.
In future, we will provide the processes execution in one simple bash file.

Our iterative learning consists of the following processes \
- Sampling predictions (pdata_collection.py) - Make SFT set (pred_to_sft.py) \
- SFT - Sampling predictions (pdata_collection.py) - Make preference set (pred_to_preference.py) \
- DPO - Sampling predictions (pdata_collection.py) - Make preference set (pred_to_preference.py) \
- DPO - Sampling predictions (pdata_collection.py) - Make preference set (pred_to_preference.py) \
- DPO 

## FactScore
TBA

## FAQ
TBA

## Citation
```
@misc{jeong2024olaph,
      title={OLAPH: Improving Factuality in Biomedical Long-form Question Answering}, 
      author={Minbyul Jeong and Hyeon Hwang and Chanwoong Yoon and Taewhoo Lee and Jaewoo Kang},
      year={2024},
      eprint={2405.12701},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Contact Information
For help or issues using **MedLFQA** & **OLAPH**, please submit a GitHub issue. \
Please contact Minbyul Jeong (`minbyuljeong (at) korea.ac.kr`) or Hyeon Hwang (`hyeon-hwang (at) korea.ac.kr`) for communication related to OLAPH.
