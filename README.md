# OLAPH
OLAPH: Improving Factuality in Biomedical Long-form Question Answering

This is a repository for [OLAPH: Improving Factuality in Biomedical Long-form Question Answering]() by .

[MedLFQA]() | [7B Models]() | [Summary]() | [Paper]() 

**MedLFQA** is a reconstructed format of long-form question-answering (LFQA) benchmark datasets in biomedical domain to facilitate automatic evaluation.

**OLAPH** is a framework that reduces hallucinations and includes crucial claims by utilizing automatic evaluation to select the best response in sampling predictions and designing to answer questions in preferred manner.

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

For training,
```
conda env create -f training.yaml
conda activate olaph_training
```

For inference,
```
conda env create -f inference.yaml
conda activate olaph_inference
```


## Quick Usage
You can download 7B models trained with our OLAPH framework from HuggingFace hub.
```py
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

query = "Alright so I don't know much about Lexapro would you tell me ore about it?"

input_ids = tokenizer.encode(query, return_tensors="pt").to(device)
output = model.generate(input_ids, max_length=512, no_repeat_ngram_size=2, do_sample=False, top_p=1.0).to(device)
response = tokenizer.decode(output[0], skip_special_tokens=True).strip()

print ("Model prediction: ", response)
```

## Datasets
**MedLFQA** is a reconstructed format of long-form question-answering (LFQA) benchmark datasets in biomedical domain to facilitate automatic evaluation.
We construct **MedLFQA** with four biomedical LFQA benchmark datasets: [LiveQA](https://github.com/abachaa/LiveQA_MedicalTask_TREC2017), [MedicationQA](https://github.com/abachaa/Medication_QA_MedInfo2019), [HealthSearchQA](https://huggingface.co/datasets/katielink/healthsearchqa), and [K-QA](https://github.com/Itaymanes/K-QA).
Our **MedLFQA** instance is comprised of four components: question (Q), long-form answer (A), Must Have Statements (MH), Nice to Have Statements (NH).

## Training


## Inference

## Iterative Learning

## FactScore

## FAQ

## Citation

## Contact Information