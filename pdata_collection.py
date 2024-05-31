import os
import json
import vllm
import tqdm
import torch
import openai
import random
import backoff
import argparse
import numpy as np
import torch.nn.functional as F

from rouge_score import rouge_scorer
from vllm import LLM, SamplingParams
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

from nltk.translate.gleu_score import sentence_gleu # 24.05.31 update - fluency of prediction compared to long-form answer

# from openai.error import APIError, Timeout, APIConnectionError

# openai.api_key_path = "./key.txt"
# @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
# def completions_with_backoff(**kwargs):
#     return openai.ChatCompletion.create(**kwargs)

PROMPT = {
    "contradict": ("# OVERALL INSTRUCTIONS\n"
                   "- You have a deep understanding of logical relationships, such as entailment and contradiction, to evaluate given triplets of (question, premise, hypothesis).\n\n"
                   "# TASK INSTRUCTIONS\n"
                   "Your goal is to determine whether the Premise effectively contradicts the corresponding Hypothesis. Carefully analyze each triplet, focusing on details.\n"
                   "- If the premise and the hypothesis are unrelated or lack sufficient evidence to ascertain their truthfulness, label your answer as False.\n"
                   "- be vigilant in identifying cases where the premise doesn't rule out the possibility of an entity (e.g., vaccine, symptom) appearing in the hypothesis. In such cases, classify the answer as False.\n"
                   "- Approach each question methodically, considering the step-by-step process outlined below.\n\n"
                   "# INPUT DATA\n"
                   "Question: What does trich test for? Let's think step by step.\n"
                   "Premise: The term 'trich test' can refer to two different medical tests, depending on the context. Here are the two possibilities: Trichomoniasis Test: Trichomoniasis is a sexually transmitted infection (STI) caused by the parasite Trichomonas vaginalis. The trichomoniasis test, also known as a trich test or trichomonas test, is used to detect the presence of this parasite in the body. The test is typically performed on a sample of vaginal discharge in women or urine in men. Trichogram: A trichogram is a diagnostic test used to evaluate hair loss and assess the health and condition of hair follicles. It involves plucking a small number of hairs from the scalp and examining them under a microscope. It's important to note that without additional context, it's difficult to determine which specific test you are referring to.\n"
                   "Hypothesis: Trichamoniasis- a parasitic infection that can cause your symptoms.\n"
                   "Answer: According to the premise 'trich test' refer to two different medical tests. A Trichamoniasis test is one of them, which is used to detect this parasite's presence. As stated in the hypothesis, the trich test is used to diagnose parasitic infections. Ths premise entails the hypothesis. The answer is False.\n"
                   "###\n"
                   "Question: Can diabetics eat sweets? Let's think step by step.\n"
                   "Premise: Individuals with diabetes are recommended to limit their consumption of sweets to one or two times per week. It is also suggested being selective with desserts and to focus on foods with a low glycemic index, such as high fiber foods like whole grains and legumes, as well as certain lower sugar fruits like berries, melons, and apples.\n"
                   "Hypothesis: It is recommended that diabetics avoid sweets.\n"
                   "Answer: The premise suggests that diabetics can eat sweets but limit their consumption. According to the hypothesis diabetics should avoid sweets. Diabetics are allowed to consume sweets according to the premise, but they are prohibited according to the hypothesis. There is a contradiction between the premise and the hypothesis. The answer is True.\n"
                   "###\n"
                   "Question: 25 yo female with right lower abdominal pain, what might be causing it? Let's think step by step.\n"
                   "Premise: Right lower abdominal pain in a 25-year-old female could be caused by a variety of medical conditions. Some potential causes include: Ovarian cyst: a fluid-filled sac on the ovary - Ectopic pregnancy: a pregnancy that occurs outside the uterus.\n"
                   "Hypothesis: possible cause for right lower abdominal pain in a young female can be Appendicitis.\n"
                   "Answer: The premise lists several potential causes of right lower abdominal pain in a 25-year-old female, not including appendicitis. The hypothesis states that Appendicitis could be a cause of right lower abdominal pain in a young female. There is no direct contradiction between the premise and the hypothesis, as the premise does not exclude the possibility of appendicitis as the cause of the pain. The answer is False.\n"
                   "###\n"
                   "Question: Can a headache last longer than a few days? Let's think step by step.\n"
                   "Premise: Yes, it is possible. If you are experiencing a headache that lasts longer than a few days, it is important to see a doctor to get the appropriate treatment. This will help to relieve the pain and prevent any further complications.\n"
                   "Hypothesis: It is not a cause for concern if a headache lasts longer than a few days.\n"
                   "Answer: This premise acknowledges that a headache can last for several days, but emphasizes that seeing a doctor to prevent further complications is important. According to this hypothesis, headaches lasting longer than a few days are not cause of concern. There is a contradiction between the premise and hypothesis due to the premise emphasizing the importance of seeking medical consultation, while the hypothesis posits that there is no cause for concern. The answer is True.\n"
    ),
    "entail": ("# OVERALL INSTRUCTIONS\n"
               "- You have a deep understanding of logical relationships, such as entailment and contradiction, to evaluate given triplets of (question, premise, hypothesis).\n\n"
               "# TASK INSTRUCTIONS\n"
               "Your goal is to determine whether the Premise effectively entails the corresponding Hypothesis. Carefully analyze each triplet, focusing on details.\n"
               "- If the premise disagrees with, is unrelated to, or does not support the hypothesis, there is not enough evidence to determine whether it is true, and so you answer should be False.\n"
               "- Approach each question methodically, considering the step-by-step process outlined below.\n\n"
               "# INPUT DATA\n"
                "Question: What does trich test for? Let's think step by step.\n"
                "Premise: The term 'trich test' can refer to two different medical tests, depending on the context. Here are the two possibilities: Trichomoniasis Test: Trichomoniasis is a sexually transmitted infection (STI) caused by the parasite Trichomonas vaginalis. The trichomoniasis test, also known as a trich test or trichomonas test, is used to detect the presence of this parasite in the body. The test is typically performed on a sample of vaginal discharge in women or urine in men. Trichogram: A trichogram is a diagnostic test used to evaluate hair loss and assess the health and condition of hair follicles. It involves plucking a small number of hairs from the scalp and examining them under a microscope. It's important to note that without additional context, it's difficult to determine which specific test you are referring to.\n"
                "Hypothesis: Trichamoniasis- a parasitic infection that can cause your symptoms.\n"
                "Answer: According to the premise 'trich test' refer to two different medical tests. A Trichamoniasis test is one of them, which is used to detect this parasite's presence. As stated in the hypothesis, the trich test is used to diagnose parasitic infections. Ths premise entails the hypothesis. The answer is True.\n"
                "###\n"
                "Question: Can diabetics eat sweets? Let's think step by step.\n"
                "Premise: Individuals with diabetes are recommended to limit their consumption of sweets to one or two times per week. It is also suggested being selective with desserts and to focus on foods with a low glycemic index, such as high fiber foods like whole grains and legumes, as well as certain lower sugar fruits like berries, melons, and apples.\n"
                "Hypothesis: It is recommended that diabetics avoid sweets.\n"
                "Answer: The premise suggests that diabetics can eat sweets but limit their consumption. According to the hypothesis diabetics should avoid sweets. Diabetics are allowed to consume sweets according to the premise, but they are prohibited according to the hypothesis. There is a contradiction between the premise and the hypothesis. The answer is False.\n"
                "###\n"
                "Question: What is the best hypertension treatment for patients who are also have Crohn's disease? Let's think step by step.\n"
                "Premise: For patients with Crohn's disease and hypertension, the recommended treatment is a combination of lifestyle changes and medication. The ACC/AHA recommends initiation of antihypertensive drug therapy at a BP \u2265130/80 mm Hg for adults with hypertension. It is also important to monitor your blood pressure regularly to make sure that it is under control.\n"
                "Hypothesis: reducing sodium intake, are the first-line treatment for hypertension in individuals with  Crohn's disease\n"
                "Answer: The premise suggests that the recommended treatment for patients with diabetes and hypertension is a combination of lifestyle changes and medication, including antihypertensive drug therapy. The hypothesis focuses on reducing sodium intake as the first-line treatment. A reduction in sodium intake could be a part of the lifestyle changes, but since it is not mentioned in the premise, the premise do not entail the hypothesis. The answer is False.\n"
                "###\n"
                "Question: 25 yo female with right lower abdominal pain, what might be causing it? Let's think step by step.\n"
                "Premise: Right lower abdominal pain in a 25-year-old female could be caused by a variety of medical conditions. Some potential causes include: Ovarian cyst: a fluid-filled sac on the ovary - Ectopic pregnancy: a pregnancy that occurs outside the uterus.\n"
                "Hypothesis: possible cause for right lower abdominal pain in a young female can be Appendicitis.\n"
                "Answer: The premise lists several potential causes of right lower abdominal pain in a 25-year-old female, not including appendicitis. The hypothesis states that Appendicitis could be a cause of right lower abdominal pain in a young female. There is no direct contradiction between the premise and the hypothesis, as the premise does not exclude the possibility of appendicitis as the cause of the pain. The answer is True.\n"
    )
}

def BERTSCORE(pred, answer):
    import bert_score
    from bert_score import score
    prec, rec, f1 = score([pred], [answer], lang='en', verbose=True)
    return prec.mean().item(), rec.mean().item(), f1.mean().item()

def ROUGESCORE(pred, answer):
    # for ASQA, K-QA datset
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    # scorer.score(pred, answer)['rougeL']
    # precision, recall, fmeasure
    rouge1_score = scorer.score(pred, answer)['rouge1']
    rouge2_score = scorer.score(pred, answer)['rouge2']
    rougel_score = scorer.score(pred, answer)['rougeL']
    # return scorer.score(pred, answer)['rougeL'].fmeasure
    return rouge1_score, rouge2_score, rougel_score

def BLEURT(pred, answer, model=None, tokenizer=None, device=None):
    # config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20-D12') # lucadiliello/BLEURT-20
    model.eval()
    with torch.no_grad():
        try:
            inputs = tokenizer([answer], [pred], padding='longest', return_tensors='pt').to(device)
            output = model(**inputs)
            res = output.logits.flatten().tolist()
        except:
            # truncate to max length
            inputs['input_ids'] = inputs['input_ids'][:, :512].to(device)
            inputs['attention_mask'] = inputs['attention_mask'][:, :512].to(device)
            inputs['token_type_ids'] = inputs['token_type_ids'][:, :512].to(device)
            output = model(**inputs)
            res = output.logits.flatten().tolist()

    return res

def HALLUCINATION(query, pred, must_have, nice_to_have, use_gpt=False, model=None, tokenizer=None, device=None):
    all_statements = must_have + nice_to_have
    hall_cnt = 0
    for statement in tqdm.tqdm(all_statements, desc="hallucination"):
        if use_gpt:
            prompt = PROMPT["contradict"]
            prompt += f"###\nQuestion: {query} Let's think step by step.\nPremise: {pred}\nHypothesis: {statement}\nAnswer: "
            result = completions_with_backoff(
                model="gpt-4",
                messages=[
                    {"role": "user",
                    "content": prompt},
                ],
                request_timeout=60,
                max_tokens=512,
            )
            
            result_text = result['choices'][0]['message']['content']
            # post-process result
            if "True" in result_text[-21:]:
                hall_cnt += 1
        else:
            def mean_pooling(model_output, attention_mask):
                token_embeddings = model_output[0]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
            encoded_input = tokenizer([pred, statement], padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            # encoded_input = tokenizer([pred, statement], padding=True, truncation=True, max_length=512, return_tensors='pt') # no gpu
            with torch.no_grad():
                model_output = model(**encoded_input)
            
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            
            cos = torch.nn.CosineSimilarity(dim=0)
            cos_score = cos(sentence_embeddings[0], sentence_embeddings[1]).item()
            
            if cos_score < 0.5:
                hall_cnt += 1
            
    try:
        return hall_cnt / len(all_statements) * 100
    except ZeroDivisionError:
        return 0


def COMPREHENSIVENESS(query, pred, must_have, use_gpt=False, model=None, tokenizer=None, device=None):
    if len(must_have) == 0:
        return 0
    
    comp_cnt = 0
    for statement in tqdm.tqdm(must_have, desc="Comprehensiveness"):
        if use_gpt:
            prompt = PROMPT["entail"]
            prompt += f"###\nQuestion: {query} Let's think step by step.\nPremise: {pred}\nHypothesis: {statement}\nAnswer: "
            result = completions_with_backoff(
                model="gpt-4",
                messages=[
                    {"role": "user",
                    "content": prompt},
                ],
                request_timeout=60,
                max_tokens=512,
            )
            
            result_text = result['choices'][0]['message']['content']
            # post-process result
            if "True" in result_text[-21:]:
                comp_cnt += 1
        else:
            def mean_pooling(model_output, attention_mask):
                token_embeddings = model_output[0]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            encoded_input = tokenizer([pred, statement], padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            # encoded_input = tokenizer([pred, statement], padding=True, truncation=True, max_length=512, return_tensors='pt') # no gpu
            with torch.no_grad():
                model_output = model(**encoded_input)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            cos = torch.nn.CosineSimilarity(dim=0)
            cos_score = cos(sentence_embeddings[0], sentence_embeddings[1]).item()
            if cos_score >= 0.5:
                comp_cnt += 1

    return comp_cnt / len(must_have) * 100
    

def vllm_infer(client, tokenizer, prompt, stop_seq, max_new_tokens=1024, cot=False, temperature=0.0):
    """
    Generates a single output for a given input prompt using the VLLM backend (offline mode).
    Returns the output text.

    Reference:

    :param client: vllm.LLM, the LLM offline generation engine to use for querying the VLLM backend
    :param tokenizer: transformers.PreTrainedTokenizer, the tokenizer to use for inference
    :param prompt: str, the prompt to generate from
    :param stop_seq: list, the stop sequence to use for generation
    :param max_new_tokens: int, the maximum number of tokens to generate
    :param cot: bool, whether to use chain-or-thought or not
    :param temperature: float, the temperature to use for sampling
    """

    response = client.generate(prompt, sampling_params=vllm.SamplingParams(
        # See https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
        best_of=1,
        presence_penalty=0.0,
        frequency_penalty=1.0,
        top_k=-1,
        top_p=1.0,
        temperature=temperature,
        stop=stop_seq,
        use_beam_search=False,
        max_tokens=max_new_tokens,
        logprobs=5
    ))

    def top_answer(logprob):
        top_token = max(logprob, key=logprob.get)
        output_text = tokenizer.decode(top_token, skip_special_tokens=True)
        return output_text

    if len(response) > 0:
        return [r.outputs[0].text for r in response]

    if not cot:
        return top_answer(response[0].outputs[0].logprobs[0])
    else:
        return response[0].outputs[0].text

def tokenizer_param(tokenizer, target, shots=0, cot=False, task_type="mcq"):
    """
    Determines the maximum number of tokens to generate for a given prompt and target.
    Also determines the stop sequence to use for generation.

    :param tokenizer: transformers.PreTrainedTokenizer, the tokenizer to use for inference
    :param target: str, the target to generate
    :param shots: int, the number of shots to use for few-shot learning
    :param cot: bool, whether to use chain-or-thought or not
    :param task_type: str, the type of answer to generate (mcq or open)
    """
    max_new_tokens = len(tokenizer(target, add_special_tokens=True)['input_ids'])
    stop_seq = [tokenizer.eos_token, tokenizer.pad_token, "###"]

    if not cot and task_type == "mcq":
        max_new_tokens = len(tokenizer(target[0], add_special_tokens=False)['input_ids'])
        if shots > 0:
            max_new_tokens += 8
    if cot:
        max_new_tokens = 1024

    return max_new_tokens, stop_seq


def main():
    # check Evaluation Metrics
    """
    query = "Alright so I dont know much about Lexapro would you tell me more about it?"
    answer = "Escitalopram, sold under the brand names Lexapro and Cipralex, is an antidepressant of the SSRI (selective serotonin reuptake inhibitors) class. It is a medication for major depressive disorder and several types of anxiety disorders. It is considered an effective and well-tolerated antidepressant. The benefits of Lexapro for treating depression occur within a few weeks, but it can take about 1 to 2 months before you feel its full effects.\nLike other SSRIs, side effects include headache, nausea, sleepiness, ejaculation disorder, and insomnia. The FDA had published a black box warning for Escitalopram and other antidepressants, alerting for an increased risk of suicidal thinking and behavior in children, adolescents, and young adults. Therefore, Lexapro is not approved for use in pediatric patients less than 12 years of age."
    pred = "Lexapro is a medication that belongs to a class of drugs called selective serotonin reuptake inhibitors (SSRIs).Lexapro is primarily used to treat depression and anxiety disorders.It may take a few weeks for Lexapro to take effect, so it is important to be patient and continue taking the medication as prescribed by your healthcare provider.It is also important to discuss any potential side effects with your doctor.Lexapro can cause some side effects, but not everyone experiences them.Remember, everyone's response to medication can vary, so it's important to work closely with your healthcare provider to determine if Lexapro is right for you."
    must_have = ["Escitalopram is an antidepressant of the SSRI (Selective serotonin reuptake inhibitors) class","Escitalopram is sold under the brand names Lexapro and Cipralex","Side effects of Escitalopram include GI symptoms such as nausea, diarrhoea, constipation","Side effects of Escitalopram include headache","Side effects of Escitalopram include ejaculation disorder","The benefits of Lexapro for treating depression occurs within a few weeks","Side effects of Escitalopram include sleepiness","Side effects of Escitalopram include insomnia","The FDA had published a black box warning regarding Escitalopram, alerting for an increased risk of suicidal thinking and behavior in children","The FDA had published a black box warning for Escitalopram, alerting for an increased risk of suicidal thinking and behavior in adolescents and young adults"," Lexapro is not approved for use in pediatric patients less than 12 years of age."]
    nice_to_have = ["Escitalopram is a medication for major depressive disorder","Escitalopram is a medication for several types of anxiety disorders","Escitalopram is considered an effective and well-tolerated antidepressant"]
    """
    
    # load NLI model - gpt4
    # rougel_score = rougel(pred, answer)
    # hall_score = hallucination(query, pred, must_have, nice_to_have)
    # comp_score = comprehensiveness(query, pred, must_have)
    # 0.0 / 18.18
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="dmis-lab/selfbiorag_7b") # mistralai/Mistral-7B-v0.1, BioMistral/BioMistral-7B, meta-llama/Llama-2-7b-hf, dmis-lab/selfbiorag_7b, epfl-llm/meditron-7b
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--download_dir', type=str, help="specify vllm model download dir",
                        default="./ssd0/minbyul/cache/") # need change
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument("--world_size",  type=int, default=1,
                        help="world size to use multiple GPUs.")
    parser.add_argument("--dtype",  type=str, default="half",
                        help="We use bfloat16 for training. If you run inference on GPUs that do not support BF16, please set this to be `half`.")
    parser.add_argument("--sampling_trials",  type=int, default=5,
                        help="sampling_trials to derive sampled predictions")
    parser.add_argument("--use_gpt", action="store_true", help="use gpt-4 with openai key")
    parser.add_argument("--eval_data", type=str, default="")
    parser.add_argument('--data_size', type=str, default="")
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    args = parser.parse_args()

    if not os.path.exists("./alignment-handbook/predictions"):
        os.mkdir("./alignment-handbook/predictions")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if "selfbiorag" in args.model_name_or_path.lower():
        model_name = "selfbiorag-7b"
    elif "biomistral" in args.model_name_or_path.lower():
        model_name = "biomistral-7b"
    elif "mistral" in args.model_name_or_path.lower():
        model_name = "mistral-7b"
    elif "llama-2" in args.model_name_or_path.lower():
        model_name = "llama2-7b"
    elif "llama-3-8b-instruct" in args.model_name_or_path.lower():
        model_name = "llama3-8b-instruct"
    elif "llama-3" in args.model_name_or_path.lower():
        model_name = "llama3-8b"
    elif "meditron" in args.model_name_or_path.lower():
        model_name = "meditron-7b"
    elif "gemma" in args.model_name_or_path.lower():
        model_name = "gemma-7b"
    else:
        model_name = args.model_name_or_path.split("/")[1]

    if "meditron" in args.model_name_or_path.lower() or "llama" in args.model_name_or_path.lower() or "mistral" in args.model_name_or_path.lower():
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
    else:
        model = LLM(model=args.model_name_or_path, download_dir=args.download_dir,
                    dtype=args.dtype, tensor_parallel_size=args.world_size,)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
        
    # load prediction and dataset
    prompt = ""
    eval_name = args.eval_data
    train_examples = []
    
    if os.path.exists(f"./alignment-handbook/predictions/pdata_{model_name}_{eval_name}_sampling.jsonl_tmp"):
        filename = f"./alignment-handbook/predictions/pdata_{model_name}_{eval_name}_sampling.jsonl_tmp"
        with open(filename, 'r') as fp:
            for line in fp.readlines():
                train_examples.append(json.loads(line))
    else:
        filename = f"./MedLFQA/{eval_name}_test_MedLFQA.jsonl"
        with open(filename, 'r') as fp:
            for line in fp.readlines():
                train_examples.append(json.loads(line))
    
    for inst_idx ,inst in enumerate(train_examples):
        # query
        query = prompt + inst['Question']
        
        # add question mark
        if query[-1] != "?":
            query += "?"

        if "tmp" in filename and "sample_predictions" in inst and "prediction_scores" in inst:
            continue

        # ten generation to make preference collections - check hallucination
        sample_predictions = []
        if "meditron-7b" == model_name or "llama2-7b" == model_name or "mistral-7b" == model_name or "llama3-8b" == model_name:
            input_ids = tokenizer.encode(query, return_tensors="pt").to(device)
            output = model.generate(input_ids, max_length=512, no_repeat_ngram_size=2, do_sample=False, top_p=1.0, repetition_penalty=args.repetition_penalty).to(device)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            pred = response[len(query):].strip()
            sample_predictions.append(pred)

            for _ in range(args.sampling_trials):
                input_ids = tokenizer.encode(query, return_tensors="pt").to(device)
                output = model.generate(input_ids, max_length=512, no_repeat_ngram_size=2, do_sample=True, top_p=1.0, temperature=1.0, repetition_penalty=args.repetition_penalty).to(device)
                response = tokenizer.decode(output[0], skip_special_tokens=True)
                pred = response[len(query):].strip()
                sample_predictions.append(pred)
        elif "llama3-8b-instruct" == model_name:
            messages = [
                {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
                {"role": "user", "content": query},
            ]

            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            outputs = model.generate(input_ids, max_new_tokens=512, eos_token_id=terminators, do_sample=False, temperature=0.0, top_p=0.9)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred = response[len(query):].strip()
            sample_predictions.append(pred)

            for _ in range(args.sampling_trials):
                outputs = model.generate(input_ids, max_new_tokens=512, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                pred = response[len(query):].strip()
                sample_predictions.append(pred)

        else:
            if "selfbiorag" in args.model_name_or_path:
                query += "[No Retrieval]"
            
            sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)
            preds = model.generate([query], sampling_params)
            pred = preds[0].outputs[0].text.strip()
            sample_predictions.append(pred)

            sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_new_tokens)
            for _ in range(args.sampling_trials):
                preds = model.generate([query], sampling_params)
                pred = preds[0].outputs[0].text.strip()
                sample_predictions.append(pred)
        
        inst['sample_predictions'] = sample_predictions

        # load bleurt model
        bleurt_model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20-D12')
        bleurt_tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20-D12')

        # load nli model for hallucination and comprehensiveness
        if not args.use_gpt:
            nli_model = AutoModel.from_pretrained('gsarti/biobert-nli', max_length=512).to(device)
            # nli_model = AutoModel.from_pretrained('gsarti/biobert-nli', max_length=512) # no gpu
            nli_tokenizer = AutoTokenizer.from_pretrained('gsarti/biobert-nli') #gsarti/biobert-nli

        prediction_scores = []
        for sample_idx,sample in enumerate(sample_predictions):
            sample = sample.strip()
            rouge1, rouge2, rougel = ROUGESCORE(sample, inst['Free_form_answer']) # higher better
            bleurt = BLEURT(sample, inst['Free_form_answer'], model=bleurt_model, tokenizer=bleurt_tokenizer) # higher better
            bs_p, bs_r, bs_f1 = BERTSCORE(sample, inst['Free_form_answer']) # higher better

            # hallucination and comprehensiveneess with gpt-4 or biobert-nli model
            hall_score = HALLUCINATION(inst["Question"], sample, inst["Must_have"], inst["Nice_to_have"], use_gpt=args.use_gpt, model=nli_model, tokenizer=nli_tokenizer, device=device) # lower better
            comp_score = COMPREHENSIVENESS(inst["Question"], sample, inst["Must_have"], use_gpt=args.use_gpt, model=nli_model, tokenizer=nli_tokenizer, device=device) # higher better

            # 24.05.31 update - fluency
            fluency_score = sentence_gleu([answer], sample)

            prediction_scores.append({"idx":sample_idx, "rouge1_p":round(rouge1.precision, 4), "rouge1_r": round(rouge1.recall, 4), "rouge1_f1": round(rouge1.fmeasure, 4), "rouge2_p": round(rouge2.precision, 4), "rouge2_r": round(rouge2.recall, 4), "rouge2_f1": round(rouge2.fmeasure, 4), "rougel_p": round(rougel.precision, 4), "rougel_r": round(rougel.recall, 4), "rougel_f1": round(rougel.fmeasure, 4), "bleurt": round(bleurt[0], 4), "bert_score_p": round(bs_p, 4), "bert_score_r": round(bs_r, 4), "bert_score_f1": round(bs_f1, 4), "hallucination": hall_score, "comprehensive": comp_score, "fluency": round(fluency_score, 4)})
        
        inst['prediction_scores'] = prediction_scores

        if (inst_idx+1) % 5 == 0:
            print (inst)
            
            with open(f"./alignment-handbook/predictions/pdata_{model_name}_{eval_name}_sampling.jsonl_tmp", "w") as outfile:
                for inst in train_examples:
                    outfile.write(json.dumps(inst))
                    outfile.write("\n")

    with open(f"./alignment-handbook/predictions/pdata_{model_name}_{eval_name}_sampling.jsonl", "w") as outfile:
        for inst in train_examples:
            outfile.write(json.dumps(inst))
            outfile.write("\n")
            

if __name__ == "__main__":
    main()