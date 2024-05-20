import os
import nltk
import tqdm
import json
import torch
import random
import openai
import backoff
import argparse
import jsonlines

import numpy as np
import pandas as pd

# openai.api_key_path = "./key.txt"
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

seed = 633

torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
print('Cuda:', torch.cuda.is_available())
print('pwd', os.getcwd())


def main():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="K-QA benchmcark evaluation")
    parser.add_argument("--model_name", type=str, default="gpt-4")
    args = parser.parse_args()
    
    silver_standard = []
    # load silver-standard data including only questions
    with open("./MedLFQA/healthsearch_qa_test.jsonl", "r") as fp:
        lines = fp.readlines()
        for line in lines:
            silver_standard.append(json.loads(line))
            # Question, {Free_form_answer, Must_have, Nice_to_have}, entity_text, matched_entity, entity_tag, ICD_10_diag
    
    check_silver_standard = []
    with open("./MedLFQA/healthsearch_qa_silver_test.jsonl", "r") as fp:
        lines = fp.readlines()
        for line in lines:
            check_silver_standard.append(json.loads(line))
    check_queries = [inst["Question"] for inst in check_silver_standard]

    # generate Free_form_answer, Must_have, Nice_to_have statements using GPT-4 of silver_standard dataset.
    content = "Instruction: Answer the question in a 'Long Form Answer'. If you could not answer the question or question is vague, then response with 'Vague Question to answer'. In the process, generate 'Must Have Statements' and 'Nice to Have Statements' according to the conditions below.\n Must Have Statements: it indicates that a model must include this statement in order to be medically accurate (e.g., providing all contrindications for a drug).\n Nice to Have Statements: it indicates the statement is supplemental in nature (e.g., providing additional conditions where this drug may be helpful).\n\n"
    prompt = "###Question: And what happens if I miss a dose of Saxenda?\n"
    prompt += "Long Form Answer: Liraglutide (Saxenda) is a prescription drug that is used for weight loss and to help keep weight off once weight has been lost. It is used for obese adults or overweight adults who have weight-related medical problems. If you miss your dose of Saxenda, take a dose as soon as you remember on the same day. Then take your next daily dose as usual on the following day. Do not take an extra dose of Saxenda or increase your dose to make up for a missed dose. If you miss your dose of Saxenda for 3 days or more, contact your healthcare provider to consult about how to restart your treatment.\n\n"
    prompt += "Must Have Statements: If a dose of Saxenda is missed for 3 days or more, a healthcare provider should be contacted to consult about restarting the treatment. The dose of Saxenda should not be increased to make up for a missed dose. An extra dose of Saxenda should not be taken to make up for a missed dose. The next daily dose of Saxenda should be taken as usual on the following day after a missed dose. If a dose of Saxenda is missed, take a dose as soon as remembered on the same day.\n\n"
    prompt += "Nice to Have Statements: Liraglutide (Saxenda) is a prescription drug used for weight loss and to maintain weight loss in obese or overweight adults with weight-related medical problems.\n\n"

    for inst_idx, inst in tqdm.tqdm(enumerate(silver_standard)):
        if inst["Question"] in check_queries:
            continue
        # generate answer of questions that belongs to ground truth list and compare how GPT-4 well generated
        question = f"###Question: {inst['Question']}\n"
        question += f"Long Form Answer: "
        # query = inst['instances']['input'][10:]
        # question = f"###Question: {query}"
        # question += f"Long Form Answer: {inst['instances']['output']}"
        new_input = content + prompt + question
        
        # show examples
        try:
            results = completions_with_backoff(
                model=args.model_name,
                messages=[
                    {"role": "user",
                    "content": new_input},
                ],
                request_timeout=60,
                max_tokens=512,
            )
        except:
            print (inst_idx)
            continue

        gpt4_result_text = results['choices'][0]['message']['content']
        # postprocess gpt4_result
        gpt4_result = gpt4_result_text.split("\n\n")

        if len(gpt4_result) == 3:
            must_have_list, nice_to_have_list = [], []
            # postprocess must_have_list and nice_to_have_list
            must_have_list = nltk.sent_tokenize(gpt4_result[1][21:].strip())
            nice_to_have_list = nltk.sent_tokenize(gpt4_result[2][24:].strip())

            new_instance = {"Question": inst['Question'], "Free_form_answer": gpt4_result[0], "Must_have": must_have_list, \
                            "Nice_to_have": nice_to_have_list}
            
            with open("./MedLFQA/healthsearch_qa_silver_test.jsonl", "a") as out_:
                out_.write(json.dumps(new_instance))
                out_.write("\n")
            out_.close()
        else:
            with open("./healthsearch_qa_hard_to_response.txt", "a") as out_:
                out_.write(question)
                out_.write(gpt4_result_text)
                out_.write("\n")
            out_.close()


if __name__ == "__main__":
    main()

