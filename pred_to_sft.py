import os
import json
import tqdm
import random
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="dmis-lab/selfbiorag_7b")
    parser.add_argument('--wodata_name', type=str, default="kqa_golden")
    parser.add_argument('--data_names', type=str, split="+", default="live_qa+medication_qa+healthsearch_qa+kqa_silver_wogold+kqa_golden")
    parser.add_argument('--alpha', type=int, default=1)
    parser.add_argument('--beta', type=int, default=1)
    parser.add_argument('--gamma', type=int, default=1)
    parser.add_argument('--threshold', type=int, default=200)
    args = parser.parse_args()

    if "selfbiorag" in args.model_name_or_path.lower():
        model_name = "selfbiorag-7b"
    elif "biomistral" in args.model_name_or_path.lower():
        model_name = "biomistral-7b"
    elif "mistral" in args.model_name_or_path.lower():
        model_name = "mistral-7b"
    elif "llama" in args.model_name_or_path.lower():
        model_name = "llama2-7b"
    elif "meditron" in args.model_name_or_path.lower():
        model_name = "meditron-7b"
    else:
        model_name = args.model_name_or_path.split("/")[1]

    # args.data_names = ["live_qa", "medication_qa", "healthsearch_qa", "kqa_golden", "kqa_silver_wogold"]
    # args.wodata_name = "kqa_silver_wogold"
    all_datasets = []
    for data_name in args.data_names:
        if data_name == args.wodata_name:
            continue
        alls = []
        with open(f"./predictions/pdata_{model_name}_{data_name}_sampling.jsonl") as fp:
            for line in fp.readlines():
                dictionary = json.loads(line)
                if dictionary['Question'][-1] != "?":
                    dictionary['Question'] += "?"

                r1, r2, rl, bl, bs, hl, cp = [], [], [], [], [], [], []
                tw, ss, f = [], [], []
                all = []
                for pred_score in dictionary['prediction_scores']:
                    r1.append(pred_score['rouge1_f1'] * 100)
                    r2.append(pred_score['rouge2_f1'] * 100)
                    rl.append(pred_score['rougel_f1'] * 100)
                    bl.append(pred_score['bleurt'] * 100)
                    bs.append(pred_score['bert_score_f1'] * 100)
                    hl.append(pred_score['hallucination'])
                    cp.append(pred_score['comprehensive'])
                    
                    tw.append(r1[-1] + r2[-1] + rl[-1])
                    ss.append(bl[-1] + bs[-1])
                    f.append(cp[-1] - hl[-1])
                    all.append(args.alpha * tw[-1] + args.beta * ss[-1] + args.gamma * f[-1])
                    
                alls.append(np.argmax(all))
                answer = dictionary['sample_predictions'][alls[0]]
                if "### Answer:" in answer:
                    answer = answer.split("### Answer:")[1].strip()
                all_datasets.append({"text":f"Question: {dictionary['Question']}\n ### Answer: {answer}"})
                
    with open(f"./predictions/wo_{args.wodata_name}_train_iter_sft_step1.jsonl", "a") as out_:
        for inst in all_datasets:
            out_.write(json.dumps(inst))
            out_.write("\n")
            

if __name__ == "__main__":
    main()