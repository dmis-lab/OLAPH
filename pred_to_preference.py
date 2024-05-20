import json
import nltk

import argparse
import numpy as np


def eval_score(inst, alpha=1, beta=1, gamma=1):
    idx2score = {}
    for score in inst['prediction_scores']:
        rouge = alpha * 100 * (score['rouge1_f1'] + score['rouge2_f1'] + score['rougel_f1'])
        similarity = beta * 100 * (score['bleurt'] + score['bert_score_f1'])
        factuality = gamma * (score['comprehensive'] - score['hallucination'])
        total = rouge + similarity + factuality
        idx2score[score['idx']] = total
        
    return {key:value for key,value in sorted(idx2score.items(), key=lambda item: item[1], reverse=True)}

def make_inst(model_name, chosen_idx, reject_idx, inst):
    if model_name == "selfbiorag_7b":
        chosen = "[No Retrieval] " + inst["sample_predictions"][chosen_idx]
        rejected = "[No Retrieval] " + inst["sample_predictions"][reject_idx]
    else:
        chosen = inst["sample_predictions"][chosen_idx]
        rejected = inst["sample_predictions"][reject_idx]

    if "\u00ef\u00bb\u00bf" in chosen or "\u00ef\u00bb\u00bf" in rejected:
        return None
    elif "\ub530\uc628" in chosen or "\uac83\uc785\ub2c8" in chosen:
        return None
    elif "::" in chosen or "::" in rejected:
        return None
    elif "....." in chosen or "....." in rejected:
        return None
    elif "?" == chosen:
        return None
    else:
        if chosen == "" or rejected == "":
            return None
        new_inst = {"chosen": chosen, "rejected": rejected, "prompt": inst["Question"]}
        return new_inst
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="dmis-lab/selfbiorag_7b")
    parser.add_argument('--wodata_name', type=str, default="kqa_golden")
    parser.add_argument('--data_names', type=str, split="+")
    parser.add_argument('--alpha', type=int, default=1)
    parser.add_argument('--beta', type=int, default=1)
    parser.add_argument('--gamma', type=int, default=1)
    parser.add_argument('--threshold', type=int, default=200)
    args = parser.parse_args()

    examples = []
    model_name = args.model_name
    wodata_name = args.wodata_name
    # data_names = ["live_qa", "medication_qa", "healthsearch_qa", "kqa_silver_wogold", "kqa_golden"]
    data_names = args.data_names
    # data_names = ["kqa_golden"]
    
    for data_name in data_names:
        if len(data_names) == 1:
            target_data_name = data_name
        else:
            if data_name == wodata_name:
                continue
        
        with open(f"./predictions/pdata_{model_name}-wo-{wodata_name}-iter-dpo-step_{data_name}_iter-dpo-step.jsonl", "r") as fp:
            for line in fp.readlines():
                examples.append(json.loads(line))


    new_examples = []
    for inst_idx,inst in enumerate(examples):
        # check scores to make lists of preference/dispreference
        if "none" in inst["Question"].lower() or inst["Question"] == None:
            continue
        r1, r2, rl, bl, bs, hl, cp = [], [], [], [], [], [], []
        tw, ss, f = [], [], []
        all = []
        
        if inst["Question"][-1] != "?":
            inst["Question"] += "?"
        idx2score = eval_score(inst, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
        idx_list = list(idx2score.keys())
        
        new_sample_predictions = []
        new_prediction_scores = []
        for prediction_idx, prediction in enumerate(inst['sample_predictions']):
            if inst["Question"] in prediction:
                prediction = prediction.split(inst["Question"])[0]
                sent_text = nltk.sent_tokenize(prediction)
                prediction = " ".join(sent_text[:-1])
                
            if "Question: " in prediction:
                prediction = prediction.split("Question: ")[0].strip()

            if prediction == "?":
                continue
                
            new_sample_predictions.append(prediction)
            new_prediction_scores.append(inst['prediction_scores'][prediction_idx])
        
        inst['sample_predictions'] = new_sample_predictions
        inst['prediction_scores'] = new_prediction_scores
        
        for pred_idx, pred_score in enumerate(inst['prediction_scores']):
            if inst["sample_predictions"] == "?":
                continue
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
            all.append(tw[-1] + ss[-1] + f[-1])

        if all[idx_list[0]] < args.threshold:
            continue
        
        try:
            if not "?" in inst['sample_predictions'][idx_list[0]]:
                temp = 20
                if all[idx_list[0]] - all[idx_list[2]] >= temp:                
                    if make_inst(model_name, chosen_idx=idx_list[0], reject_idx=idx_list[2], inst=inst):
                        new_examples.append(make_inst(model_name, chosen_idx=idx_list[0], reject_idx=idx_list[2], inst=inst))
        
                """top 1"""     
                if inst['sample_predictions'][idx_list[1]] != inst['sample_predictions'][idx_list[2]]:
                    if all[idx_list[0]] - all[idx_list[1]] >= temp:
                        if make_inst(model_name, chosen_idx=idx_list[0], reject_idx=idx_list[1], inst=inst):
                            new_examples.append(make_inst(model_name, chosen_idx=idx_list[0], reject_idx=idx_list[1], inst=inst)) 
                if inst['sample_predictions'][idx_list[3]] != inst['sample_predictions'][idx_list[2]]:
                    if all[idx_list[0]] - all[idx_list[3]] >= temp:
                        if make_inst(model_name, chosen_idx=idx_list[0], reject_idx=idx_list[3], inst=inst):
                            new_examples.append(make_inst(model_name, chosen_idx=idx_list[0], reject_idx=idx_list[3], inst=inst))
                if inst['sample_predictions'][idx_list[4]] != inst['sample_predictions'][idx_list[2]]:
                    if all[idx_list[0]] - all[idx_list[4]] >= temp:
                        if make_inst(model_name, chosen_idx=idx_list[0], reject_idx=idx_list[4], inst=inst):
                            new_examples.append(make_inst(model_name, chosen_idx=idx_list[0], reject_idx=idx_list[4], inst=inst))
                if inst['sample_predictions'][idx_list[5]] != inst['sample_predictions'][idx_list[2]]:
                    if all[idx_list[0]] - all[idx_list[5]] >= temp:
                        if make_inst(model_name, chosen_idx=idx_list[0], reject_idx=idx_list[5], inst=inst):
                            new_examples.append(make_inst(model_name, chosen_idx=idx_list[0], reject_idx=idx_list[5], inst=inst))
        except:
            continue
            
    if len(data_names) == 1:
        with open(f"./predictions/preference_{model_name}_test_{target_data_name}_iter-dpo-step.jsonl", "a") as out_:
            for inst in new_examples:
                out_.write(json.dumps(inst))
                out_.write("\n")
    else:
        with open(f"./predictions/preference_{model_name}_test_all_wo_{wodata_name}_iter-dpo-step.jsonl", "a") as out_:
            for inst in new_examples:
                out_.write(json.dumps(inst))
                out_.write("\n")


if __name__ == "__main__":
    main()