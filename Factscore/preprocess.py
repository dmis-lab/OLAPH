import numpy as np
import json
import argparse

ALPHA = 1
BETA = 1
GAMMA = 1
alls = []
data = []

def main(input_file, evidence_file, output_file, top_k):
    topic_dict = {}
    evidence_dict = {}
    with open(evidence_file, 'r') as fp:
        for line in fp.readlines():
            dictionary = json.loads(line)
            question = dictionary['Question']
            topic = dictionary['Topic']
            evidence = dictionary['Evidence'][:top_k]
            topic_dict.update({question:topic})
            evidence_dict.update({question:evidence})
    with open(input_file, 'r') as fp:
        for line in fp.readlines():
            dictionary = json.loads(line)
            question = dictionary['Question']
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
                all.append(ALPHA * tw[-1] + BETA * ss[-1] + GAMMA * f[-1])
                
            alls.append(np.argmax(all))
            answer = dictionary['sample_predictions'][alls[0]]
            if "### Answer:" in answer:
                answer = answer.split("### Answer:")[1].strip()
            inst = {"input": question, "output": answer, "topic": topic_dict[question], "evidence": evidence_dict[question]}
            data.append(inst)
            
    with open(output_file, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Path to the input JSONL file")
    parser.add_argument("--evidence_file", type=str, help="Path to the evidence JSONL file", default="kqa_golden_test_evidence_factscore.jsonl")
    parser.add_argument("--output_file", type=str, help="Path to the output JSONL file")
    parser.add_argument("--top_k", type=int, help="Top k evidences", default=20)
    args = parser.parse_args()
    main(args.input_file, args.evidence_file, args.output_file, args.top_k)
