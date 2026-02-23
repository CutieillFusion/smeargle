import argparse
import json
from transformers import AutoTokenizer
import numpy as np

parser = argparse.ArgumentParser(description="Calculate speed ratio between Smeargle and baseline")
parser.add_argument("--smeargle", type=str, required=True, help="Path to Smeargle results jsonl file")
parser.add_argument("--baseline", type=str, required=True, help="Path to baseline results jsonl file")
parser.add_argument("--tokenizer-path", type=str, default="../models/llama_3_1_8b_instruct", help="Path to tokenizer")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
jsonl_file = args.smeargle
jsonl_file_base = args.baseline
data_smeargle = []
with open(jsonl_file, "r", encoding="utf-8") as file:
    for line in file:
        data_smeargle.append(json.loads(line))

data_baseline = []
with open(jsonl_file_base, "r", encoding="utf-8") as file:
    for line in file:
        data_baseline.append(json.loads(line))

for smeargle_datapoint, baseline_datapoint in zip(data_smeargle, data_baseline):
    if smeargle_datapoint["question_id"] != baseline_datapoint["question_id"]:
        print(f"Question ID mismatch: {smeargle_datapoint['question_id']} != {baseline_datapoint['question_id']}")

    for smeargle_choice, baseline_choice in zip(smeargle_datapoint["choices"], baseline_datapoint["choices"]):
        for smeargle_ids, baseline_ids in zip(smeargle_choice["turn_token_ids"], baseline_choice["turn_token_ids"]):
            mismatched_tokens = False
            minlen = min(len(smeargle_ids), len(baseline_ids))
            for pos, (smeargle_id, baseline_id) in enumerate(zip(smeargle_ids, baseline_ids)):
                if not mismatched_tokens and smeargle_id != baseline_id:
                    mismatched_tokens = True
                    print(f"Position, Baseline, Smeargle")
                if mismatched_tokens:
                    print(f"{pos}, {baseline_id}, {smeargle_id}")
