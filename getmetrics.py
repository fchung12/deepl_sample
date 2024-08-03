import sys
import os
import json
import numpy as np
import jsonlines

from pytorch_pretrained_bert import BertTokenizer

# Run with python3 getmetrics.py <input JSON file>
# This script runs the BERT tokenizer on summaries in order to obtain valuable statistics

model_version = 'bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(model_version) 

lengths = []
curr = 0
with jsonlines.open(sys.argv[1]) as reader:
    for obj in reader:
        if curr % 1000 == 0:
            print("Metrics running on article " + str(curr))
        summ = obj["summary"]
        summ_tok = bert_tokenizer.tokenize(summ)
        if curr % 1000 == 0:
            print(summ_tok)
        lens = len(summ.split())
        lengths.append(lens)
        curr += 1

lens_np = np.asarray(lengths)

print("Max Length: " + str(np.amax(lens_np)))
print("Min Length: " + str(np.amin(lens_np)))
print("Mean Length: " + str(np.mean(lens_np)))
print("Median Length: " + str(np.median(lens_np)))
print("25th Percentile Length: " + str(np.percentile(lens_np, 25)))
print("75th Percentile Length: " + str(np.percentile(lens_np, 75)))
print("90th Percentile Length: " + str(np.percentile(lens_np, 90)))

