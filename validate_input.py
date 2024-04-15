import pandas as pd
import sys
import pickle

inp = pd.read_csv(sys.argv[1])

model_fn = sys.argv[2]

with open(model_fn + ".w2idx", "rb") as w2idx_f:
    w2idx = pickle.load(w2idx_f)

error_count = 0 
num_sents = 0
for sentence, target, alternative in zip(inp["sentence"], inp["target"], inp["alternative"]):
    num_sents += 1
    for word in sentence.split(): 
        if word.lower().strip() not in w2idx:
            print(word.lower().strip(), "---", sentence)
            error_count += 1
    if target.lower().strip() not in w2idx:
        print(target)
        error_count += 1 
    if alternative.lower().strip() not in w2idx:
        print(alternative)
        error_count += 1

print("{} UNKs in {} items".format(error_count, num_sents))
