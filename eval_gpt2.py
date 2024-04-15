from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plot

import argparse
import pickle
import tqdm

import pandas as pd
import seaborn as sns

def sample_prob_scoring(target, alternative):
    """ sampling linking hypothesis in log-space """
    denom = np.logaddexp(target.item(), alternative.item())
    correct = target.item() - denom
    
    return 100 * np.exp(correct)

def max_prob_scoring(target, alternative):
    """ max prob linking hypothesis in log-space """
    correct = 1 if target > alternative else 0 

    return 100 * correct

def surprisal_scoring(target, alternative):
    """ surprisal difference scoring """
    return -target + alternative

def get_score(targets, next_pred, tokenizer):
    """ Get a score for a set of target words """
    score = -np.inf
    for target in targets:
        target = tokenizer(" " + target)["input_ids"]
        assert len(target) == 1
        target = target[0]
        score = np.logaddexp(score, next_pred[target].item())
    return score


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--scoring", type=str, default="max")
parser.add_argument("--paradigm", type=str, default="prod")
parser.add_argument("--col_tag", type=str, default="")
parser.add_argument("--out", type=str)
parser.add_argument("--latex", action="store_true")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--flip", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--plot", action="store_true")

args = parser.parse_args()

# Make it reproduceable
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)

# Load model
tokenizer = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Load vocab

# Load experimental data csv
inp = pd.read_csv(args.input)

# Get scorers from comma-separated arg
scorers = []
args.scoring = set(args.scoring.split(","))
if "max" in args.scoring:
    scorers.append(("max", max_prob_scoring))
if "sample" in args.scoring:
    scorers.append(("sample", sample_prob_scoring))
if "surp" in args.scoring:
    scorers.append(("surprisal", surprisal_scoring))

corrects = {}
target_s = []
alt_s = []

surpss = []
total_surps = []

with torch.no_grad():
    for sentence, target, alternative in tqdm.tqdm(list(zip(inp["sentence"], inp["target"], inp["alternative"]))):
        input = tokenizer(sentence, return_tensors="pt")
        ids = input["input_ids"]
        tokens = tokenizer.tokenize(sentence)

        out = model(**input, labels=ids)

        if args.paradigm == "comp":
            # Get surprisals for the first 10 words
            surps = []
            for i, word_idx in enumerate(ids[0][1:11]):
                surps.append(-F.log_softmax(out.logits[0][i], dim=-1).view(-1)[word_idx].item())
            for i in range(10 - len(ids[0][1:11])):
                surps.append(-1)
            surpss.append(surps)
            # Get surprisals over the full sentence
            total_surps.append(sum([-F.log_softmax(out.logits[0][i], dim=-1).view(-1)[word_idx].item() 
                                    for i,word_idx in enumerate(ids[1:])]))

        elif args.paradigm == "prod":
            next_pred = out.logits[0][-1] 

            # use scorers/linking functions to eval on the target and alternative words' scores
            next_pred = F.log_softmax(next_pred, dim=-1).view(-1)
            target_scores = get_score(target.split(","), next_pred, tokenizer)
            alt_scores = get_score(alternative.split(","), next_pred, tokenizer)

            for scorer in scorers:
                correct = scorer[1](target_scores, alt_scores)
                corrects[scorer[0]] = corrects.get(scorer[0], []) + [correct]

            target_s.append(target_scores.item())
            alt_s.append(alt_scores.item())

# write out to csv
if args.paradigm == "prod":
    for scorer_name, _ in scorers:
        inp["gpt2 correct ({})".format(scorer_name)] = corrects[scorer_name]

    inp["gpt2 target score"] = target_s
    inp["gpt2 alternative score"] = alt_s

if args.paradigm == "comp":
    surpss = np.array(surpss)
    surpss = surpss.transpose()
    for i, position in enumerate(surpss):
        print(len(position))
        inp["(gpt2 at word {})".format(i + 1)] = position
    inp["gpt2 (surprisal)"] = total_surps


if args.out is None: args.out = args.input
inp.to_csv(args.out)
