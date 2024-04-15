import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plot

import argparse
import pickle
import tqdm
import sys
sys.path.insert(0, "./CCGMultitask/")
from model import MultiTaskModel

import pandas as pd
import seaborn as sns

def indexify(word, w2idx):
    """ Convert word to an index into the embedding matrix """
    try:
        return w2idx[word] if word in w2idx else w2idx["<oov>"]
    except:
        print("error on ", word)

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

def get_score(targets, next_pred, w2idx):
    """ Get a score for a set of target words """
    score = -np.inf
    for target in targets:
        target = indexify(target, w2idx)
        score = np.logaddexp(score, next_pred[target].item())
    return score


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
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

# Load experimental data csv
inp = pd.read_csv(args.input)

# Load model
model_fns = args.model.split(",")
w2idxs = []
c2idxs = []
for model_fn in model_fns:
    with open(model_fn + ".w2idx", "rb") as w2idx_f:
        w2idx = pickle.load(w2idx_f)
    w2idxs.append(w2idx)

    with open(model_fn + ".c2idx", "rb") as c2idx_f:
        c2idx = pickle.load(c2idx_f)
    c2idxs.append(c2idx)

if args.paradigm == "allverb_prod":
    with open("./all_verbs/verb_dist.dict", "rb") as in_f:
        verb_dist = pickle.load(in_f)

if args.paradigm == "allverb_byverb":
    with open("./all_verbs/verb_pairs.dict", "rb") as in_f:
        verb_pairs = pickle.load(in_f)
    out = []
    for _, row in inp.iterrows():
        for sing, plural in verb_pairs:
            if (sing not in w2idxs[0]) or (plural not in w2idxs[0]): continue
            new = {"item":row["item"], "condition":row["condition"], "sentence":row["sentence"], "word":sing, "word_pl":plural}
            out.append(new)

    out_df = pd.DataFrame.from_records(out)
    

# Get scorers from comma-separated arg
scorers = []
args.scoring = set(args.scoring.split(","))
if "max" in args.scoring:
    scorers.append(("max", max_prob_scoring))
if "sample" in args.scoring:
    scorers.append(("sample", sample_prob_scoring))
if "surp" in args.scoring:
    scorers.append(("surprisal", surprisal_scoring))

for model_num, model_fn in (enumerate(model_fns)):
    corrects = {}
    target_s = []
    alt_s = []

    surpss = []
    total_surps = []

    # Load model
    model = MultiTaskModel(len(w2idxs[0].keys()), 650, 650, 
                           [len(w2idxs[0].keys()), len(c2idxs[0].keys())], 2)
    model.load_state_dict(torch.load(model_fn + ".pt", 
                                     map_location = torch.device("cuda" if args.cuda 
                                                             else "cpu")))
    if args.cuda:
        model = model.cuda()
    else:
        model = model.cpu()
    model.eval()
 
    with torch.no_grad():
        for sentence, target, alternative in tqdm.tqdm(list(zip(inp["sentence"], inp["target"], inp["alternative"]))):
            input = torch.LongTensor([indexify(w.lower(), w2idxs[model_num]) for w in sentence.split()])

            if args.cuda:
                input.cuda()

            out, _, _ = model(input.view(-1, 1), model.init_hidden(1))

            if args.paradigm == "comp":
                # Get surprisals for the first 10 words
                surps = []
                for i, word_idx in enumerate(input[1:11]):
                    surps.append(-F.log_softmax(out[i], dim=-1).view(-1)[word_idx].item())
                for i in range(10 - len(input[1:11])):
                    surps.append(-1)
                surpss.append(surps)
                # Get surprisals over the fill sentence
                total_surps.append(sum([-F.log_softmax(out[i], dim=-1).view(-1)[word_idx].item() 
                                        for i,word_idx in enumerate(input[1:])]))

            elif args.paradigm == "prod":
                next_pred = out[-1] 

                # use scorers/linking functions to eval on the target and alternative words' scores
                next_pred = F.log_softmax(next_pred, dim=-1).view(-1)
                target_scores = get_score(target.split(","), next_pred, w2idxs[model_num])
                alt_scores = get_score(alternative.split(","), next_pred, w2idxs[model_num])

                for scorer in scorers:
                    correct = scorer[1](target_scores, alt_scores)
                    corrects[scorer[0]] = corrects.get(scorer[0], []) + [correct]

                target_s.append(target_scores.item())
                alt_s.append(alt_scores.item())

            elif args.paradigm == "allverb_prod_weight":
                singular,plural = 0.0,0.0
                next_pred = F.log_softmax(out[-1], dim=-1).view(-1)

                for word, ts in verb_dist.items():
                    vbp, vbz = -np.inf, -np.inf
                    idx = w2idxs[model_num].get(word, None)
                    if idx is None: continue 
                    if ts["VBP"] > 0:
                        vbp = next_pred[idx] + np.log(ts["VBP"])
                        plural = np.logaddexp(plural, vbp)
                    if ts["VBZ"] > 0:
                        vbz = next_pred[idx] + np.log(ts["VBZ"])
                        singular = np.logaddexp(singular, vbz)
                    if args.verbose: print(sentence + "... " + word, vbp, vbz)
                singular = np.exp(singular.item())
                plural = np.exp(plural.item())
                target_s.append(plural/(singular + plural))
                alt_s.append(singular/(singular + plural))

            elif args.paradigm == "allverb_prod":
                singular,plural = 0.0,0.0
                next_pred = F.log_softmax(out[-1], dim=-1).view(-1)

                for word, ts in verb_dist.items():
                    vbp, vbz = -np.inf, -np.inf
                    idx = w2idxs[model_num].get(word, None)
                    if idx is None: continue 
                    if ts["VBP"] > 0:
                        vbp = next_pred[idx] 
                        plural = np.logaddexp(plural, vbp)
                    if ts["VBZ"] > 0:
                        vbz = next_pred[idx]
                        singular = np.logaddexp(singular, vbz)
                    if args.verbose: print(sentence + "... " + word, vbp, vbz)
                singular = np.exp(singular.item())
                plural = np.exp(plural.item())
                target_s.append(plural/(singular + plural))
                alt_s.append(singular/(singular + plural))

            elif args.paradigm == "allverb_byverb":
                next_pred = F.log_softmax(out[-1], dim=-1).view(-1)

                for singular, plural in verb_pairs:
                    pl_idx = w2idxs[model_num].get(plural, None)
                    sg_idx = w2idxs[model_num].get(singular, None)
                    if (sg_idx is None) or (pl_idx is None): continue 

                    p_prob = np.exp(next_pred[pl_idx]).item()
                    s_prob = np.exp(next_pred[sg_idx]).item()
                    
                    if args.verbose: print(sentence + "... " + singular, plural, p_prob, s_prob)
                    target_s.append(p_prob/(s_prob + p_prob))
                    alt_s.append(s_prob/(s_prob + p_prob))



                

    # write out to csv
    if args.paradigm == "prod":
        for scorer_name, _ in scorers:
            inp["lstm{} {} correct ({})".format(args.col_tag, model_num, scorer_name)] = corrects[scorer_name]

        inp["lstm{} {} target score".format(args.col_tag, model_num)] = target_s
        inp["lstm{} {} alternative score".format(args.col_tag, model_num)] = alt_s

    if args.paradigm == "allverb_byverb":
        out_df["{} plural".format(model_num)] = target_s
        out_df["{} singular".format(model_num)] = alt_s

    if args.paradigm[:len("allverb_prod")] == "allverb_prod":
        inp["lstm{} {} plural score".format(args.col_tag, model_num)] = target_s
        inp["lstm{} {} singular score".format(args.col_tag, model_num)] = alt_s

    if args.paradigm == "comp":
        surpss = np.array(surpss)
        surpss = surpss.transpose()
        for i, position in enumerate(surpss):
            print(len(position))
            inp["(lstm{} {} at word {})".format(args.col_tag, model_num, i + 1)] = position
        inp["lstm{} {}  (surprisal)".format(args.col_tag, model_num)] = total_surps



if args.out is None: args.out = args.input

# disgusting hack to get the allverb stuff to work :(
if args.paradigm == "allverb_byverb":
    out_df.to_csv(args.out)
else: inp.to_csv(args.out)
