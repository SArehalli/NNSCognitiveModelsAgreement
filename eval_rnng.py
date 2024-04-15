import subprocess

import stanfordnlp
from stanfordnlp.server import CoreNLPClient
import argparse

import sys
sys.path.insert(0, "./colorlessgreenRNNs/src/language_models")
from dictionary_corpus import Dictionary

import pandas as pd
import numpy as np

def sample_prob_scoring(target, alternative):
    denom = np.logaddexp(target, alternative)
    correct = target - denom
    
    return 100 * np.exp(correct)

def max_prob_scoring(target, alternative):
    correct = 1 if target > alternative else 0 

    return 100 * correct

def surprisal_scoring(target, alternative):
    return -target + alternative

def parseToString(parse):
    """ Convert stanfordnlp parsetree to a string that the rnng will accept """
    if parse.value == "ROOT":
        parse = parse.child[0]

    if len(parse.child) == 0:
        return parse.value

    parse.value, *rest = parse.value.split("-")
    if parse.value == "":
        parse.value, *rest = rest

    s = "({} ".format(parse.value)

    for child in parse.child:
        s += parseToString(child)
    s += ")"

    return s

def logsumexp(l):
    s, *l = l
    for x in l:
        s = np.logaddexp(s,x)  
    return s


parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--mgen", type=str, required=True)
parser.add_argument("--mdis", type=str, required=True)
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--scoring", type=str, default="max")
parser.add_argument("--paradigm", type=str, default="prod")
parser.add_argument("--out", type=str)
parser.add_argument("--prefix", type=str)
parser.add_argument("--latex", action="store_true")
parser.add_argument("--flip", action="store_true")
parser.add_argument("--verbose", action="store_true")

args = parser.parse_args()

scorers = []
args.scoring = set(args.scoring.split(","))

if "max" in args.scoring:
    scorers.append(("max", max_prob_scoring))
if "sample" in args.scoring:
    scorers.append(("sample", sample_prob_scoring))
if "surp" in args.scoring:
    scorers.append(("surprisal", surprisal_scoring))

inp = pd.read_csv(args.input)

# Get parse trees from the stanford parser
rnng_inputs = []
tars, alts = [],[]
with CoreNLPClient(annotators=["parse"], timeout=30000, memory="8G") as client:
    print(len(inp["sentence"]))
    for prompt, *endings in zip(inp["sentence"], inp["target"], inp["alternative"]):
        if args.paradigm == "prod":
            tar_split = len(endings[0].split(","))
            tars.append(tar_split)
            endings = [x for y in endings for x in y.split(",")]
            total_endings = len(endings)
            alts.append(total_endings - tar_split)
            for ending in endings:
                sentence = prompt + " " + ending 
                parse = parseToString(client.annotate(sentence).sentence[0].parseTree)
                rnng_inputs.append(parse + "\n")
        else:
                sentence = prompt
                parse = parseToString(client.annotate(sentence).sentence[0].parseTree)
                rnng_inputs.append(parse + "\n")

test_file = args.input.split(".")[0] + ".rnng"
test_oracle = args.input.split(".")[0] + ".oracle"
path = "/".join(args.input.split("/")[:-1]) + "/"
with open(test_file, "w") as rnng_inp_f:
    rnng_inp_f.writelines(rnng_inputs)

if args.verbose:
    print("parses for rnng")
    for line in rnng_inputs:
        print("\t" + line)

sampling_template = "rnng/build/nt-parser/nt-parser --cnn-mem 1700 -x -T {}/{}train.oracle -p {} -C {} -P --lstm_input_dim 128 --hidden_dim 128 -m {} --alpha 0.8 -s 100"

join_template = "rnng/build/nt-parser/nt-parser-gen -x -T {}/{}train-gen.oracle --clusters rnng/clusters-train-berk.txt --input_dim 256 --lstm_input_dim 256 -p {}test-samples.trees -m {}"

marginals_template = "rnng/utils/is-estimate-marginal-llh.pl {} 100 {}test-samples.props {}test-samples.llh"

for model_num, (mgen, mdis) in enumerate(zip(args.mgen.split(","), args.mdis.split(","))):
    # Automate running the RNNGs 
    cmd1 = "python2 rnng/get_oracle.py {}train.ptb {}".format(args.prefix, test_file)
    if args.verbose: print(cmd1)
    r1 = subprocess.run(cmd1.split(), check=True, stdout=open(test_oracle, "w"))

    cmd2 = sampling_template.format(args.data, args.prefix, test_oracle, test_file, mdis)
    if args.verbose: print(cmd2)
    r2 = subprocess.run(cmd2.split(), stdout=open(path + "test-samples.props", "w"), check=True)

    cmd3 = "rnng/utils/cut-corpus.pl 3 {}test-samples.props".format(path)
    if args.verbose: print(cmd3)
    r3 = subprocess.run(cmd3.split(), stdout=open(path+"test-samples.trees", "w"), check=True)

    cmd4 = join_template.format(args.data, args.prefix, path, mgen)
    if args.verbose: print(cmd4)
    r4 = subprocess.run(cmd4.split(), stdout=open(path + "test-samples.llh","w"), check=True)

    cmd5 = marginals_template.format(len(rnng_inputs), path, path)
    if args.verbose: print(cmd5)
    r5 = subprocess.run(cmd5.split(), stdout=open(path + "llh.txt","w"), stderr=open(path+"rescored.trees","w"), check=True)

    # Extract scores from RNNG output
    scores = []
    with open(path+"rescored.trees", "r") as rescored_f:
        for line in rescored_f:
            score = float(line.split("|||")[1])
            scores.append(score)

    # Scoring/linking functions
    if args.paradigm == "prod":
        # Merge multiple target/alt scores (he/she/it for Bock1999)
        i = 0
        targets = []
        alternatives = []
        for tar,alt in zip(tars,alts):
            target_opts = scores[i:i+tar]
            targets.append(logsumexp(target_opts))
            i += tar

            alt_opts = scores[i:i+alt]
            alternatives.append(logsumexp(alt_opts))
            i += alt

        assert len(targets) == len(alternatives)

        corrects = {}
        for scorer in scorers:
            correct = [scorer[1](target, alt) for target, alt in zip(targets, alternatives)]
            corrects[scorer[0]] = correct

        inp["rnng {} target score".format(model_num)] = targets
        inp["rnng {} alternative score".format(model_num)] = alternatives

    if args.paradigm == "comp":
        corrects = {}
        corrects["surprisal"] = [-x for x in scores]

    for scorer_name, _ in scorers:
        inp["rnng {} correct ({})".format(model_num, scorer_name)] = corrects[scorer_name]

# Write out to csv
if args.out is None: args.out = args.input
inp.to_csv(args.out)
