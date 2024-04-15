import argparse
import pickle
from pattern.en import pluralize

parser = argparse.ArgumentParser()
parser.add_argument("target", type=str)
parser.add_argument("w2idx", type=str)
parser.add_argument("--output", type=str)
args = parser.parse_args()

def check_vocab(w, w2idx):
    xs = w.lower().replace("[", "").replace("]", "")
    if "/" in xs:
        xs = xs.split("/")
    elif "(" in xs:
        xs = [xs, pluralize(xs)]
    else:
        xs = [xs]
    for x in xs:
        if x not in w2idx:
            return "!" + w
    return w 

with open(args.target, "r") as target_f, open(args.w2idx, "rb") as w2idx_f:

    w2idx = pickle.load(w2idx_f)
    targets = target_f.read().split("\n")
    targets = [x.split() for x in targets]
    targets = [[check_vocab(w, w2idx) for w in target] for target in targets]

output = [" ".join(target) for target in targets]

if args.output:
    output = "\n".join(output)
    with open(args.output, "w") as out_f:
        out_f.write(output)
else:
    for line in output:
        print(line)
    

