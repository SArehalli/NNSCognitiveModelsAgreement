import csv
import sys
import pickle

with open(sys.argv[3], "rb") as w2idx_f:
    w2idx = pickle.load(w2idx_f)

inp = csv.DictReader(open(sys.argv[1]))
cond = sys.argv[2]
for row in inp:
    if row["condition"] == cond:
        print("\\item {}".format(" ".join([w if w.lower() in w2idx else "<UNK>" for w in row["sentence"].split()])))
