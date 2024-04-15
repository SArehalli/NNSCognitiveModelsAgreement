import sys
import pickle
import pandas as pd

model_fn = sys.argv[1]
orig = pd.read_csv(sys.argv[2])
new = pd.read_csv(sys.argv[3])

with open(model_fn + ".w2idx", "rb") as w2idx_f:
    w2idx = pickle.load(w2idx_f)

edits = []
unk_sents = []
edit_sents = []
for orig, new in zip(orig["sentence"], new["sentence"]):
    unk_item = []
    edit_item = []
    edited = False
    for word_o, word_n in zip(orig.split(), new.split()):
        if word_o != word_n:
            edits.append((word_o, word_n))
            edit_item.append("{} ({})".format(word_n.lower(), word_o.lower()))
        if word_n.lower() not in w2idx:
            w = "<UNK> ({})".format(word_n.lower())
            unk_item.append(w)
            edit_item.append(w)
            edited = True
        else:
            unk_item.append(word_n.lower())
            edit_item.append(word_n.lower())
    if edited:
        unk_sents.append(" ".join(unk_item))
    edit_sents.append(" ".join(edit_item))

for sent in unk_sents:
    print(sent)

print("----")
edits = list(set(edits))
for old, new in edits:
    print("{} -> {}".format(old, new))

print("----")
for sent in edit_sents:
    print(sent)
