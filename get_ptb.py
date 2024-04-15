from nltk.corpus import ptb
from nltk.tree import Tree
import sys

def canonicalize_nts(tree, strict):
    """ Following section 6.2 of Eisner 2001 """
    for child in tree:
        canonicalize_nt(tree, strict)
        if isinstance(child, Tree):
            canonicalize_nts(child, strict)
    return tree

def canonicalize_nt(node, strict):
    label, *mods = node.label().split("-")
    if label == "":
        label = mods[0]
        mods = mods[1:]
    if not strict:
        for mod in mods:
            if mod in ["TMP", "LOC", "ADV", "PRD"]:
                label += "-" + mod
            elif label == "S" and mod in ["NOM", "SBJ", "PRP", "TPC"]:
                label += "-" + mod
    node.set_label(label)

# Use the reported 2-21/24/23 split from Dyer et al 2016
wsj_ids = [f for f in ptb.fileids() if "WSJ" in f]
train_ids = [f for f in wsj_ids if int(f.split("/")[1]) >= 2 and 
                                   int(f.split("/")[1]) <= 21]
dev_ids = [f for f in wsj_ids if int(f.split("/")[1]) == 24]
test_ids = [f for f in wsj_ids if int(f.split("/")[1]) == 23]

t = ptb.parsed_sents()[0]

strict = True if sys.argv[1] == "-s" else False
if strict:
    print("Running with minimal POS")
# Write out the trees to file, one line per tree     
with open("train.ptb", "w") as train, \
     open("dev.ptb", "w") as dev, \
     open("test.ptb", "w") as test:
    train_lines = [" ".join(str(canonicalize_nts(tree, strict)).split()) + "\n" 
                   for tree in ptb.parsed_sents(train_ids)]
    print("writing {} train parses".format(len(train_lines)))
    train.writelines(train_lines)
    dev_lines = [" ".join(str(canonicalize_nts(tree, strict)).split()) + "\n" 
                 for tree in ptb.parsed_sents(dev_ids)]
    print("writing {} dev parses".format(len(dev_lines)))
    dev.writelines(dev_lines)
    test_lines = [" ".join(str(canonicalize_nts(tree, strict)).split()) + "\n" 
                  for tree in ptb.parsed_sents(test_ids)]
    print("writing {} test parses".format(len(test_lines)))
    test.writelines(test_lines)
