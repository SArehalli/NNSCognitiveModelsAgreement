from nltk.tree import Tree
import argparse

NP_S = ["NN", "NP"]
NP_P = ["NNS", "NPS"]

V_S = ["VHZ", "VVZ", "VBZ"]
V_ = ["VHP", "VBP"]
V_P = ["VVP"]

def get_num(tag, word):
    if tag in NP_S + NP_P:
        if tag in NP_S:
            return "S"
        else:
            return "P"
    if tag in V_S:
        return "S"
    if tag in V_P:
        return "P"
    if tag in V_:
        if word == ["have", "are", "were"]:
            return "P"
        if word == ["has", "am", "was"]:
            return "S"
        else:
            print("V*P: {}".format(word))
    return None

def compute_agree(st):
    assert st.label() == "S"
    subj = None
    vp = None
    for child in st:
        if child.label() == "NP":
            if subj is None:
                subj = child
        if child.label() == "VP":
            if vp is None:
                vp = child
    
    if (subj is None) or (vp is None):
        return None
    subj_N = [(tag, word) for (word, tag) in subj.pos() if tag[0] == "N"]

    if len(subj_N) < 1:
        return None
    
    subj_num = get_num(*subj_N[0])

    if subj_num is None:
        return None

    verb_N = [(tag, word) for (word, tag) in vp.pos() if tag[0] == "V"]

    if len(verb_N) < 1:
        return None
    
    verb_num = get_num(*verb_N[0])

    if subj_num is None:
        return None
    if verb_num is None:
        return None

    print("SUBJ", subj, "\nVERB", vp, "\nNUMS", subj_num, verb_num)

    return subj_num == verb_num

def is_demb_pp(t):
    for st in t.subtrees(lambda x : x.label() == "PP"):
        for stc in st:
            for st2 in stc.subtrees(lambda x : x.label() == "PP"):
                print(st)
                return 1
    return 0

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str)

args = parser.parse_args()

train_trees = []
with open(args.file) as train_f:
    train_trees = [Tree.fromstring(s) for s in train_f]

min_agrees = 0
min_corr_agrees = 0
demb_pps = 0

for tree in train_trees:
    for st in tree.subtrees(lambda t: t.label() == "S"):
        out = compute_agree(st)
        
        if out is not None:
            min_agrees += 1
        if out:
            min_corr_agrees += 1
    demb_pps += is_demb_pp(tree)
        
print(min_agrees, min_corr_agrees, min_corr_agrees/min_agrees)
print(demb_pps)
