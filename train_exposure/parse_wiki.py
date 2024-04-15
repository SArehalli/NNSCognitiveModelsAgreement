import stanfordnlp
from stanfordnlp.server import CoreNLPClient
import argparse

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

parser = argparse.ArgumentParser()

parser.add_argument("--inf", type=str)
parser.add_argument("--out", type=str)

args = parser.parse_args()

out = []
with CoreNLPClient(annotators=["parse"], timeout=120000, memory="8G") as client:
    with open(args.inf) as in_f:
        for sentence in in_f:
            sentence = " ".join(sentence.split()[:-1])
            print(sentence)
            parse = parseToString(client.annotate(sentence).sentence[0].parseTree)
            out.append(parse + "\n")

with open(args.out, "w") as out_f:
    out_f.writelines(out)
