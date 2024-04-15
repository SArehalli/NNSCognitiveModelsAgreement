import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import argparse
import pickle
import tqdm

import sys
sys.path.insert(0, "./CCGMultitask/")
from model import MultiTaskModel

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--out", type=str)
parser.add_argument("--cuda", action="store_true")

args = parser.parse_args()

# Load models from comma-separated arg
model_fns = args.model.split(",")
models = []
w2idxs = []
for model_fn in model_fns:
    with open(model_fn + ".w2idx", "rb") as w2idx_f:
        w2idx = pickle.load(w2idx_f)
    w2idxs.append(w2idx)

    with open(model_fn + ".c2idx", "rb") as c2idx_f:
        c2idx = pickle.load(c2idx_f)

    model = MultiTaskModel(len(w2idx.keys()), 650, 650, 
                           [len(w2idx.keys()), len(c2idx.keys())], 2,)
    model.load_state_dict(torch.load(model_fn + ".pt", 
                                     map_location = torch.device("cuda" if args.cuda 
                                                             else "cpu")))

    if args.cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    model.eval()
    models.append(model)


for model_num, model in (enumerate(models)):
    mode     
