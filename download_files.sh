#!/bin/bash

mkdir colorlessgreenRNNs/data/lm/English
wget -P colorlessgreenRNNs/data/lm/English/ https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English/train.txt
wget -P colorlessgreenRNNs/data/lm/English/ https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English/test.txt
wget -P colorlessgreenRNNs/data/lm/English/ https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English/valid.txt
wget -P colorlessgreenRNNs/data/lm/English/ https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English/vocab.txt
      
wget https://dl.fbaipublicfiles.com/colorless-green-rnns/best-models/English/hidden650_batch128_dropout0.2_lr20.0.pt
