#!/bin/sh

# run from AgreementRNNs/

model='CCGMultitask/models/wiki_lm/lm_{}_sgd'

# Bock 1992 rc/pp
echo 'Bock1992'
python eval_ccgmulti.py --model ${model/'{}'/0},${model/'{}'/1},${model/'{}'/2},${model/'{}'/3},${model/'{}'/4}  --input evalsets/Bock1992/items.csv --scoring max,sample --paradigm prod --col_tag wiki_lm

echo 'Franck2002'
python eval_ccgmulti.py --model ${model/'{}'/0},${model/'{}'/1},${model/'{}'/2},${model/'{}'/3},${model/'{}'/4}  --input evalsets/Franck2002/items.csv --scoring max,sample --paradigm prod --col_tag wiki_lm

echo 'Haskell2005'
python eval_ccgmulti.py --model ${model/'{}'/0},${model/'{}'/1},${model/'{}'/2},${model/'{}'/3},${model/'{}'/4}  --input evalsets/HaskellMacdonald2011/items.csv --scoring max,sample --paradigm prod --col_tag wiki_lm

echo 'HumphreysBock2005'
python eval_ccgmulti.py --model ${model/'{}'/0},${model/'{}'/1},${model/'{}'/2},${model/'{}'/3},${model/'{}'/4}  --input evalsets/HumphreysBock2005/items.csv --scoring max,sample --paradigm prod --col_tag wiki_lm

echo 'ParkerAn2018'
python eval_ccgmulti.py --model ${model/'{}'/0},${model/'{}'/1},${model/'{}'/2},${model/'{}'/3},${model/'{}'/4}  --input evalsets/ParkerAn2018/items.comp.csv --scoring surprisal --paradigm comp --col_tag wiki_lm

echo 'Wagers2009'
python eval_ccgmulti.py --model ${model/'{}'/0},${model/'{}'/1},${model/'{}'/2},${model/'{}'/3},${model/'{}'/4}  --input evalsets/Wagers2009/23_illusion/items.csv --scoring surprisal --paradigm comp --col_tag wiki_lm
