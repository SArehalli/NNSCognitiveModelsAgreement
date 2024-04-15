#!/bin/sh

# run from AgreementRNNs/

model='../models_pt/ccglm_tied_011422/augment_0.5_{}_sgd'

# Bock 1992 rc/pp
echo 'Bock1992'
python eval_ccgmulti.py --model ${model/'{}'/0},${model/'{}'/1},${model/'{}'/2},${model/'{}'/3}  --input evalsets/Bock1992/items.csv --scoring max,sample --paradigm prod --col_tag augment_tied

echo 'Franck2002'
python eval_ccgmulti.py --model ${model/'{}'/0},${model/'{}'/1},${model/'{}'/2},${model/'{}'/3}  --input evalsets/Franck2002/items.csv --scoring max,sample --paradigm prod --col_tag augment_tied

echo 'Haskell2005'
python eval_ccgmulti.py --model ${model/'{}'/0},${model/'{}'/1},${model/'{}'/2},${model/'{}'/3}  --input evalsets/HaskellMacdonald2011/items.csv --scoring max,sample --paradigm prod --col_tag augment_tied

echo 'HumphreysBock2005'
python eval_ccgmulti.py --model ${model/'{}'/0},${model/'{}'/1},${model/'{}'/2},${model/'{}'/3}  --input evalsets/HumphreysBock2005/items.csv --scoring max,sample --paradigm prod --col_tag augment_tied

echo 'ParkerAn2018'
python eval_ccgmulti.py --model ${model/'{}'/0},${model/'{}'/1},${model/'{}'/2},${model/'{}'/3}  --input evalsets/ParkerAn2018/items.comp.csv --scoring surprisal --paradigm comp --col_tag augment_tied

echo 'Wagers2009'
python eval_ccgmulti.py --model ${model/'{}'/0},${model/'{}'/1},${model/'{}'/2},${model/'{}'/3}  --input evalsets/Wagers2009/23_illusion/items.csv --scoring surprisal --paradigm comp --col_tag augment_tied
