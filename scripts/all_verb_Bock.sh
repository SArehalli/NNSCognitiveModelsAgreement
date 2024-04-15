#!/bin/sh

# run from AgreementRNNs/

model='CCGMultitask/models/augment/augment_.50_{}_sgd_continue'
model2='CCGMultitask/models/augment/augment_1.00_{}_sgd_continue'

# Bock 1992 rc/pp
echo 'Bock1992'
python eval_ccgmulti.py --model ${model/'{}'/0},${model/'{}'/1},${model/'{}'/2},${model/'{}'/3},${model/'{}'/4}  --input evalsets/Bock1992/items.csv --scoring max,sample --paradigm allverb_prod --col_tag lmccg --out all_verbs/bock1992_lmccg.csv
python eval_ccgmulti.py --model ${model2/'{}'/0},${model2/'{}'/1},${model2/'{}'/2},${model2/'{}'/3},${model2/'{}'/4}  --input evalsets/Bock1992/items.csv --scoring max,sample --paradigm allverb_prod --col_tag lmonly --out all_verbs/bock1992_lmonly.csv
python eval_ccgmulti.py --model ${model/'{}'/0},${model/'{}'/1},${model/'{}'/2},${model/'{}'/3},${model/'{}'/4}  --input evalsets/Bock1992/items.csv --scoring max,sample --paradigm allverb_byverb --col_tag lmccg --out all_verbs/bock1992_byverb_lmccg.csv
python eval_ccgmulti.py --model ${model2/'{}'/0},${model2/'{}'/1},${model2/'{}'/2},${model2/'{}'/3},${model2/'{}'/4}  --input evalsets/Bock1992/items.csv --scoring max,sample --paradigm allverb_byverb --col_tag lmonly --out all_verbs/bock1992_byverb_lmonly.csv

