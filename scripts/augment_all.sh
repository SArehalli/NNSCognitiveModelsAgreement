#!/bin/sh

# run from AgreementRNNs/

model='CCGMultitask/models/augment/augment_.50_{}_sgd_continue'
model2='CCGMultitask/models/augment/augment_1.00_{}_sgd_continue'

# Bock 1992 rc/pp
echo 'Bock1992'
python eval_ccgmulti.py --model ${model/'{}'/0},${model/'{}'/1},${model/'{}'/2},${model/'{}'/3},${model/'{}'/4}  --input evalsets/Bock1992/items.csv --scoring max,sample --paradigm prod --col_tag augment
python eval_ccgmulti.py --model ${model2/'{}'/0},${model2/'{}'/1},${model2/'{}'/2},${model2/'{}'/3},${model2/'{}'/4}  --input evalsets/Bock1992/items.csv --scoring max,sample --paradigm prod --col_tag lmaug

echo 'Franck2002'
python eval_ccgmulti.py --model ${model/'{}'/0},${model/'{}'/1},${model/'{}'/2},${model/'{}'/3},${model/'{}'/4}  --input evalsets/Franck2002/items.csv --scoring max,sample --paradigm prod --col_tag augment
python eval_ccgmulti.py --model ${model2/'{}'/0},${model2/'{}'/1},${model2/'{}'/2},${model2/'{}'/3},${model2/'{}'/4}  --input evalsets/Franck2002/items.csv --scoring max,sample --paradigm prod --col_tag lmaug

echo 'Haskell2005'
python eval_ccgmulti.py --model ${model/'{}'/0},${model/'{}'/1},${model/'{}'/2},${model/'{}'/3},${model/'{}'/4}  --input evalsets/HaskellMacdonald2011/items.csv --scoring max,sample --paradigm prod --col_tag augment
python eval_ccgmulti.py --model ${model2/'{}'/0},${model2/'{}'/1},${model2/'{}'/2},${model2/'{}'/3},${model2/'{}'/4}  --input evalsets/HaskellMacdonald2011/items.csv --scoring max,sample --paradigm prod --col_tag lmaug

echo 'HumphreysBock2005'
python eval_ccgmulti.py --model ${model/'{}'/0},${model/'{}'/1},${model/'{}'/2},${model/'{}'/3},${model/'{}'/4}  --input evalsets/HumphreysBock2005/items.csv --scoring max,sample --paradigm prod --col_tag augment
python eval_ccgmulti.py --model ${model2/'{}'/0},${model2/'{}'/1},${model2/'{}'/2},${model2/'{}'/3},${model2/'{}'/4}  --input evalsets/HumphreysBock2005/items.csv --scoring max,sample --paradigm prod --col_tag lmaug

echo 'ParkerAn2018'
python eval_ccgmulti.py --model ${model/'{}'/0},${model/'{}'/1},${model/'{}'/2},${model/'{}'/3},${model/'{}'/4}  --input evalsets/ParkerAn2018/items.comp.csv --scoring surprisal --paradigm comp --col_tag augment
python eval_ccgmulti.py --model ${model2/'{}'/0},${model2/'{}'/1},${model2/'{}'/2},${model2/'{}'/3},${model2/'{}'/4}  --input evalsets/ParkerAn2018/items.comp.csv --scoring surprisal --paradigm comp --col_tag lmaug

echo 'Wagers2009'
python eval_ccgmulti.py --model ${model/'{}'/0},${model/'{}'/1},${model/'{}'/2},${model/'{}'/3},${model/'{}'/4}  --input evalsets/Wagers2009/23_illusion/items.csv --scoring surprisal --paradigm comp --col_tag augment
python eval_ccgmulti.py --model ${model2/'{}'/0},${model2/'{}'/1},${model2/'{}'/2},${model2/'{}'/3},${model2/'{}'/4}  --input evalsets/Wagers2009/23_illusion/items.csv --scoring surprisal --paradigm comp --col_tag lmaug
