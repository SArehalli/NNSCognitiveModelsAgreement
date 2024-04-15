#!/bin/sh

# run from AgreementRNNs/

# Bock 1992 rc/pp
echo 'Bock1992'
python eval_gpt2.py --input evalsets/Bock1992/items.csv --scoring max,sample --paradigm prod --col_tag gpt2

echo 'Franck2002'
python eval_gpt2.py --input evalsets/Franck2002/items.csv --scoring max,sample --paradigm prod --col_tag gpt2

echo 'Haskell2005'
python eval_gpt2.py --input evalsets/HaskellMacdonald2011/items.csv --scoring max,sample --paradigm prod --col_tag gpt2

echo 'HumphreysBock2005'
python eval_gpt2.py --input evalsets/HumphreysBock2005/items.csv --scoring max,sample --paradigm prod --col_tag gpt2

echo 'ParkerAn2018'
python eval_gpt2.py --input evalsets/ParkerAn2018/items.comp.csv --scoring surprisal --paradigm comp --col_tag gpt2

echo 'Wagers2009'
python eval_gpt2.py --input evalsets/Wagers2009/23_illusion/items.csv --scoring surprisal --paradigm comp --col_tag gpt2
