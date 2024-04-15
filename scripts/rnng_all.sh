#!/bin/sh

# run from AgreementRNNs/
# Bock 1992
echo 'Bock1992'
python eval_rnng.py --mgen models/rnng_gen-1,models/rnng_gen-2,models/rnng_gen-3,models/rnng_gen-4,models/rnng_gen-5 --mdis models/rnng_dis-1,models/rnng_dis-2,models/rnng_dis-3,models/rnng_dis-4,models/rnng_dis-5 --data rnng --input evalsets/Bock1992/items.csv  --prefix small- --paradigm prod --scoring max,sample

# Franck 2002 syntactic distance
echo 'Franck'
python eval_rnng.py --mgen models/rnng_gen-1,models/rnng_gen-2,models/rnng_gen-3,models/rnng_gen-4,models/rnng_gen-5 --mdis models/rnng_dis-1,models/rnng_dis-2,models/rnng_dis-3,models/rnng_dis-4,models/rnng_dis-5 --data rnng --input evalsets/Franck2002/items.csv  --prefix small- --paradigm prod --scoring max,sample

# Bock 1999 pronouns vs subj verb
echo 'Bock1999'
python eval_rnng.py --mgen models/rnng_gen-1,models/rnng_gen-2,models/rnng_gen-3,models/rnng_gen-4,models/rnng_gen-5 --mdis models/rnng_dis-1,models/rnng_dis-2,models/rnng_dis-3,models/rnng_dis-4,models/rnng_dis-5 --data rnng --input evalsets/Bock1999/items.csv  --prefix small- --paradigm prod --scoring max,sample


# Haskell & MacDonald Conjunction linear distance
echo 'HaskellMacDonald'
python eval_rnng.py --mgen models/rnng_gen-1,models/rnng_gen-2,models/rnng_gen-3,models/rnng_gen-4,models/rnng_gen-5 --mdis models/rnng_dis-1,models/rnng_dis-2,models/rnng_dis-3,models/rnng_dis-4,models/rnng_dis-5 --data rnng --input evalsets/HaskellMacdonald2011/items.csv  --prefix small- --paradigm prod --scoring max,sample


# Humphreys & Bock 2005 distributivity
echo 'HumphreysBock'
python eval_rnng.py --mgen models/rnng_gen-1,models/rnng_gen-2,models/rnng_gen-3,models/rnng_gen-4,models/rnng_gen-5 --mdis models/rnng_dis-1,models/rnng_dis-2,models/rnng_dis-3,models/rnng_dis-4,models/rnng_dis-5 --data rnng --input evalsets/HumphreysBock2005/items.csv  --prefix small- --paradigm prod --scoring max,sample


# Parker & An 2018 argument status
echo 'ParkerAn'
python eval_rnng.py --mgen models/rnng_gen-1,models/rnng_gen-2,models/rnng_gen-3,models/rnng_gen-4,models/rnng_gen-5 --mdis models/rnng_dis-1,models/rnng_dis-2,models/rnng_dis-3,models/rnng_dis-4,models/rnng_dis-5 --data rnng --input evalsets/ParkerAn2018/items.csv  --prefix small- --paradigm prod --scoring max,sample


python eval_rnng.py --mgen models/rnng_gen-1,models/rnng_gen-2,models/rnng_gen-3,models/rnng_gen-4,models/rnng_gen-5 --mdis models/rnng_dis-1,models/rnng_dis-2,models/rnng_dis-3,models/rnng_dis-4,models/rnng_dis-5 --data rnng --input evalsets/ParkerAn2018/items.csv  --prefix small- --paradigm comp --scoring surp

# Wagers 2009 Agreement

python eval_rnng.py --mgen models/rnng_gen-1,models/rnng_gen-2,models/rnng_gen-3,models/rnng_gen-4,models/rnng_gen-5 --mdis models/rnng_dis-1,models/rnng_dis-2,models/rnng_dis-3,models/rnng_dis-4,models/rnng_dis-5 --data rnng --input evalsets/Wagers2009/1_Baseline/items.csv  --prefix small- --paradigm comp --scoring surp

# Wagers 2009 Illusion of Grammaticality

python eval_rnng.py --mgen models/rnng_gen-1,models/rnng_gen-2,models/rnng_gen-3,models/rnng_gen-4,models/rnng_gen-5 --mdis models/rnng_dis-1,models/rnng_dis-2,models/rnng_dis-3,models/rnng_dis-4,models/rnng_dis-5 --data rnng --input evalsets/Wagers2009/23_illusion/items.csv  --prefix small- --paradigm comp --scoring surp
