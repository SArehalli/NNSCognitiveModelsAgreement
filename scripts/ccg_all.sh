#!/bin/sh

# run from AgreementRNNs/

model='CCGMultitask/models/joint_.50_{}_40_06_sgd'
model2='CCGMultitask/models/joint_1.00_{}_40_06_sgd'

echo "test ppls lm"
python CCGMultitask/main.py --eval --prog --test_data CCGMultitask/data/ccg_supertags/ccg.23.common --lm_weight 1 --hid_dim 650 --emb_dim 650 --seq_len 40 --n_layers 2 --batch_size 20 --seed 0 --dropout 0.5 --w2idx ${model2/'{}'/0}.w2idx  --load ${model2/'{}'/0}.pt --data_ccg CCGMultitask/data/ccg_supertags/
python CCGMultitask/main.py --eval --prog --test_data CCGMultitask/data/ccg_supertags/ccg.23.common --lm_weight 1 --hid_dim 650 --emb_dim 650 --seq_len 40 --n_layers 2 --batch_size 20 --seed 0 --dropout 0.5 --w2idx ${model2/'{}'/1}.w2idx  --load ${model2/'{}'/1}.pt --data_ccg CCGMultitask/data/ccg_supertags/
python CCGMultitask/main.py --eval --prog --test_data CCGMultitask/data/ccg_supertags/ccg.23.common --lm_weight 1 --hid_dim 650 --emb_dim 650 --seq_len 40 --n_layers 2 --batch_size 20 --seed 0 --dropout 0.5 --w2idx ${model2/'{}'/2}.w2idx  --load ${model2/'{}'/2}.pt --data_ccg CCGMultitask/data/ccg_supertags/
python CCGMultitask/main.py --eval --prog --test_data CCGMultitask/data/ccg_supertags/ccg.23.common --lm_weight 1 --hid_dim 650 --emb_dim 650 --seq_len 40 --n_layers 2 --batch_size 20 --seed 0 --dropout 0.5 --w2idx ${model2/'{}'/3}.w2idx  --load ${model2/'{}'/3}.pt --data_ccg CCGMultitask/data/ccg_supertags/
python CCGMultitask/main.py --eval --prog --test_data CCGMultitask/data/ccg_supertags/ccg.23.common --lm_weight 1 --hid_dim 650 --emb_dim 650 --seq_len 40 --n_layers 2 --batch_size 20 --seed 0 --dropout 0.5 --w2idx ${model2/'{}'/4}.w2idx  --load ${model2/'{}'/4}.pt --data_ccg CCGMultitask/data/ccg_supertags/
echo "test ppls joint"
python CCGMultitask/main.py --eval --prog --test_data CCGMultitask/data/ccg_supertags/ccg.23.common --lm_weight 1 --hid_dim 650 --emb_dim 650 --seq_len 40 --n_layers 2 --batch_size 20 --seed 0 --dropout 0.5 --w2idx ${model/'{}'/0}.w2idx  --load ${model/'{}'/0}.pt --data_ccg CCGMultitask/data/ccg_supertags/
python CCGMultitask/main.py --eval --prog --test_data CCGMultitask/data/ccg_supertags/ccg.23.common --lm_weight 1 --hid_dim 650 --emb_dim 650 --seq_len 40 --n_layers 2 --batch_size 20 --seed 0 --dropout 0.5 --w2idx ${model/'{}'/1}.w2idx  --load ${model/'{}'/1}.pt --data_ccg CCGMultitask/data/ccg_supertags/
python CCGMultitask/main.py --eval --prog --test_data CCGMultitask/data/ccg_supertags/ccg.23.common --lm_weight 1 --hid_dim 650 --emb_dim 650 --seq_len 40 --n_layers 2 --batch_size 20 --seed 0 --dropout 0.5 --w2idx ${model/'{}'/2}.w2idx  --load ${model/'{}'/2}.pt --data_ccg CCGMultitask/data/ccg_supertags/
python CCGMultitask/main.py --eval --prog --test_data CCGMultitask/data/ccg_supertags/ccg.23.common --lm_weight 1 --hid_dim 650 --emb_dim 650 --seq_len 40 --n_layers 2 --batch_size 20 --seed 0 --dropout 0.5 --w2idx ${model/'{}'/3}.w2idx  --load ${model/'{}'/3}.pt --data_ccg CCGMultitask/data/ccg_supertags/
python CCGMultitask/main.py --eval --prog --test_data CCGMultitask/data/ccg_supertags/ccg.23.common --lm_weight 1 --hid_dim 650 --emb_dim 650 --seq_len 40 --n_layers 2 --batch_size 20 --seed 0 --dropout 0.5 --w2idx ${model/'{}'/4}.w2idx  --load ${model/'{}'/4}.pt --data_ccg CCGMultitask/data/ccg_supertags/

# Bock 1992 rc/pp
echo 'Bock1992'
python eval_ccgmulti.py --model ${model/'{}'/0},${model/'{}'/1},${model/'{}'/2},${model/'{}'/3},${model/'{}'/4}  --input evalsets/Bock1992/items.ptb.csv --scoring max,sample --paradigm prod --col_tag ccgjoint
python eval_ccgmulti.py --model ${model2/'{}'/0},${model2/'{}'/1},${model2/'{}'/2},${model2/'{}'/3},${model2/'{}'/4}  --input evalsets/Bock1992/items.ptb.csv --scoring max,sample --paradigm prod --col_tag ccglm

echo 'Franck2002'
python eval_ccgmulti.py --model ${model/'{}'/0},${model/'{}'/1},${model/'{}'/2},${model/'{}'/3},${model/'{}'/4}  --input evalsets/Franck2002/items.ptb.csv --scoring max,sample --paradigm prod --col_tag ccgjoint
python eval_ccgmulti.py --model ${model2/'{}'/0},${model2/'{}'/1},${model2/'{}'/2},${model2/'{}'/3},${model2/'{}'/4}  --input evalsets/Franck2002/items.ptb.csv --scoring max,sample --paradigm prod --col_tag ccglm

echo 'Haskell2005'
python eval_ccgmulti.py --model ${model/'{}'/0},${model/'{}'/1},${model/'{}'/2},${model/'{}'/3},${model/'{}'/4}  --input evalsets/HaskellMacdonald2011/items.ptb.csv --scoring max,sample --paradigm prod --col_tag ccgjoint
python eval_ccgmulti.py --model ${model2/'{}'/0},${model2/'{}'/1},${model2/'{}'/2},${model2/'{}'/3},${model2/'{}'/4}  --input evalsets/HaskellMacdonald2011/items.ptb.csv --scoring max,sample --paradigm prod --col_tag ccglm

echo 'HumphreysBock2005'
python eval_ccgmulti.py --model ${model/'{}'/0},${model/'{}'/1},${model/'{}'/2},${model/'{}'/3},${model/'{}'/4}  --input evalsets/HumphreysBock2005/items.csv --scoring max,sample --paradigm prod --col_tag ccgjoint
python eval_ccgmulti.py --model ${model2/'{}'/0},${model2/'{}'/1},${model2/'{}'/2},${model2/'{}'/3},${model2/'{}'/4}  --input evalsets/HumphreysBock2005/items.csv --scoring max,sample --paradigm prod --col_tag ccglm

echo 'ParkerAn2018'
python eval_ccgmulti.py --model ${model/'{}'/0},${model/'{}'/1},${model/'{}'/2},${model/'{}'/3},${model/'{}'/4}  --input evalsets/ParkerAn2018/items.comp.csv --scoring surprisal --paradigm comp --col_tag ccgjoint
python eval_ccgmulti.py --model ${model2/'{}'/0},${model2/'{}'/1},${model2/'{}'/2},${model2/'{}'/3},${model2/'{}'/4}  --input evalsets/ParkerAn2018/items.comp.csv --scoring surprisal --paradigm comp --col_tag ccglm

echo 'Wagers2009'
python eval_ccgmulti.py --model ${model/'{}'/0},${model/'{}'/1},${model/'{}'/2},${model/'{}'/3},${model/'{}'/4}  --input evalsets/Wagers2009/23_illusion/items.csv --scoring surprisal --paradigm comp --col_tag ccgjoint
python eval_ccgmulti.py --model ${model2/'{}'/0},${model2/'{}'/1},${model2/'{}'/2},${model2/'{}'/3},${model2/'{}'/4}  --input evalsets/Wagers2009/23_illusion/items.csv --scoring surprisal --paradigm comp --col_tag ccgjoint
