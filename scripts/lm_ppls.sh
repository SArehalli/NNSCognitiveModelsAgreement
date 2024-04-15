#!/bin/sh

# run from AgreementRNNs/
model2='CCGMultitask/models/wiki_lm/lm_{}_sgd'

echo "test ppls lm"
python CCGMultitask/main.py --eval --prog --data_lm CCGMultitask/data/ --test_data CCGMultitask/data/ccg_supertags/ccg.23.common --lm_weight 1 --hid_dim 650 --emb_dim 650 --seq_len 40 --n_layers 2 --batch_size 32 --seed 0 --dropout 0.5 --w2idx ${model2/'{}'/0}.w2idx  --load ${model2/'{}'/0}.pt --data_ccg CCGMultitask/data/ccg_supertags/ --cuda
python CCGMultitask/main.py --eval --prog --data_lm CCGMultitask/data/  --test_data CCGMultitask/data/ccg_supertags/ccg.23.common --lm_weight 1 --hid_dim 650 --emb_dim 650 --seq_len 40 --n_layers 2 --batch_size 32 --seed 0 --dropout 0.5 --w2idx ${model2/'{}'/1}.w2idx  --load ${model2/'{}'/1}.pt --data_ccg CCGMultitask/data/ccg_supertags/ --cuda
python CCGMultitask/main.py --eval --prog --data_lm CCGMultitask/data/  --test_data CCGMultitask/data/ccg_supertags/ccg.23.common --lm_weight 1 --hid_dim 650 --emb_dim 650 --seq_len 40 --n_layers 2 --batch_size 32 --seed 0 --dropout 0.5 --w2idx ${model2/'{}'/2}.w2idx  --load ${model2/'{}'/2}.pt --data_ccg CCGMultitask/data/ccg_supertags/ --cuda
python CCGMultitask/main.py --eval --prog --data_lm CCGMultitask/data/  --test_data CCGMultitask/data/ccg_supertags/ccg.23.common --lm_weight 1 --hid_dim 650 --emb_dim 650 --seq_len 40 --n_layers 2 --batch_size 32 --seed 0 --dropout 0.5 --w2idx ${model2/'{}'/3}.w2idx  --load ${model2/'{}'/3}.pt --data_ccg CCGMultitask/data/ccg_supertags/ --cuda
python CCGMultitask/main.py --eval --prog --data_lm CCGMultitask/data/  --test_data CCGMultitask/data/ccg_supertags/ccg.23.common --lm_weight 1 --hid_dim 650 --emb_dim 650 --seq_len 40 --n_layers 2 --batch_size 32 --seed 0 --dropout 0.5 --w2idx ${model2/'{}'/4}.w2idx  --load ${model2/'{}'/4}.pt --data_ccg CCGMultitask/data/ccg_supertags/ --cuda
