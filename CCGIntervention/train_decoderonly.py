# Small script to train only the CCG decoder part of an existing MultiTaskModel while freezing the weights of the encoder
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import sys

import numpy as np

import argparse
import tqdm
import pickle

sys.path.append("../CCGMultitask")
from train_joint import train_joint, evaluate_joint
from train_augment import evaluate_lm, evaluate_ccg
from data import joint_tag_lm, tag_dataset, augment_tag_lm, AugmentDataset, BatchSampler
from model import MultiTaskModel

def train_ccgonly(model, optimizer,
                  train_loader, valid_loader_lm,
                  valid_loader_ccg, batch_size,
                  loss_f, clip,
                  max_epochs, save_path, cuda=True, batch_first=False, 
                  early_stop=False, wrap=lambda x: x, patience=3,
                  init_epoch=0):

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.25, verbose=True)
    best_loss = np.inf
    losses = []

    for epoch in wrap(range(max_epochs)):
        total_loss = {"lm":0.0, "ccg":0.0}
        prev_total_loss = {"lm":0.0, "ccg":0.0}
        hidden_ccg = model.init_hidden(batch_size)
        for i, (input_ccg, target_ccg) in enumerate(train_loader):
            model.train()
            loss_ccg = 0
            model.train()

            # Move to GPU if needed
            if cuda:
                input_ccg = input_ccg.cuda()
                target_ccg = target_ccg.cuda()

            # Targets are batch x seq_len, so transpose to seq_len x batch
            # (to match RNN format of seq_len x batch x word/tag_idx)
            input_ccg = input_ccg.transpose(0,1).contiguous()
            target_ccg = target_ccg.transpose(0,1).contiguous()

            ## Detach to enforce bptt limit
            hidden_ccg = (hidden_ccg[0].detach(), hidden_ccg[1].detach())
            
            _, out_ccg, hidden_ccg = model(input_ccg, hidden_ccg)

            # CCG loss

            ## out from seq_len x batch x word_idx -> (seq_len * batch) x word_idx
            ## target from seq_len x batch -> (seq_len * batch)
            loss_ccg = loss_f(out_ccg, target_ccg.view(-1))
                
            # Sum the total loss for this epoch
            total_loss["ccg"] += loss_ccg.item()

            # Compute gradients from the weighted loss
            optimizer.zero_grad()
            model.zero_grad()
            loss_ccg.backward()

            # Clip large gradients
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            # adjust weights
            optimizer.step()


        # Compute validation loss at each interval (a triple of lm loss, ccg loss, and ccg acc)
        valid_loss_lm = evaluate_lm(model, valid_loader_lm, batch_size, loss_f)
        valid_loss_ccg = evaluate_ccg(model, valid_loader_ccg, batch_size, loss_f)
        valid_loss = [valid_loss_lm] + list(valid_loss_ccg)

        # keep list of train/valid losses to return (for plotting)
        losses.append((total_loss["ccg"], valid_loss[1]))

        # Save and record if we've gotten a new best model
        best = " "
        if valid_loss[1] < best_loss:
            best_loss = valid_loss[1]
            torch.save(model.state_dict(), save_path + ".pt")
            optim_state = {"optimizer":optimizer.state_dict(),
                           "epoch":epoch}
            torch.save(optim_state, save_path + ".opt")
            best = "*"

        print(("epoch {:2}{} \t| batch {:2} \t| train lm {:.5f} ccg {:.5f} "
               "\n\t\t|  valid lm nll {:.5f} ppl {:6.2f} \t| valid ccg nll {:.5f} 1-best {:.5f}").format(
              init_epoch + epoch, best, i, total_loss["lm"]/len(train_loader), total_loss["ccg"]/len(train_loader),  
              valid_loss[0], np.exp(valid_loss[0]), valid_loss[1], valid_loss[2]))


        # step the scheduler to see if we reduce the LR
        scheduler.step(valid_loss[1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str)
    parser.add_argument("--data_lm", type=str)
    parser.add_argument("--data_ccg", type=str)
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--save", type=str)
    parser.add_argument("--load", type=str)
    parser.add_argument("--opt", type=str)
    parser.add_argument("--w2idx", type=str)
    parser.add_argument("--sample", type=str)

    parser.add_argument("--hid_dim", type=int)
    parser.add_argument("--emb_dim", type=int)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--n_layers", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)

    parser.add_argument("--dropout", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--wd", type=float)
    parser.add_argument("--clip", type=float)

    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--prog", action="store_true")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    w2idx = None
    with open(args.load + ".w2idx", "rb") as w2idx_f:
        w2idx = pickle.load(w2idx_f)
    
        
    # Train data is handled differently, so that we can loop over one iterator

    train_data = tag_dataset(args.data_ccg + "ccg.02-21.common", args.data_ccg + "categories", 
                             args.seq_len, w2idx=w2idx)
    train_sampler = BatchSampler(train_data, args.batch_size)

    train_loader = DataLoader(train_data, batch_sampler=train_sampler)


    valid_data_lm = augment_tag_lm(args.data_lm + "valid.txt", None, 
                             args.seq_len, w2idx=w2idx)
    valid_sampler_lm = BatchSampler(valid_data_lm, args.batch_size)
    valid_loader_lm = DataLoader(valid_data_lm, batch_sampler=valid_sampler_lm)

    test_data_lm = augment_tag_lm(args.data_lm + "test.txt", None,
                             args.seq_len, w2idx=w2idx)
    test_sampler_lm = BatchSampler(test_data_lm, args.batch_size)
    test_loader_lm = DataLoader(test_data_lm, batch_sampler=test_sampler_lm)

    valid_data_ccg = tag_dataset(args.data_ccg + "ccg.24.common", args.data_ccg + "categories", 
                             args.seq_len, w2idx=w2idx)
    valid_sampler_ccg = BatchSampler(valid_data_ccg, args.batch_size)
    valid_loader_ccg = DataLoader(valid_data_ccg, batch_sampler=valid_sampler_ccg)

    test_data_ccg = tag_dataset(args.data_ccg + "ccg.23.common", args.data_ccg + "categories", 
                             args.seq_len, w2idx=w2idx)
    test_sampler_ccg = BatchSampler(test_data_ccg, args.batch_size)
    test_loader_ccg = DataLoader(test_data_ccg, batch_sampler=test_sampler_ccg)

    mt_model = MultiTaskModel(len(w2idx), args.emb_dim, args.hid_dim, 
                             [len(w2idx), len(train_data.categories)],
                             args.n_layers, dropout = args.dropout)

    device = torch.device("cuda" if args.cuda else "cpu")
    mt_model.load_state_dict(torch.load(args.load + ".pt", map_location=device))
        
    if args.cuda:
        mt_model.cuda()

    if args.opt == "sgd":
        optimizer = optim.SGD(mt_model.decoders[1].parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = optim.Adam(mt_model.decoders[1].parameters(), lr=args.lr, weight_decay=args.wd)
    
    epoch = 0
    
    for param in mt_model.lstm.parameters():
        param.requires_grad = False

    for param in mt_model.decoders[0].parameters():
        param.requires_grad = False

    for name, param in mt_model.named_parameters():
        if param.requires_grad:
                print(name, param.size())

    
    losses = train_ccgonly(mt_model, optimizer,  
                           train_loader, valid_loader_lm, 
                           valid_loader_ccg, args.batch_size,
                           nn.NLLLoss(), args.clip, 
                           args.epochs, args.save, 
                           patience=args.patience, cuda=args.cuda, 
                           init_epoch=epoch, wrap=(tqdm.tqdm if args.prog else (lambda x: x)))

    lm_loss = evaluate_lm(mt_model, test_loader_lm, nn.NLLLoss(), cuda=args.cuda)
    ccg_loss, one_best_ccg = evaluate_ccg(mt_model, test_loader_ccg, args.batch_size,
                                          nn.NLLLoss(), cuda=args.cuda)
    print("TEST: lm ppl {:6.2f} \t| ccg 1-best {:.5f}".format(np.exp(lm_loss), one_best_ccg))


